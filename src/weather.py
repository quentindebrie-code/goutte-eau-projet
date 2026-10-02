"""Daily weather validation and rain-at-J+1 inference, independent of the UI."""
from datetime import timedelta
import hashlib

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = ["temp_max", "temp_min", "humidity_avg", "pressure_avg", "wind_avg",
            "cloud_cover", "temp_range", "month", "day_of_year"]
RENAME = {"time": "date", "temperature_2m_max": "temp_max",
          "temperature_2m_min": "temp_min", "relative_humidity_2m_mean": "humidity_avg",
          "surface_pressure_mean": "pressure_avg", "wind_speed_10m_max": "wind_avg",
          "cloud_cover_mean": "cloud_cover"}
BOUNDS = {"temp_max": (-60, 60), "temp_min": (-60, 60), "humidity_avg": (0, 100),
          "pressure_avg": (800, 1100), "wind_avg": (0, 250), "cloud_cover": (0, 100)}


def prepare_observations(raw, as_of=None):
    """Return valid features and an explicit quality ledger for every input row.

    The last feature day remains available for inference. Its unknown target is
    NA. Targets join the next calendar day; neither a missing day nor missing
    precipitation is silently converted to a dry day.
    """
    df = raw.rename(columns=RENAME).copy()
    required = {"date", "precipitation_sum", *BOUNDS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError("Colonnes manquantes : " + ", ".join(sorted(missing)))
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    for c in [*BOUNDS, "precipitation_sum"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    invalid = df["date"].isna()
    reasons = pd.Series("", index=df.index)
    def reject(mask, reason):
        nonlocal invalid
        invalid |= mask
        reasons.loc[mask] += reason + "; "
    reject(df["date"].isna(), "date invalide")
    reject(df["date"].duplicated(keep=False), "date dupliquée")
    if as_of is not None:
        reject(~df["date"].between(pd.Timestamp("2020-01-01"), pd.Timestamp(as_of)),
               "date hors du périmètre historique")
    for c, (low, high) in BOUNDS.items():
        reject(~df[c].between(low, high), c + " absent ou hors plage")
    reject(df["temp_min"] > df["temp_max"], "temp_min supérieure à temp_max")
    quality = pd.DataFrame({"date": df["date"], "valide": ~invalid,
                            "mode_degrade": invalid.astype(int), "motif": reasons.str.rstrip("; ")})
    # Rain is validated separately from features; an unknown outcome must not
    # prevent prediction from otherwise valid observations.
    outcomes = df.loc[df["date"].notna() & ~df["date"].duplicated(keep=False),
                      ["date", "precipitation_sum"]].set_index("date")["precipitation_sum"]
    outcomes = outcomes.where(outcomes.between(0, 1000))
    df = df.loc[~invalid].sort_values("date").copy()
    tomorrow_rain = (df["date"] + pd.Timedelta(days=1)).map(outcomes)
    df["rain_tomorrow"] = (tomorrow_rain > .5).astype("Int64").where(tomorrow_rain.notna())
    df["temp_range"] = df["temp_max"] - df["temp_min"]
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    quality["date"] = quality["date"].dt.strftime("%Y-%m-%d")
    return df.reset_index(drop=True), quality.reset_index(drop=True)


def train(df):
    labeled = df.dropna(subset=["rain_tomorrow"]).sort_values("date")
    if len(labeled) < 100:
        raise ValueError("Au moins 100 observations avec cible connue sont nécessaires.")
    x = labeled[FEATURES].to_numpy(dtype=float)
    y = labeled["rain_tomorrow"].to_numpy(dtype=int)
    split = int(len(x) * .8)
    if len(np.unique(y[:split])) != 2:
        raise ValueError("L’entraînement doit contenir les classes pluie et sec.")
    pipeline = Pipeline([("scaler", StandardScaler()), ("clf", RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_leaf=5,
        class_weight="balanced", random_state=42, n_jobs=-1))])
    pipeline.fit(x[:split], y[:split])
    proba = pipeline.predict_proba(x[split:])[:, list(pipeline.classes_).index(1)]
    pred = (proba >= .5).astype(int)
    digest = hashlib.sha256(labeled[["date", *FEATURES, "rain_tomorrow"]]
                            .to_csv(index=False).encode()).hexdigest()[:12]
    metrics = {"accuracy": float(accuracy_score(y[split:], pred)),
               "f1_score": float(f1_score(y[split:], pred, zero_division=0)),
               "roc_auc": float(roc_auc_score(y[split:], proba)) if len(np.unique(y[split:])) == 2 else None,
               "confusion_matrix": confusion_matrix(y[split:], pred, labels=[0, 1]).tolist(),
               "feature_importance": dict(zip(FEATURES, pipeline.named_steps["clf"].feature_importances_.tolist())),
               "n_train": split, "n_test": len(y) - split,
               "test_start": labeled.iloc[split]["date"], "test_end": labeled.iloc[-1]["date"],
               "version_modele": "rf-" + digest}
    return pipeline, metrics


def risk_band(probability):
    if probability is None or not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Probabilité invalide")
    return "faible" if probability < .35 else "modere" if probability < .6 else "eleve"


def forecast(pipeline, df, target_date, quality=None, source="automatique"):
    if source not in {"automatique", "observee", "proxy_saisonnier"}:
        raise ValueError("Source inconnue")
    if df.empty:
        return None
    feature_date = target_date - timedelta(days=1)
    anchor = feature_date.isoformat()
    row = df.loc[df["date"] == anchor]
    if quality is not None:
        state = quality.loc[quality["date"] == anchor]
        if not state.empty and not state["valide"].all():
            return None
    if source == "observee" and row.empty:
        return None
    if not row.empty and source != "proxy_saisonnier":
        feature = row.iloc[0][FEATURES]
        chosen = "observee"
    else:
        # An absent past observation is an unavailable result, not a seasonal
        # fallback. Only future scenarios can fall back automatically.
        if source == "automatique" and anchor <= df["date"].max():
            return None
        monthly = df.loc[pd.to_datetime(df["date"]).dt.month == feature_date.month]
        if monthly.empty:
            return None
        feature = monthly[FEATURES].mean()
        feature["month"] = feature_date.month
        feature["day_of_year"] = feature_date.timetuple().tm_yday
        chosen = "proxy_saisonnier"
    x = np.array([[feature[f] for f in FEATURES]], dtype=float)
    if not np.isfinite(x).all():
        return None
    classes = list(pipeline.classes_)
    probability = float(pipeline.predict_proba(x)[0][classes.index(1)])
    risk = risk_band(probability)
    return {"date_cible": target_date.isoformat(), "date_features": anchor,
            "source_features": chosen, "probability": probability, "risk_level": risk,
            "risk_label": {"faible": "Risque faible de pluie", "modere": "Risque modéré de pluie",
                           "eleve": "Risque élevé de pluie"}[risk],
            "confidence": "non calibrée", "advice": "Consultez une prévision officielle et évaluez vos contraintes avant toute intervention."}
