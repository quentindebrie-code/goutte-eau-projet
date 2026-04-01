"""
train.py — Entraînement, évaluation et export du modèle ML
Projet Goutte d'Eau — MVP BLOC 2

Modèle : Random Forest Classifier (scikit-learn)
  - Choix justifié : léger, interprétable (feature importance), pas de GPU requis
  - Pas de deep learning (C12 — éco-responsabilité)
  - Variable cible : rain_tomorrow (classification binaire)

Usage :
    python train.py                      # entraîne avec les paramètres par défaut
    python train.py --db ../data/weather.db --model ../model/model.pkl
    python train.py --n-estimators 200 --test-size 0.2

Sorties :
    - model/model.pkl          : modèle entraîné sérialisé
    - model/evaluation.json    : métriques d'évaluation
"""

import argparse
import json
import sqlite3
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

warnings.filterwarnings("ignore")


# ─── Configuration ───────────────────────────────────────────────────────────

DEFAULT_DB    = Path(__file__).parent.parent / "data" / "weather.db"
DEFAULT_MODEL = Path(__file__).parent.parent / "model" / "model.pkl"
DEFAULT_EVAL  = Path(__file__).parent.parent / "model" / "evaluation.json"

FEATURES = [
    "temp_max",
    "temp_min",
    "humidity_avg",
    "pressure_avg",
    "wind_avg",
    "cloud_cover",
    # Features dérivées (feature engineering)
    "temp_range",      # amplitude thermique journalière
    "month",           # saisonnalité
    "day_of_year",     # position dans l'année
]

TARGET = "rain_tomorrow"


# ─── Chargement des données ───────────────────────────────────────────────────

def load_data(db_path: Path) -> pd.DataFrame:
    """Charge les observations nettoyées depuis SQLite."""
    if not db_path.exists():
        raise FileNotFoundError(
            f"Base de données introuvable : {db_path}\n"
            "Lancez d'abord : python collect.py"
        )

    conn = sqlite3.connect(db_path)
    df = pd.read_sql("SELECT * FROM observations ORDER BY date", conn)
    conn.close()

    print(f"[DATA] {len(df)} observations chargées depuis {db_path}")
    return df


# ─── Feature Engineering ─────────────────────────────────────────────────────

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Construit les features dérivées à partir des données brutes.
    
    Eco-responsabilité : uniquement des features pertinentes et légères,
    calculées sans bibliothèque externe lourde.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    # Amplitude thermique journalière (signal fort pour instabilité atmosphérique)
    df["temp_range"] = df["temp_max"] - df["temp_min"]

    # Saisonnalité (corrélée aux patterns pluvieux à Paris)
    df["month"] = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear

    # Vérification des features disponibles
    missing = [f for f in FEATURES if f not in df.columns]
    if missing:
        raise ValueError(f"Features manquantes dans la base : {missing}")

    print(f"[FEATURES] {len(FEATURES)} features construites : {FEATURES}")
    return df


# ─── Entraînement ────────────────────────────────────────────────────────────

def train_model(
    df: pd.DataFrame,
    n_estimators: int = 100,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple:
    """
    Entraîne le Random Forest Classifier avec cross-validation.
    
    Returns:
        (pipeline, X_test, y_test, feature_names)
    """
    X = df[FEATURES].values
    y = df[TARGET].values

    # Split temporel (pas de shuffle — respect de la chronologie)
    split_idx = int(len(X) * (1 - test_size))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"[TRAIN] Split : {len(X_train)} train / {len(X_test)} test")
    print(f"[TRAIN] Distribution cible (train) : {np.bincount(y_train.astype(int))}")

    # Pipeline : StandardScaler + Random Forest
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=10,           # limite la complexité (éco + sur-apprentissage)
            min_samples_leaf=5,     # régularisation
            class_weight="balanced",# gestion déséquilibre pluie/sec
            random_state=random_state,
            n_jobs=-1,
        ))
    ])

    # Cross-validation stratifiée (5 folds)
    print("[TRAIN] Cross-validation 5-fold en cours...")
    cv = StratifiedKFold(n_splits=5, shuffle=False)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="f1")
    print(f"[TRAIN] CV F1-scores : {cv_scores.round(3)} | Moyenne : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Entraînement final
    pipeline.fit(X_train, y_train)
    print("[TRAIN] Modèle entraîné.")

    return pipeline, X_test, y_test, FEATURES


# ─── Évaluation ──────────────────────────────────────────────────────────────

def evaluate_model(pipeline, X_test: np.ndarray, y_test: np.ndarray, feature_names: list) -> dict:
    """
    Évalue le modèle et retourne les métriques sous forme de dictionnaire.
    Affiche un rapport complet dans la console.
    """
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    accuracy  = accuracy_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    roc_auc   = roc_auc_score(y_test, y_proba)
    cm        = confusion_matrix(y_test, y_pred).tolist()

    # Feature importance
    clf = pipeline.named_steps["clf"]
    importances = dict(zip(feature_names, clf.feature_importances_.round(4).tolist()))
    importances = dict(sorted(importances.items(), key=lambda x: -x[1]))

    metrics = {
        "accuracy":        round(accuracy, 4),
        "f1_score":        round(f1, 4),
        "roc_auc":         round(roc_auc, 4),
        "confusion_matrix": cm,
        "feature_importance": importances,
        "n_test_samples":  len(X_test),
        "class_labels":    ["pas de pluie (0)", "pluie (1)"],
    }

    # Affichage console
    print("\n" + "=" * 55)
    print("  RAPPORT D'ÉVALUATION DU MODÈLE")
    print("=" * 55)
    print(f"  Accuracy     : {accuracy:.4f}  (seuil MVP ≥ 0.65)")
    print(f"  F1-Score     : {f1:.4f}  (seuil MVP ≥ 0.60)")
    print(f"  ROC-AUC      : {roc_auc:.4f}  (seuil MVP ≥ 0.65)")
    print("\n  Matrice de confusion :")
    print(f"  TN={cm[0][0]}  FP={cm[0][1]}")
    print(f"  FN={cm[1][0]}  TP={cm[1][1]}")
    print("\n  Feature Importance :")
    for feat, imp in importances.items():
        bar = "█" * int(imp * 40)
        print(f"  {feat:<20} {imp:.4f}  {bar}")
    print("\n" + classification_report(y_test, y_pred,
          target_names=["Sec", "Pluie"]))
    print("=" * 55)

    # Alertes si sous les seuils
    if accuracy < 0.65:
        print("⚠️  [AVERTISSEMENT] Accuracy < seuil MVP (0.65). Considérer plus de données.")
    if f1 < 0.60:
        print("⚠️  [AVERTISSEMENT] F1-Score < seuil MVP (0.60).")

    return metrics


# ─── Sauvegarde ──────────────────────────────────────────────────────────────

def save_model(pipeline, model_path: Path) -> None:
    """Sérialise le pipeline entraîné avec joblib."""
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    size_kb = model_path.stat().st_size / 1024
    print(f"[SAVE] Modèle sauvegardé : {model_path} ({size_kb:.1f} Ko)")


def save_evaluation(metrics: dict, eval_path: Path) -> None:
    """Sauvegarde les métriques d'évaluation en JSON."""
    eval_path.parent.mkdir(parents=True, exist_ok=True)
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)
    print(f"[SAVE] Métriques sauvegardées : {eval_path}")


# ─── Point d'entrée ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Entraînement du modèle Goutte d'Eau")
    parser.add_argument("--db",           default=str(DEFAULT_DB))
    parser.add_argument("--model",        default=str(DEFAULT_MODEL))
    parser.add_argument("--eval",         default=str(DEFAULT_EVAL))
    parser.add_argument("--n-estimators", default=100, type=int)
    parser.add_argument("--test-size",    default=0.2, type=float)
    args = parser.parse_args()

    # 1. Chargement
    df = load_data(Path(args.db))

    # 2. Feature engineering
    df = build_features(df)

    # 3. Entraînement
    pipeline, X_test, y_test, feature_names = train_model(
        df,
        n_estimators=args.n_estimators,
        test_size=args.test_size,
    )

    # 4. Évaluation
    metrics = evaluate_model(pipeline, X_test, y_test, feature_names)

    # 5. Sauvegarde
    save_model(pipeline, Path(args.model))
    save_evaluation(metrics, Path(args.eval))

    print("\n[OK] Entraînement terminé.")


if __name__ == "__main__":
    main()
