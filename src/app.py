"""
app.py — Interface Streamlit autonome — Projet Goutte d'Eau MVP
Bloc 2 — Compétence C15

Version autonome : collecte, entraînement et prédiction intégrés directement.
Aucune dépendance à FastAPI — fonctionne sur Streamlit Cloud.

Lancement :
    streamlit run app.py
"""

from datetime import date, timedelta
from io import StringIO

import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ─── Configuration page ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="Projet Goutte d'Eau",
    page_icon="💧",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── Constantes ───────────────────────────────────────────────────────────────

STATION_ID = "07156"  # Paris-Montsouris
SYNOP_URL  = (
    "https://public.opendatasoft.com/api/explore/v2.1/catalog/datasets/"
    "donnees-synop-essentielles-omm/exports/csv"
)
FEATURES = [
    "temp_max", "temp_min", "humidity_avg",
    "pressure_avg", "wind_avg", "cloud_cover",
    "temp_range", "month", "day_of_year",
]

# ─── Styles CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1F497D, #2E75B6);
        color: white; padding: 2rem; border-radius: 10px;
        margin-bottom: 2rem; text-align: center;
    }
    .main-header h1 { color: white; margin: 0; font-size: 2rem; }
    .main-header p  { color: #B8D4F0; margin: 0.5rem 0 0 0; font-size: 1rem; }
    .risk-card { padding: 1.5rem; border-radius: 10px; text-align: center; margin: 1rem 0; }
    .risk-faible { background: #E2EFDA; border-left: 6px solid #375623; }
    .risk-modere { background: #FFF3CD; border-left: 6px solid #C55A11; }
    .risk-eleve  { background: #FCE4D6; border-left: 6px solid #C00000; }
    .risk-label  { font-size: 1.5rem; font-weight: bold; margin-bottom: 0.5rem; }
    .advice-box  {
        background: #F0F4FF; border-radius: 8px;
        padding: 1rem 1.5rem; margin-top: 1rem;
        font-size: 0.95rem; color: #1F497D;
    }
    .disclaimer { font-size: 0.8rem; color: #888; margin-top: 2rem; }
    .metric-card {
        background: #F8FBFF; border-radius: 8px; padding: 1rem;
        text-align: center; border: 1px solid #D0E4F7;
    }
</style>
""", unsafe_allow_html=True)


# ─── Collecte SYNOP ───────────────────────────────────────────────────────────

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"

@st.cache_data(show_spinner=False)
def load_synop_data():
    params = {
        "latitude": 48.8566,
        "longitude": 2.3522,
        "start_date": "2020-01-01",
        "end_date": str(date.today() - timedelta(days=2)),
        "daily": ",".join([
            "temperature_2m_max", "temperature_2m_min",
            "relative_humidity_2m_mean", "surface_pressure_mean",
            "wind_speed_10m_max", "cloud_cover_mean", "precipitation_sum"
        ]),
        "timezone": "Europe/Paris",
    }
    try:
        r = requests.get(OPEN_METEO_URL, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        return pd.DataFrame(data["daily"])
    except Exception as e:
        st.error(f"Erreur : {e}")
        return pd.DataFrame()


# ─── Nettoyage & agrégation journalière ──────────────────────────────────────

@st.cache_data(show_spinner=False)
def prepare_dataset(df_raw):
    df = df_raw.rename(columns={
        "time": "date",
        "temperature_2m_max": "temp_max",
        "temperature_2m_min": "temp_min",
        "relative_humidity_2m_mean": "humidity_avg",
        "surface_pressure_mean": "pressure_avg",
        "wind_speed_10m_max": "wind_avg",
        "cloud_cover_mean": "cloud_cover",
    })
    df["rain_tomorrow"] = (df["precipitation_sum"].shift(-1) > 0.5).astype(int)
    df = df.iloc[:-1].copy().drop(columns=["precipitation_sum"])
    df["date"] = pd.to_datetime(df["date"])
    df["temp_range"]  = df["temp_max"] - df["temp_min"]
    df["month"]       = df["date"].dt.month
    df["day_of_year"] = df["date"].dt.dayofyear
    df["date"]        = df["date"].dt.strftime("%Y-%m-%d")
    return df.dropna()


# ─── Entraînement ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def train_model(_df):
    X = _df[FEATURES].values
    y = _df["rain_tomorrow"].values
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=100, max_depth=10,
            min_samples_leaf=5, class_weight="balanced",
            random_state=42, n_jobs=-1,
        ))
    ])
    pipeline.fit(X_train, y_train)

    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": round(accuracy_score(y_test, y_pred), 4),
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc":  round(roc_auc_score(y_test, y_proba), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "feature_importance": dict(zip(
            FEATURES,
            pipeline.named_steps["clf"].feature_importances_.round(4).tolist()
        )),
        "n_train": len(X_train),
        "n_test":  len(X_test),
    }
    return pipeline, metrics


# ─── Prédiction ───────────────────────────────────────────────────────────────

def predict(pipeline, df, target_date):
    date_str = target_date.strftime("%Y-%m-%d")
    row = df[df["date"] == date_str]

    if not row.empty:
        feat_row = row.iloc[0]
    else:
        month  = target_date.month
        df_tmp = df.copy()
        df_tmp["date"] = pd.to_datetime(df_tmp["date"])
        monthly = df_tmp[df_tmp["date"].dt.month == month]
        if monthly.empty:
            return None
        feat_row = monthly[FEATURES].mean()
        feat_row["month"]       = month
        feat_row["day_of_year"] = target_date.timetuple().tm_yday

    features = np.array([[feat_row[f] for f in FEATURES]])
    probability = float(pipeline.predict_proba(features)[0][1])

    if probability < 0.35:
        return {
            "risk_level": "faible", "risk_label": "Risque faible de pluie",
            "probability": round(probability, 3),
            "confidence": "haute" if probability < 0.20 else "modérée",
            "advice": "Conditions favorables. Risque de précipitations limité pour demain.",
        }
    elif probability < 0.60:
        return {
            "risk_level": "modere", "risk_label": "Risque modéré de pluie",
            "probability": round(probability, 3), "confidence": "modérée",
            "advice": "Incertitude sur les précipitations. Prévoir un plan de secours.",
        }
    else:
        return {
            "risk_level": "eleve", "risk_label": "Risque élevé de pluie",
            "probability": round(probability, 3),
            "confidence": "haute" if probability > 0.80 else "modérée",
            "advice": "Probabilité élevée de pluie demain. Déconseillé pour les interventions sensibles.",
        }


def risk_color(r): return {"faible":"#375623","modere":"#C55A11","eleve":"#C00000"}.get(r,"#333")
def risk_emoji(r): return {"faible":"✅","modere":"⚠️","eleve":"🚨"}.get(r,"❓")


# ─── Interface ───────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>💧 Projet Goutte d'Eau</h1>
    <p>Estimation du risque de pluie — Paris (75) — MVP BLOC 2</p>
</div>
""", unsafe_allow_html=True)

with st.spinner("⏳ Chargement des données SYNOP Météo France..."):
    df_raw = load_synop_data()

if df_raw.empty:
    st.error("Impossible de charger les données météo.")
    st.stop()

with st.spinner("⚙️ Préparation des données..."):
    df = prepare_dataset(df_raw)

if len(df) < 100:
    st.error("Données insuffisantes.")
    st.stop()

with st.spinner("🤖 Entraînement du modèle..."):
    pipeline, metrics = train_model(df)

st.success(f"✅ Modèle prêt — {len(df)} jours d'observations — Accuracy : {metrics['accuracy']:.1%}")
st.markdown("---")

# Sélecteur de date
st.subheader("📅 Sélectionnez une date")
st.caption("Le modèle estimera le risque de pluie pour le lendemain de la date choisie.")

col1, col2 = st.columns([2, 1])
with col1:
    selected_date = st.date_input(
        "Date", value=date.today() - timedelta(days=1),
        min_value=date(2020, 1, 1),
        max_value=date.today() + timedelta(days=365),
        label_visibility="collapsed",
    )
with col2:
    predict_btn = st.button("🔍 Estimer le risque", use_container_width=True, type="primary")

st.caption(f"Estimation pour : **{(selected_date + timedelta(days=1)).strftime('%A %d %B %Y').capitalize()}**")

# Résultat
if predict_btn:
    result = predict(pipeline, df, selected_date)
    if result is None:
        st.warning("Données insuffisantes pour cette date.")
    else:
        risk  = result["risk_level"]
        proba = result["probability"]
        color = risk_color(risk)

        st.markdown(f"""
        <div class="risk-card risk-{risk}">
            <div class="risk-label" style="color:{color}">
                {risk_emoji(risk)} {result['risk_label']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_g, col_i = st.columns([1.5, 1])
        with col_g:
            st.caption("Probabilité de pluie")
            st.progress(proba)
            st.metric(label="", value=f"{round(proba*100,1)} %")
        with col_i:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.85rem;color:#888">Confiance</div>
                <div style="font-size:1.2rem;font-weight:bold;color:#1F497D">
                    {result['confidence'].upper()}
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="advice-box"><strong>💡 Conseil :</strong> {result['advice']}</div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="disclaimer">
        ℹ️ Estimation basée sur les données SYNOP historiques de Paris (75).
        Ne se substitue pas à une prévision officielle Météo France.
        </div>
        """, unsafe_allow_html=True)

# Section métriques
st.markdown("---")
with st.expander("📊 Performances du modèle — Transparence & Limites"):
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    c2.metric("F1-Score", f"{metrics['f1_score']:.3f}")
    c3.metric("ROC-AUC",  f"{metrics['roc_auc']:.3f}")
    st.caption(f"Entraîné sur {metrics['n_train']} jours — Testé sur {metrics['n_test']} jours")

    st.markdown("### Matrice de confusion")
    cm = metrics["confusion_matrix"]
    st.dataframe(pd.DataFrame(
        cm, index=["Réel : Sec","Réel : Pluie"],
        columns=["Prédit : Sec","Prédit : Pluie"]
    ), use_container_width=True)

    st.markdown("### Importance des variables")
    imp_df = pd.DataFrame(
        metrics["feature_importance"].items(), columns=["Variable","Importance"]
    ).sort_values("Importance", ascending=False)
    imp_df["Variable"] = imp_df["Variable"].str.replace("_"," ").str.title()
    st.bar_chart(imp_df.set_index("Variable"))

    st.markdown("### ⚠️ Limitations")
    st.markdown("""
    - **Périmètre** : Paris (75) uniquement — biais urbain documenté
    - **Dates futures** : moyennes saisonnières utilisées comme proxy
    - **Modèle léger** : Random Forest, pas de deep learning (éco-responsabilité C12)
    - **Ne se substitue pas** à une prévision officielle Météo France
    """)

st.markdown("---")
st.markdown(
    "<div style='text-align:center;font-size:0.8rem;color:#aaa'>"
    "Projet Goutte d'Eau — MVP BLOC 2 — Mastère MTD IA — Institut Léonard de Vinci<br>"
    "Source : SYNOP Météo France (data.gouv.fr) — Modèle : Random Forest (scikit-learn)"
    "</div>", unsafe_allow_html=True,
)
