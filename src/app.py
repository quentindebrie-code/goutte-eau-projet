"""
app.py — Interface Streamlit autonome — Projet Goutte d'Eau MVP
Blocs 2 et 3 — démonstration publique, qualité et traçabilité

Version autonome : collecte, entraînement et prédiction intégrés directement.
Aucune dépendance à FastAPI — fonctionne sur Streamlit Cloud.

Lancement :
    streamlit run app.py
"""

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
from src.weather import FEATURES, prepare_observations, train, forecast
import pandas as pd
import requests
import streamlit as st

# ─── Configuration page ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="Projet Goutte d'Eau",
    page_icon="💧",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ─── Constantes ───────────────────────────────────────────────────────────────

TODAY = datetime.now(ZoneInfo("Europe/Paris")).date()

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
    .risk-modere { background: #FFF3CD; border-left: 6px solid #7B4B00; }
    .risk-eleve  { background: #FCE4D6; border-left: 6px solid #C00000; }
    .risk-label  { font-size: 1.5rem; font-weight: bold; margin-bottom: 0.5rem; }
    .advice-box  {
        background: #F0F4FF; border-radius: 8px;
        padding: 1rem 1.5rem; margin-top: 1rem;
        font-size: 0.95rem; color: #1F497D;
    }
    .disclaimer { font-size: 0.8rem; color: #466271; margin-top: 2rem; }
    .metric-card {
        background: #F8FBFF; border-radius: 8px; padding: 1rem;
        text-align: center; border: 1px solid #D0E4F7;
    }
</style>
""", unsafe_allow_html=True)


# ─── Collecte Open-Meteo Archive ─────────────────────────────────────────────

OPEN_METEO_URL = "https://archive-api.open-meteo.com/v1/archive"

@st.cache_data(ttl=3600, show_spinner=False)
def load_weather_data(as_of):
    params = {"latitude": 48.8566, "longitude": 2.3522, "start_date": "2020-01-01",
              "end_date": str(as_of - timedelta(days=2)), "timezone": "Europe/Paris",
              "daily": ",".join(["temperature_2m_max", "temperature_2m_min",
                  "relative_humidity_2m_mean", "surface_pressure_mean",
                  "wind_speed_10m_max", "cloud_cover_mean", "precipitation_sum"])}
    response = requests.get(OPEN_METEO_URL, params=params, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if "daily" not in payload or not payload["daily"].get("time"):
        raise ValueError("Le flux ne contient pas d’observations journalières.")
    return pd.DataFrame(payload["daily"])


@st.cache_data(show_spinner=False)
def prepare_dataset(df_raw, as_of):
    return prepare_observations(df_raw, as_of=as_of)


@st.cache_resource(show_spinner=False, max_entries=4)
def train_model(df):
    # df must be hashed: an underscore prefix would silently reuse an old model.
    return train(df)


def predict(pipeline, df, feature_date, quality=None):
    return forecast(pipeline, df, feature_date + timedelta(days=1), quality)


def risk_color(r): return {"faible":"#375623","modere":"#7B4B00","eleve":"#C00000"}.get(r,"#333")
def risk_emoji(r): return {"faible":"✅","modere":"⚠️","eleve":"🚨"}.get(r,"❓")


# ─── Interface ───────────────────────────────────────────────────────────────

st.markdown("""
<div class="main-header">
    <h1>💧 Projet Goutte d'Eau</h1>
    <p>Estimation du risque de pluie — Paris (75) — B3 · démonstration publique</p>
</div>
""", unsafe_allow_html=True)

if st.button("Actualiser les observations"):
    load_weather_data.clear()
    st.rerun()
try:
    with st.spinner("Chargement du flux Open-Meteo Archive…"):
        df_raw = load_weather_data(TODAY)
        df, df_quality = prepare_dataset(df_raw, TODAY)
    with st.spinner("Vérification du modèle…"):
        pipeline, metrics = train_model(df)
except (requests.RequestException, ValueError, KeyError) as exc:
    st.error("Données indisponibles : la prévision est suspendue.")
    st.caption(str(exc))
    st.stop()

latest = date.fromisoformat(df["date"].max())
age = (TODAY - latest).days
st.caption(f"Dernière observation valide : {latest:%d/%m/%Y} · ancienneté {age} jours · modèle {metrics['version_modele']}")
if age > 4:
    st.warning("Flux ancien. Les scénarios saisonniers ne constituent pas une prévision actuelle officielle.")

st.success(f"✅ Modèle prêt — {len(df)} jours d'observations — Accuracy : {metrics['accuracy']:.1%}")
st.markdown("---")

# Sélecteur de date
st.subheader("📅 Sélectionnez une date")
st.caption("Le modèle estimera le risque de pluie pour le lendemain de la date choisie.")

col1, col2 = st.columns([2, 1])
with col1:
    selected_date = st.date_input(
        "Date", value=TODAY - timedelta(days=1),
        min_value=date(2020, 1, 1),
        max_value=TODAY + timedelta(days=365),
        label_visibility="collapsed",
    )
with col2:
    predict_btn = st.button("🔍 Estimer le risque", use_container_width=True, type="primary")

st.caption(f"Estimation pour : **{(selected_date + timedelta(days=1)):%d/%m/%Y}**")

# Résultat
if predict_btn:
    result = predict(pipeline, df, selected_date, df_quality)
    if result is None:
        st.error("Prévision indisponible : observation historique absente ou invalide. Aucune absence n’est convertie en risque faible.")
    else:
        st.caption(f"Date cible : {result['date_cible']} · variables du {result['date_features']} · source : {result['source_features']}")
        if result["source_features"] == "proxy_saisonnier":
            st.info("Scénario saisonnier : moyennes historiques du mois, sans prévision météo de ce jour.")
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
            st.metric(label="Probabilité de pluie > 0,5 mm", value=f"{round(proba*100,1)} %")
        with col_i:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.85rem;color:#466271">Confiance du score</div>
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
        ℹ️ Estimation basée sur les données Open-Meteo Archive historiques de Paris (75).
        Ne se substitue pas à une prévision officielle Météo France.
        </div>
        """, unsafe_allow_html=True)

# Quality and history: complete value alternatives and actionable empty states.
st.markdown("---")
with st.expander("Qualité et disponibilité des observations"):
    invalid = df_quality.loc[~df_quality["valide"]]
    st.metric("Lignes valides", f"{df_quality['valide'].mean():.1%}")
    if invalid.empty:
        st.success("Aucune anomalie sur les variables contrôlées du flux reçu.")
    else:
        st.warning(f"{len(invalid)} lignes exclues. La prévision observée de ces jours est suspendue.")
        st.dataframe(invalid, hide_index=True, use_container_width=True)
    st.caption("Dates, doublons, valeurs manquantes, plages physiques et cohérence min/max. Une précipitation J+1 absente conserve une cible inconnue.")
    st.dataframe(df_quality, hide_index=True, use_container_width=True)

with st.expander("Historique météo et export"):
    start = st.date_input("Début de période", latest - timedelta(days=30))
    end = st.date_input("Fin de période", latest)
    if start > end:
        st.error("La date de début doit précéder la date de fin.")
    else:
        history = df.loc[df["date"].between(start.isoformat(), end.isoformat())]
        if history.empty:
            st.info("Aucun résultat pour cette période. Modifiez les dates.")
        else:
            st.line_chart(history.set_index("date")[["temp_max", "temp_min"]])
            st.caption("Températures en °C. Alternative : tableau complet ci-dessous. wind_avg est le maximum journalier du vent en km/h.")
            st.dataframe(history, hide_index=True, use_container_width=True)
            st.download_button("Télécharger la période CSV", history.to_csv(index=False).encode("utf-8-sig"), "observations_periode.csv", "text/csv")

# Section métriques
st.markdown("---")
with st.expander("📊 Performances du modèle — Transparence & Limites"):
    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{metrics['accuracy']:.1%}")
    c2.metric("F1-Score", f"{metrics['f1_score']:.3f}")
    c3.metric("ROC-AUC", "Indisponible" if metrics["roc_auc"] is None else f"{metrics['roc_auc']:.3f}")
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
    st.dataframe(imp_df, hide_index=True, use_container_width=True)
    st.caption("Alternative au graphique : tableau ci-dessus. Importance globale, sans relation causale individuelle.")

    st.markdown("### ⚠️ Limitations")
    st.markdown("""
    - **Périmètre** : Paris (75) uniquement — biais urbain documenté
    - **Dates futures** : moyennes saisonnières utilisées comme proxy
    - **Modèle léger** : Random Forest, pas de deep learning (éco-responsabilité C12)
    - **Ne se substitue pas** à une prévision officielle Météo France
    """)

st.markdown("---")
st.markdown(
    "<div style='text-align:center;font-size:0.8rem;color:#466271'>"
    "Projet Goutte d'Eau — B3 · démonstration publique — Mastère MTD IA — Institut Léonard de Vinci<br>"
    "Source : Open-Meteo Archive · Pluie à J+1 strictement > 0,5 mm — Modèle : Random Forest (scikit-learn)"
    "</div>", unsafe_allow_html=True,
)
