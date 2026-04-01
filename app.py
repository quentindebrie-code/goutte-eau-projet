"""
app.py — Interface Streamlit — Projet Goutte d'Eau MVP
Bloc 2 — Compétence C15

Interface agriculteur : information immédiatement actionnable, sans jargon technique.

Prérequis : API FastAPI lancée sur http://localhost:8000
    uvicorn main:app --reload  (depuis src/)

Lancement :
    streamlit run app.py
"""

from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

# ─── Configuration page ───────────────────────────────────────────────────────

st.set_page_config(
    page_title="Projet Goutte d'Eau",
    page_icon="💧",
    layout="centered",
    initial_sidebar_state="collapsed",
)

API_BASE = "http://localhost:8000"

# ─── Styles CSS ───────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* En-tête */
    .main-header {
        background: linear-gradient(135deg, #1F497D, #2E75B6);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
    }
    .main-header h1 { color: white; margin: 0; font-size: 2rem; }
    .main-header p { color: #B8D4F0; margin: 0.5rem 0 0 0; font-size: 1rem; }

    /* Cartes de risque */
    .risk-card {
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        margin: 1rem 0;
    }
    .risk-faible  { background: #E2EFDA; border-left: 6px solid #375623; }
    .risk-modere  { background: #FFF3CD; border-left: 6px solid #C55A11; }
    .risk-eleve   { background: #FCE4D6; border-left: 6px solid #C00000; }

    /* Textes */
    .risk-label  { font-size: 1.5rem; font-weight: bold; margin-bottom: 0.5rem; }
    .risk-proba  { font-size: 2.5rem; font-weight: bold; }
    .advice-box  {
        background: #F0F4FF;
        border-radius: 8px;
        padding: 1rem 1.5rem;
        margin-top: 1rem;
        font-size: 0.95rem;
        color: #1F497D;
    }
    .disclaimer  { font-size: 0.8rem; color: #888; margin-top: 2rem; }
    .metric-card {
        background: #F8FBFF;
        border-radius: 8px;
        padding: 1rem;
        text-align: center;
        border: 1px solid #D0E4F7;
    }
</style>
""", unsafe_allow_html=True)


# ─── Fonctions utilitaires ───────────────────────────────────────────────────

def check_api_health() -> dict | None:
    """Vérifie que l'API FastAPI est accessible."""
    try:
        r = requests.get(f"{API_BASE}/health", timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def get_prediction(target_date: str) -> dict | None:
    """Interroge l'API pour obtenir la prédiction."""
    try:
        r = requests.get(f"{API_BASE}/predict", params={"date": target_date}, timeout=10)
        if r.status_code == 200:
            return r.json()
        st.error(f"Erreur API ({r.status_code}) : {r.json().get('detail', 'Erreur inconnue')}")
        return None
    except requests.exceptions.ConnectionError:
        st.error("Impossible de joindre l'API. Vérifiez que FastAPI est lancée (uvicorn main:app).")
        return None


def get_metrics() -> dict | None:
    """Récupère les métriques du modèle depuis l'API."""
    try:
        r = requests.get(f"{API_BASE}/metrics", timeout=5)
        return r.json() if r.status_code == 200 else None
    except Exception:
        return None


def risk_color(risk_level: str) -> str:
    return {"faible": "#375623", "modere": "#C55A11", "eleve": "#C00000"}.get(risk_level, "#333")


def risk_emoji(risk_level: str) -> str:
    return {"faible": "✅", "modere": "⚠️", "eleve": "🚨"}.get(risk_level, "❓")


def probability_gauge(probability: float, risk_level: str) -> go.Figure:
    """Crée une jauge Plotly pour afficher la probabilité de pluie."""
    color = risk_color(risk_level)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=round(probability * 100, 1),
        number={"suffix": "%", "font": {"size": 36, "color": color}},
        title={"text": "Probabilité de pluie", "font": {"size": 14, "color": "#555"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#888"},
            "bar": {"color": color, "thickness": 0.3},
            "bgcolor": "white",
            "borderwidth": 1,
            "bordercolor": "#ddd",
            "steps": [
                {"range": [0, 35],  "color": "#E2EFDA"},
                {"range": [35, 60], "color": "#FFF3CD"},
                {"range": [60, 100],"color": "#FCE4D6"},
            ],
            "threshold": {
                "line": {"color": color, "width": 4},
                "thickness": 0.75,
                "value": probability * 100,
            },
        },
    ))
    fig.update_layout(
        height=250,
        margin={"t": 40, "b": 20, "l": 40, "r": 40},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def feature_importance_chart(importances: dict) -> go.Figure:
    """Graphique en barres de l'importance des variables."""
    items = sorted(importances.items(), key=lambda x: x[1])
    labels = [k.replace("_", " ").title() for k, _ in items]
    values = [v for _, v in items]

    fig = go.Figure(go.Bar(
        x=values, y=labels,
        orientation="h",
        marker_color="#2E75B6",
        text=[f"{v:.3f}" for v in values],
        textposition="outside",
    ))
    fig.update_layout(
        title="Importance des variables du modèle",
        xaxis_title="Importance relative",
        height=350,
        margin={"t": 50, "b": 20, "l": 150, "r": 60},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# ─── Interface principale ─────────────────────────────────────────────────────

# En-tête
st.markdown("""
<div class="main-header">
    <h1>💧 Projet Goutte d'Eau</h1>
    <p>Estimation du risque de pluie — Paris (75) — MVP BLOC 2</p>
</div>
""", unsafe_allow_html=True)

# Statut API
health = check_api_health()
if health is None:
    st.error("⛔ API FastAPI non accessible. Lancez : `uvicorn main:app --reload` depuis le dossier `src/`")
    st.stop()

if not health.get("model_loaded"):
    st.warning("⚠️ Modèle non chargé. Lancez : `python train.py` depuis le dossier `src/`")
    st.stop()

if not health.get("db_available"):
    st.warning("⚠️ Base de données non trouvée. Lancez : `python collect.py` depuis le dossier `src/`")
    st.stop()

st.success("✅ API connectée — Modèle chargé — Base de données disponible")

st.markdown("---")

# ─── Sélecteur de date + prédiction ──────────────────────────────────────────

st.subheader("📅 Sélectionnez une date")
st.caption("L'API estimera le risque de pluie pour le lendemain de la date choisie.")

col1, col2 = st.columns([2, 1])
with col1:
    selected_date = st.date_input(
        "Date de référence",
        value=date.today() - timedelta(days=1),
        min_value=date(2020, 1, 1),
        max_value=date.today() + timedelta(days=365),
        label_visibility="collapsed",
    )
with col2:
    predict_btn = st.button("🔍 Estimer le risque", use_container_width=True, type="primary")

st.caption(f"Estimation pour : **{(selected_date + timedelta(days=1)).strftime('%A %d %B %Y').capitalize()}**")

# ─── Résultat ─────────────────────────────────────────────────────────────────

if predict_btn:
    with st.spinner("Calcul en cours..."):
        result = get_prediction(selected_date.strftime("%Y-%m-%d"))

    if result:
        risk = result["risk_level"]
        proba = result["probability"]
        emoji = risk_emoji(risk)

        # Carte de risque principale
        css_class = f"risk-{risk}"
        color = risk_color(risk)
        st.markdown(f"""
        <div class="risk-card {css_class}">
            <div class="risk-label" style="color:{color}">{emoji} {result['risk_label']}</div>
        </div>
        """, unsafe_allow_html=True)

        # Jauge + métriques côte à côte
        col_gauge, col_info = st.columns([1.5, 1])
        with col_gauge:
            st.plotly_chart(probability_gauge(proba, risk), use_container_width=True)
        with col_info:
            st.markdown("<br><br>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class="metric-card">
                <div style="font-size:0.85rem; color:#888">Probabilité</div>
                <div style="font-size:2rem; font-weight:bold; color:{color}">{round(proba*100,1)}%</div>
            </div>
            <br>
            <div class="metric-card">
                <div style="font-size:0.85rem; color:#888">Confiance</div>
                <div style="font-size:1.2rem; font-weight:bold; color:#1F497D">{result['confidence'].upper()}</div>
            </div>
            """, unsafe_allow_html=True)

        # Conseil opérationnel
        st.markdown(f"""
        <div class="advice-box">
            <strong>💡 Conseil :</strong> {result['advice']}
        </div>
        """, unsafe_allow_html=True)

        # Disclaimer
        st.markdown(f"""
        <div class="disclaimer">ℹ️ {result['disclaimer']}</div>
        """, unsafe_allow_html=True)

# ─── Section "En savoir plus" (métriques modèle) ─────────────────────────────

st.markdown("---")
with st.expander("📊 Performances du modèle — Transparence & Limites"):
    metrics = get_metrics()
    if metrics:
        st.markdown("### Métriques d'évaluation (jeu de test)")

        col1, col2, col3 = st.columns(3)
        col1.metric("Accuracy", f"{metrics.get('accuracy', 'N/A'):.1%}")
        col2.metric("F1-Score", f"{metrics.get('f1_score', 'N/A'):.3f}")
        col3.metric("ROC-AUC",  f"{metrics.get('roc_auc', 'N/A'):.3f}")

        # Matrice de confusion
        cm = metrics.get("confusion_matrix")
        if cm:
            st.markdown("### Matrice de confusion")
            cm_df = pd.DataFrame(
                cm,
                index=["Réel : Sec", "Réel : Pluie"],
                columns=["Prédit : Sec", "Prédit : Pluie"]
            )
            st.dataframe(cm_df, use_container_width=True)

        # Feature importance
        imp = metrics.get("feature_importance")
        if imp:
            st.plotly_chart(feature_importance_chart(imp), use_container_width=True)

        st.markdown("### ⚠️ Limitations du modèle")
        st.markdown("""
        - **Périmètre géographique** : Paris (75) uniquement. Biais urbain (îlots de chaleur, artificialisation des sols).
        - **Données futures** : pour les dates sans données historiques, l'API utilise des moyennes saisonnières (proxy imparfait).
        - **Granularité** : prévision journalière uniquement (pas horaire).
        - **Modèle** : Random Forest, choisi pour sa légèreté et son interprétabilité. Un modèle plus complexe pourrait améliorer les performances au coût d'une empreinte carbone plus élevée.
        - **Ce MVP ne se substitue pas à une prévision météorologique officielle Météo France.**
        """)
    else:
        st.warning("Métriques non disponibles.")

# ─── Footer ───────────────────────────────────────────────────────────────────

st.markdown("---")
st.markdown(
    "<div style='text-align:center; font-size:0.8rem; color:#aaa'>"
    "Projet Goutte d'Eau — MVP BLOC 2 — Mastère MTD IA — Institut Léonard de Vinci<br>"
    "Données : Open-Meteo (open-source) — Modèle : Random Forest (scikit-learn)"
    "</div>",
    unsafe_allow_html=True,
)
