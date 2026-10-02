# 💧 Projet Goutte d'Eau — MVP

> **Estimation du risque de pluie à Paris (75) via Machine Learning**  
> MVP réalisé dans le cadre du Mastère Management de la Transformation Digitale en IA  
> Institut Léonard de Vinci — BLOC 2 : Conception et développement de l'architecture fonctionnelle

---

## Présentation

Le Projet Goutte d'Eau (PGE) est développé pour **France Météo**, établissement public national. Face à l'augmentation des phénomènes pluvieux extrêmes liés au changement climatique, ce MVP démontre la faisabilité d'une chaîne complète :

**Collecte de données → Base de données → Modèle ML → API → Interface agriculteur**

La variable cible est une **classification binaire** : *y aura-t-il de la pluie demain ?*

---

## Architecture

```
goutte-eau-projet/
├── src/
│   ├── collect.py        # Collecte des données météo (Open-Meteo API) → SQLite
│   ├── train.py          # Entraînement Random Forest + évaluation + export .pkl
│   ├── main.py           # API FastAPI — endpoint /predict?date=YYYY-MM-DD
│   └── app.py            # Interface Streamlit pour les agriculteurs
├── data/
│   └── weather.db        # Base SQLite (générée par collect.py — non versionnée)
├── model/
│   ├── model.pkl         # Modèle entraîné (généré par train.py — non versionné)
│   └── evaluation.json   # Métriques d'évaluation
├── tests/
│   └── test_api.py       # Tests unitaires (pytest)
├── .gitignore
├── requirements.txt
├── runtime.txt
└── README.md
```

---

## Installation

### Prérequis

- Python 3.11+
- pip

### Étapes

```bash
# 1. Cloner le repository
git clone https://github.com/quentindebrie-code/goutte-eau-projet.git
cd goutte-eau-projet

# 2. Créer un environnement virtuel
python -m venv venv
source venv/bin/activate        # Linux/macOS
# ou : venv\Scripts\activate    # Windows

# 3. Installer les dépendances
pip install -r requirements.txt
```

---

## Utilisation

### Étape 1 — Collecte des données

```bash
cd src
python collect.py
```

Collecte ~4 ans de données historiques météo pour Paris (75) depuis [Open-Meteo](https://open-meteo.com) et les stocke dans `data/weather.db`.

**Options disponibles :**

```bash
python collect.py --start 2020-01-01 --end 2024-12-31
python collect.py --db ../data/weather.db
```

### Étape 2 — Entraînement du modèle

```bash
python train.py
```

Entraîne un Random Forest Classifier, affiche les métriques d'évaluation, et exporte le modèle dans `model/model.pkl`.

**Options disponibles :**

```bash
python train.py --n-estimators 200 --test-size 0.2
```

### Étape 3 — Lancer l'API FastAPI

```bash
cd ..
uvicorn src.main:app --reload
```

L'API est accessible sur http://localhost:8000  
La documentation Swagger est sur http://localhost:8000/docs

**Exemple de requête :**

```bash
curl "http://localhost:8000/predict?date=2024-07-14"
```

**Réponse :**

```json
{
  "date": "2024-07-14",
  "risk_level": "modere",
  "risk_label": "Risque modéré de pluie",
  "probability": 0.483,
  "confidence": "modérée",
  "advice": "Incertitude sur les précipitations de demain. Prévoir un plan de secours pour les interventions critiques.",
  "model_version": "1.0.0",
  "disclaimer": "Estimation basée sur les données historiques de Paris (75)..."
}
```

### Étape 4 — Lancer l'interface Streamlit

L’interface collecte les observations et entraîne son modèle directement. Elle fonctionne sans lancer FastAPI, sans base SQLite et sans fichier `model.pkl`.

```bash
streamlit run src/app.py
```

L'interface est accessible sur http://localhost:8501

Sur Streamlit Community Cloud, le point d’entrée est `app.py` à la racine du dépôt. Les dépendances sont dans `requirements.txt`.

### Correctifs B3 — qualité, historique et traçabilité

- Les précipitations inconnues restent inconnues. La cible utilise la date calendaire J+1 ; un jour absent ne décale pas la cible sur le prochain jour disponible.
- Les dates dupliquées, dates hors du périmètre 2020–aujourd’hui, valeurs manquantes et valeurs physiques aberrantes sont exclues. Le contrôle affiche les motifs et suspend la prévision observée correspondante.
- La dernière observation valide reste utilisable pour estimer son lendemain, même si la cible n’est pas encore connue. Seules les cibles connues participent à l’entraînement et à l’évaluation.
- Le cache du modèle dépend des données. « Actualiser les observations » recharge le flux ; une modification des données produit une nouvelle version `rf-<empreinte>`.
- Chaque estimation affiche date cible, date des variables et source. Une observation historique absente est indisponible. Un scénario futur peut utiliser un proxy saisonnier, explicitement signalé.
- Les bandes UX restent fixes : faible sous 0,35 ; modéré de 0,35 inclus à 0,60 exclu ; élevé dès 0,60. La confiance du score est indiquée « non calibrée ».
- L’historique propose un filtre de période, un tableau alternatif au graphique et un export CSV. Les tableaux de qualité, confusion et importance permettent de lire les valeurs sans graphique.

L’application utilise **Open-Meteo Archive**, aux coordonnées de Paris ; il ne s’agit pas d’un flux SYNOP ni d’une station instrumentale Paris-Montsouris. `wind_avg` conserve le nom historique du schéma et contient le **maximum journalier** du vent, en km/h.

Cette démonstration est publique et ne comporte pas d’authentification applicative ni de RLS. Les écrans de connexion, rôles et incidents du prototype Figma constituent des spécifications ; leur présence dans la maquette ne prouve pas une implémentation en production. Le rapport Power BI doit être validé séparément dans son environnement natif.

---

## Endpoints API

| Méthode | Route                      | Description                                     |
|---------|----------------------------|-------------------------------------------------|
| `GET`   | `/`                        | Accueil API                                     |
| `GET`   | `/health`                  | Statut API + modèle + base de données           |
| `GET`   | `/predict?date=YYYY-MM-DD` | Estimation du risque de pluie                   |
| `GET`   | `/metrics`                 | Métriques d'évaluation du modèle                |
| `GET`   | `/docs`                    | Documentation Swagger UI                        |

---

## Modèle Machine Learning

### Choix technique

| Critère        | Valeur                                        |
|----------------|-----------------------------------------------|
| Algorithme     | Random Forest Classifier (scikit-learn)        |
| Variable cible | `rain_tomorrow` (classification binaire)       |
| Features       | 9 variables (température, humidité, pression, vent, nébulosité, features dérivées) |
| Split          | Chronologique (pas de shuffle — respect de la temporalité) |
| Validation     | Cross-validation stratifiée 5-fold             |

### Justification éco-responsable (C12)

Le deep learning a été **explicitement exclu** : il nécessite un entraînement énergivore (GPU) et produit des modèles "boîtes noires" difficiles à justifier dans un contexte institutionnel. Le Random Forest offre un excellent compromis performance / empreinte carbone / interprétabilité.

### Features utilisées

| Feature       | Description                  | Justification                       |
|---------------|------------------------------|-------------------------------------|
| `temp_max`    | Température max (°C)         | Instabilité atmosphérique           |
| `temp_min`    | Température min (°C)         | Amplitude thermique                 |
| `humidity_avg`| Humidité relative (%)        | Saturation en vapeur d'eau          |
| `pressure_avg`| Pression atm. (hPa)         | Prédicteur classique de changement météo |
| `wind_avg`    | Vitesse max vent (km/h)     | Advection de masses d'air           |
| `cloud_cover` | Nébulosité (%)              | Signal précurseur de précipitations |
| `temp_range`  | Amplitude thermique          | Feature dérivée : instabilité convective |
| `month`       | Mois de l'année              | Saisonnalité                        |
| `day_of_year` | Jour de l'année              | Position dans le cycle annuel       |

---

## Tests

```bash
# Depuis la racine du projet
pytest tests/ -v
pytest tests/ -v --cov=src --cov-report=html
```

---

## Indicateurs qualité

| Dimension       | Indicateur              | Seuil MVP |
|-----------------|-------------------------|-----------|
| Performance ML  | Accuracy                | ≥ 65%     |
| Performance ML  | F1-Score                | ≥ 0.60    |
| Performance ML  | ROC-AUC                 | ≥ 0.65    |
| Disponibilité   | Uptime API              | ≥ 95%     |
| Réactivité      | Temps réponse /predict  | < 500 ms  |
| Qualité code    | Couverture tests        | ≥ 60%     |

---

## Source de données

- **Open-Meteo Historical API** — https://open-meteo.com
  - Gratuite, sans authentification
  - Données historiques depuis 1940
  - Station de référence : Paris (lat=48.8566, lon=2.3522)
- **Référence sujet** : données Infoclimat / SYNOP Météo France via [data.gouv.fr](https://www.data.gouv.fr/fr/reuses/api-de-recuperation-de-donnees-meteorologiques-du-reseau-infoclimat-static-et-de-meteofrance-synop/)

---

## Éco-responsabilité

- Modèle léger (Random Forest) : aucun GPU requis
- Dataset épuré : 9 features uniquement
- SQLite sans serveur dédié
- FastAPI asynchrone : consommation mémoire réduite
- Hébergement recommandé : Infomaniak ou Scaleway (100% énergies renouvelables)

---

## Limites connues

- Périmètre Paris (75) uniquement — biais urbain documenté
- Pour les dates futures, l'API utilise des moyennes saisonnières (proxy imparfait)
- Ce MVP ne se substitue pas à une prévision météorologique officielle Météo France

---

## Auteur

**Quentin Debrie**  
Mastère Management de la Transformation Digitale en IA  
Institut Léonard de Vinci — 2025-2026

---

## Licence

MIT License — Voir fichier [LICENSE](LICENSE)
