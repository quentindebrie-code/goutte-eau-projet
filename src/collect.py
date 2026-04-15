"""
collect.py — Collecte des données météorologiques et stockage SQLite
Projet Goutte d'Eau — MVP BLOC 2

Source : Open-Meteo Historical Weather API (gratuite, sans authentification)
         https://open-meteo.com/en/docs/historical-weather-api
Périmètre : Paris (75) — station de référence lat=48.8566, lon=2.3522

Variables collectées (features retenues) :
  - temperature_2m_max     : température max journalière (°C)
  - temperature_2m_min     : température min journalière (°C)
  - relative_humidity_2m_mean : humidité relative moyenne (%)
  - surface_pressure_mean  : pression atmosphérique moyenne (hPa)
  - wind_speed_10m_max     : vitesse max du vent (km/h)
  - cloud_cover_mean       : nébulosité moyenne (%)
  - precipitation_sum      : cumul précipitations (mm) — pour construire rain_tomorrow

Usage :
    python collect.py                    # collecte 3 ans de données
    python collect.py --start 2020-01-01 --end 2023-12-31
    python collect.py --db ../data/weather.db
"""

import argparse
import sqlite3
import sys
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests


# ─── Configuration ───────────────────────────────────────────────────────────

PARIS_LAT = 48.8566
PARIS_LON = 2.3522
DEFAULT_DB = Path(__file__).parent.parent / "data" / "weather.db"
DEFAULT_START = "2020-01-01"
DEFAULT_END = str(date.today() - timedelta(days=2))  # données dispo J-2

API_URL = "https://archive-api.open-meteo.com/v1/archive"
API_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "relative_humidity_2m_mean",
    "surface_pressure_mean",
    "wind_speed_10m_max",
    "cloud_cover_mean",
    "precipitation_sum",
]


# ─── Base de données ──────────────────────────────────────────────────────────

def init_db(db_path: Path) -> sqlite3.Connection:
    """Crée la base SQLite et la table observations si elle n'existe pas."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS observations (
            date            TEXT PRIMARY KEY,
            temp_max        REAL NOT NULL,
            temp_min        REAL NOT NULL,
            humidity_avg    REAL NOT NULL,
            pressure_avg    REAL NOT NULL,
            wind_avg        REAL NOT NULL,
            cloud_cover     REAL NOT NULL,
            rain_tomorrow   INTEGER NOT NULL
        )
    """)
    conn.commit()
    print(f"[DB] Base initialisée : {db_path}")
    return conn


# ─── Collecte API ─────────────────────────────────────────────────────────────

def fetch_weather_data(start: str, end: str) -> pd.DataFrame:
    """
    Interroge l'API Open-Meteo pour les données historiques de Paris.
    
    Returns:
        DataFrame avec colonnes brutes de l'API
    Raises:
        SystemExit si la requête échoue
    """
    params = {
        "latitude": PARIS_LAT,
        "longitude": PARIS_LON,
        "start_date": start,
        "end_date": end,
        "daily": ",".join(API_VARIABLES),
        "timezone": "Europe/Paris",
    }

    print(f"[API] Requête Open-Meteo : {start} → {end}")
    try:
        response = requests.get(API_URL, params=params, timeout=30)
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        print("[ERREUR] Impossible de joindre l'API. Vérifiez votre connexion internet.")
        sys.exit(1)
    except requests.exceptions.HTTPError as e:
        print(f"[ERREUR] Réponse API invalide : {e}")
        sys.exit(1)

    data = response.json()

    if "daily" not in data:
        print("[ERREUR] Format de réponse inattendu.")
        sys.exit(1)

    df = pd.DataFrame(data["daily"])
    print(f"[API] {len(df)} observations brutes récupérées.")
    return df


# ─── Nettoyage & Feature Engineering ─────────────────────────────────────────

def clean_and_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Nettoie les données brutes et construit la variable cible rain_tomorrow.
    
    - Supprime les lignes avec valeurs manquantes critiques
    - Construit rain_tomorrow : 1 si précipitations J+1 > 0.5mm, 0 sinon
    - Renomme les colonnes pour correspondre au schéma SQLite
    - Eco-responsabilité : supprime les colonnes inutiles à l'entraînement
    
    Returns:
        DataFrame nettoyé avec les colonnes du schéma SQLite
    """
    df = df.rename(columns={"time": "date"})

    # Construction de la variable cible : pluie le lendemain
    df["rain_tomorrow"] = (df["precipitation_sum"].shift(-1) > 0.5).astype(int)

    # Suppression de la dernière ligne (rain_tomorrow non calculable)
    df = df.iloc[:-1].copy()

    # Renommage pour correspondre au schéma SQLite
    df = df.rename(columns={
        "temperature_2m_max": "temp_max",
        "temperature_2m_min": "temp_min",
        "relative_humidity_2m_mean": "humidity_avg",
        "surface_pressure_mean": "pressure_avg",
        "wind_speed_10m_max": "wind_avg",
        "cloud_cover_mean": "cloud_cover",
    })

    # Sélection des colonnes du schéma (suppression de precipitation_sum)
    columns_to_keep = [
        "date", "temp_max", "temp_min", "humidity_avg",
        "pressure_avg", "wind_avg", "cloud_cover", "rain_tomorrow"
    ]
    df = df[columns_to_keep]

    # Suppression des lignes incomplètes
    before = len(df)
    df = df.dropna()
    after = len(df)
    if before != after:
        print(f"[CLEAN] {before - after} lignes supprimées (valeurs manquantes).")

    print(f"[CLEAN] {after} observations nettoyées. Features : {list(df.columns)}")
    return df


# ─── Écriture en base ─────────────────────────────────────────────────────────

def save_to_db(conn: sqlite3.Connection, df: pd.DataFrame) -> int:
    """
    Insère les observations en base (INSERT OR REPLACE pour idempotence).
    
    Returns:
        Nombre de lignes insérées/mises à jour
    """
    df.to_sql("observations", conn, if_exists="append", index=False,
              method=_upsert_method)
    conn.commit()
    count = conn.execute("SELECT COUNT(*) FROM observations").fetchone()[0]
    print(f"[DB] Total observations en base : {count}")
    return len(df)


def _upsert_method(table, conn, keys, data_iter):
    """Méthode d'insertion avec INSERT OR REPLACE pour éviter les doublons."""
    from sqlite3 import OperationalError
    rows = list(data_iter)
    if not rows:
        return
    placeholders = ", ".join(["?"] * len(keys))
    sql = f"INSERT OR REPLACE INTO {table.name} ({', '.join(keys)}) VALUES ({placeholders})"
    try:
        conn.executemany(sql, rows)
    except OperationalError as e:
        print(f"[ERREUR DB] {e}")
        raise


# ─── Rapport de collecte ──────────────────────────────────────────────────────

def print_summary(conn: sqlite3.Connection) -> None:
    """Affiche un résumé de la base après collecte."""
    stats = conn.execute("""
        SELECT 
            COUNT(*) as total,
            MIN(date) as date_min,
            MAX(date) as date_max,
            ROUND(AVG(temp_max), 1) as temp_max_moy,
            ROUND(SUM(rain_tomorrow) * 100.0 / COUNT(*), 1) as pct_pluie
        FROM observations
    """).fetchone()

    print("\n" + "=" * 50)
    print("  RÉSUMÉ DES DONNÉES COLLECTÉES")
    print("=" * 50)
    print(f"  Observations totales : {stats[0]}")
    print(f"  Période              : {stats[1]} → {stats[2]}")
    print(f"  Temp. max moyenne    : {stats[3]} °C")
    print(f"  Jours de pluie       : {stats[4]} %")
    print("=" * 50)


# ─── Point d'entrée ───────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Collecte des données météo Paris (75)")
    parser.add_argument("--start", default=DEFAULT_START, help="Date de début (YYYY-MM-DD)")
    parser.add_argument("--end", default=DEFAULT_END, help="Date de fin (YYYY-MM-DD)")
    parser.add_argument("--db", default=str(DEFAULT_DB), help="Chemin vers la base SQLite")
    args = parser.parse_args()

    db_path = Path(args.db)
    conn = init_db(db_path)

    df_raw = fetch_weather_data(args.start, args.end)
    df_clean = clean_and_transform(df_raw)
    save_to_db(conn, df_clean)
    print_summary(conn)

    conn.close()
    print("\n[OK] Collecte terminée.")


if __name__ == "__main__":
    main()
