import sqlite3
import requests
from datetime import datetime

NOAA_KP_URL = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"

def fetch_geomagnetic_kp():
    """Pulls live NOAA planetary geomagnetic Kp-index."""
    try:
        res = requests.get(NOAA_KP_URL, timeout=8).json()
        if len(res) > 1:
            return float(res[-1][1])
    except Exception:
        pass
    return 2.0

def fetch_gdelt_tone(team_name):
    """Scrapes news sentiment and media pressure via GDELT."""
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": f'"{team_name}" (injury OR controversy OR bullpen)',
        "mode": "artlist",
        "maxrecords": "5",
        "format": "json"
    }
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    try:
        res = requests.get(url, params=params, headers=headers, timeout=8).json()
        articles = res.get('articles', [])
        tone = sum([float(a.get('tone', 0.0)) for a in articles]) / max(1, len(articles))
        return len(articles), round(tone, 2)
    except Exception:
        return 0, 0.0

def execute_discovery_ingestion():
    """Ingests NOAA geomagnetic and GDELT sentiment variables."""
    print(f"[{datetime.now()}] Ingesting Discovery Variables (NOAA + GDELT)...")
    conn = sqlite3.connect('mlb_engine.db', timeout=30)
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA busy_timeout=10000;")

    cursor.executescript('''
    CREATE TABLE IF NOT EXISTS Esoteric_Signals (
        game_pk INTEGER PRIMARY KEY,
        geomagnetic_kp REAL DEFAULT 2.0,
        solar_xray_flux REAL DEFAULT 1.0,
        home_media_pressure INTEGER DEFAULT 0,
        home_media_tone REAL DEFAULT 0.0,
        roster_birthday_active INTEGER DEFAULT 0,
        captured_at TEXT
    );
    ''')

    kp_index = fetch_geomagnetic_kp()

    cursor.execute('SELECT game_pk, home_team FROM Daily_Lineups WHERE status != "Final"')
    games = cursor.fetchall()
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    team_cache = {}
    for game_pk, home_team in games:
        if home_team not in team_cache:
            team_cache[home_team] = fetch_gdelt_tone(home_team)

        vol, tone = team_cache[home_team]
        cursor.execute('''
        INSERT OR REPLACE INTO Esoteric_Signals 
        (game_pk, geomagnetic_kp, solar_xray_flux, home_media_pressure, home_media_tone, roster_birthday_active, captured_at)
        VALUES (?, ?, 1.0, ?, ?, 0, ?)
        ''', (game_pk, kp_index, vol, tone, now_str))

    conn.commit()
    conn.close()
    print(f"Discovery variables logged across {len(games)} active matchups.")

# Aliases
discover_signals = execute_discovery_ingestion

if __name__ == "__main__":
    execute_discovery_ingestion()
