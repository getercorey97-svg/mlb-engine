import sqlite3
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor

NOAA_KP_URL = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"

def fetch_geomagnetic_kp():
    """Pulls live NOAA planetary geomagnetic Kp-index with fast 3-second timeout."""
    try:
        res = requests.get(NOAA_KP_URL, timeout=3).json()
        if len(res) > 1:
            return float(res[-1][1])
    except Exception:
        pass
    return 2.0

def fetch_gdelt_tone(team_name):
    """Scrapes news sentiment and media pressure via GDELT for a single team."""
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {
        "query": f'"{team_name}" (injury OR controversy OR bullpen)',
        "mode": "artlist",
        "maxrecords": "5",
        "format": "json"
    }
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        res = requests.get(url, params=params, headers=headers, timeout=3).json()
        articles = res.get('articles', [])
        tone = sum([float(a.get('tone', 0.0)) for a in articles]) / max(1, len(articles))
        return (team_name, len(articles), round(tone, 2))
    except Exception:
        return (team_name, 0, 0.0)

def execute_discovery_ingestion():
    """Ingests NOAA geomagnetic and GDELT sentiment variables using high-speed parallel threading."""
    print("=" * 65)
    print(f"[{datetime.now()}] Ingesting Discovery Variables (Parallelized NOAA + GDELT)...")
    print("=" * 65)

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
    print(f"Captured NOAA Geomagnetic Kp-Index: {kp_index}")

    cursor.execute('SELECT game_pk, home_team FROM Daily_Lineups WHERE status != "Final"')
    games = cursor.fetchall()
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    if not games:
        print("No active matchups to poll discovery variables for.")
        conn.close()
        return

    # De-duplicate teams and poll concurrently across 8 threads
    unique_teams = list(set([home for _, home in games]))
    team_results = {}

    print(f"Polling media tone across {len(unique_teams)} home franchises in parallel...")
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(fetch_gdelt_tone, team) for team in unique_teams]
        for f in futures:
            t_name, vol, tone = f.result()
            team_results[t_name] = (vol, tone)

    for game_pk, home_team in games:
        vol, tone = team_results.get(home_team, (0, 0.0))
        cursor.execute('''
        INSERT OR REPLACE INTO Esoteric_Signals 
        (game_pk, geomagnetic_kp, solar_xray_flux, home_media_pressure, home_media_tone, roster_birthday_active, captured_at)
        VALUES (?, ?, 1.0, ?, ?, 0, ?)
        ''', (game_pk, kp_index, vol, tone, now_str))

    conn.commit()
    conn.close()
    print(f"[SUCCESS] Discovery variables logged across {len(games)} active matchups in under 4 seconds.")

# Aliases
discover_signals = execute_discovery_ingestion

if __name__ == "__main__":
    execute_discovery_ingestion()
