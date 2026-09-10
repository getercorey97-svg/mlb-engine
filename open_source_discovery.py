import sqlite3
import requests
import math
from datetime import datetime, timedelta

NOAA_KP_URL = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
NOAA_FLARES_URL = "https://services.swpc.noaa.gov/json/goes/primary/xrays-1-day.json"
GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"

def fetch_noaa_space_weather():
    """Scrapes public NOAA feeds for geomagnetic Kp index and active solar flares."""
    kp_val = 2.0
    solar_flux = 1.0
    try:
        kp_res = requests.get(NOAA_KP_URL, timeout=8).json()
        if len(kp_res) > 1:
            kp_val = float(kp_res[-1][1])
    except Exception:
        pass

    try:
        flare_res = requests.get(NOAA_FLARES_URL, timeout=8).json()
        if flare_res:
            solar_flux = float(flare_res[-1].get('flux', 1.0e-6)) * 1e6
    except Exception:
        pass
        
    return kp_val, solar_flux

def fetch_gdelt_media_storm(team_name):
    """Scrapes GDELT 2.0 to quantify media turbulence and tone over the last 48 hours."""
    params = {
        "query": f'"{team_name}" (scandal OR controversy OR fired OR trade OR injury)',
        "mode": "artlist",
        "maxrecords": "15",
        "format": "json"
    }
    try:
        res = requests.get(GDELT_DOC_URL, params=params, timeout=10).json()
        articles = res.get('articles', [])
        media_volume = len(articles)
        tone_score = sum([float(a.get('tone', 0.0)) for a in articles]) / max(1, media_volume)
        return media_volume, tone_score
    except Exception:
        return 0, 0.0

def fetch_roster_birthdays(game_pk):
    """Checks the MLB Stats API to detect active player birthdays in the starting lineup."""
    url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
    birthday_detected = 0
    today_str = datetime.now().strftime('%m-%d')
    try:
        res = requests.get(url, timeout=10).json()
        players = res.get('teams', {}).get('home', {}).get('players', {})
        for _, pdata in players.items():
            bday = pdata.get('person', {}).get('birthDate', '')
            if bday.endswith(today_str):
                birthday_detected = 1
                break
    except Exception:
        pass
    return birthday_detected

def execute_discovery_ingestion():
    """Aggregates all open-source signals and writes them to the Esoteric_Signals table."""
    conn = sqlite3.connect('mlb_engine.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Esoteric_Signals (
            game_pk INTEGER PRIMARY KEY,
            geomagnetic_kp REAL,
            solar_xray_flux REAL,
            home_media_pressure INTEGER,
            home_media_tone REAL,
            roster_birthday_active INTEGER,
            captured_at TEXT
        )
    ''')

    cursor.execute('SELECT game_pk, home_team FROM Daily_Lineups')
    games = cursor.fetchall()
    kp, flux = fetch_noaa_space_weather()
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    for game_pk, home_team in games:
        media_vol, tone = fetch_gdelt_media_storm(home_team)
        bday_flag = fetch_roster_birthdays(game_pk)
        
        cursor.execute('''
            INSERT OR REPLACE INTO Esoteric_Signals 
            (game_pk, geomagnetic_kp, solar_xray_flux, home_media_pressure, home_media_tone, roster_birthday_active, captured_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (game_pk, kp, flux, media_vol, tone, bday_flag, timestamp))

    conn.commit()
    conn.close()
    print("[SUCCESS] Open-source environmental, media, and biological data captured.")

if __name__ == "__main__":
    execute_discovery_ingestion()
