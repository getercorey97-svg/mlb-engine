import sqlite3
import requests
import concurrent.futures
from datetime import datetime

def fetch_gdelt_media_storm(team_name):
    url = "https://api.gdeltproject.org/api/v2/doc/doc"
    params = {"query": f'"{team_name}" (scandal OR controversy OR fired OR trade OR injury)', "mode": "artlist", "maxrecords": "5", "format": "json"}
    try:
        res = requests.get(url, params=params, timeout=10).json()
        articles = res.get('articles', [])
        tone = sum([float(a.get('tone', 0.0)) for a in articles]) / max(1, len(articles))
        return len(articles), tone
    except: return 0, 0.0

def execute_discovery_ingestion():
    conn = sqlite3.connect('mlb_engine.db')
    cursor = conn.cursor()
    cursor.execute('SELECT game_pk, home_team FROM Daily_Lineups WHERE status != "Final"')
    games = cursor.fetchall()
    
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    team_cache = {}

    print(f"Executing Fast Discovery for {len(games)} Live Games...")
    for game_pk, home_team in games:
        if home_team not in team_cache:
            team_cache[home_team] = fetch_gdelt_media_storm(home_team)
        
        vol, tone = team_cache[home_team]
        cursor.execute('''INSERT OR REPLACE INTO Esoteric_Signals (game_pk, home_media_pressure, home_media_tone, captured_at)
                          VALUES (?, ?, ?, ?)''', (game_pk, vol, tone, timestamp))
    
    conn.commit()
    conn.close()

if __name__ == "__main__":
    execute_discovery_ingestion()
