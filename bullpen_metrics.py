import sqlite3
import requests

def fetch_bullpen_metrics():
    print("Executing Option B: Fetching Official Team Pitching & Bullpen Metrics...")
    
    # Tap directly into the official MLB API for 2026 team pitching stats
    url = "https://statsapi.mlb.com/api/v1/teams/stats?season=2026&group=pitching&stats=season&sportIds=1"
    response = requests.get(url).json()

    conn = sqlite3.connect('mlb_engine.db')
    cursor = conn.cursor()

    # Create the centralized bullpen table
    cursor.executescript('''
    CREATE TABLE IF NOT EXISTS Team_Bullpen (
        team_name TEXT PRIMARY KEY,
        team_era REAL,
        team_whip REAL
    );
    ''')

    print("Mapping bullpen data...")
    
    for split in response['stats'][0]['splits']:
        team_name = split['team']['name']
        era = float(split['stat']['era'])
        whip = float(split['stat']['whip'])
        
        cursor.execute('''
        INSERT OR REPLACE INTO Team_Bullpen (team_name, team_era, team_whip)
        VALUES (?, ?, ?)
        ''', (team_name, era, whip))
        
        print(f"Logged {team_name}: {era:.2f} ERA | {whip:.3f} WHIP")

    conn.commit()
    print("-" * 50)
    print("Bullpen variance locked into mlb_engine.db")
    conn.close()

if __name__ == "__main__":
    fetch_bullpen_metrics()
