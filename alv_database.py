import sqlite3
import requests
from datetime import datetime

def execute_unified_alv():
    print("Rebuilding Database Schema to enforce Pitcher Columns...")
    conn = sqlite3.connect('mlb_engine.db')
    cursor = conn.cursor()

    # Nuke the old table and rebuild it with the exact columns needed
    cursor.executescript('''
    DROP TABLE IF EXISTS Daily_Lineups;
    CREATE TABLE Daily_Lineups (
        game_pk INTEGER PRIMARY KEY,
        game_date TEXT,
        away_team TEXT,
        home_team TEXT,
        away_pitcher TEXT,
        home_pitcher TEXT,
        status TEXT
    );
    ''')
    
    today = datetime.now().strftime('%Y-%m-%d')
    
    # The 'hydrate' parameter pulls the schedule AND the pitchers in one single call
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today}&hydrate=probablePitcher"
    
    print(f"Executing Unified ALV Mandate for {today}...")
    response = requests.get(url).json()

    for date_data in response.get('dates', []):
        for game in date_data.get('games', []):
            game_pk = game['gamePk']
            status = game['status']['abstractGameState']
            
            teams = game.get('teams', {})
            away = teams['away']['team']['name']
            home = teams['home']['team']['name']
            
            away_pitcher = teams.get('away', {}).get('probablePitcher', {}).get('fullName', 'TBD')
            home_pitcher = teams.get('home', {}).get('probablePitcher', {}).get('fullName', 'TBD')
            
            cursor.execute('''
            INSERT INTO Daily_Lineups (game_pk, game_date, away_team, home_team, away_pitcher, home_pitcher, status)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (game_pk, today, away, home, away_pitcher, home_pitcher, status))
            
            print(f"Verified & Mapped: {away} ({away_pitcher}) @ {home} ({home_pitcher})")

    conn.commit()
    print("-" * 50)
    print("Unified ALV sync complete. Schema perfectly locked.")
    conn.close()

if __name__ == "__main__":
    execute_unified_alv()

