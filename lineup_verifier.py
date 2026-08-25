import sqlite3
import requests
from datetime import datetime

def verify_starting_lineups():
    print("Executing Extraction: Starting Lineup Verification...")
    
    today = datetime.now().strftime('%Y-%m-%d')
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today}&hydrate=lineups"
    
    response = requests.get(url).json()
    
    conn = sqlite3.connect('mlb_engine.db')
    cursor = conn.cursor()
    
    cursor.executescript('''
    CREATE TABLE IF NOT EXISTS Verified_Lineups (
        game_pk INTEGER,
        team_name TEXT,
        lineup_status TEXT,
        PRIMARY KEY(game_pk, team_name)
    );
    ''')

    for date_data in response.get('dates', []):
        for game in date_data.get('games', []):
            game_pk = game['gamePk']
            teams = game.get('teams', {})
            
            for side in ['away', 'home']:
                team_name = teams[side]['team']['name']
                lineup = teams[side].get('lineup', [])
                
                status = "Confirmed" if len(lineup) >= 9 else "Pending/TBD"
                
                cursor.execute('''
                INSERT OR REPLACE INTO Verified_Lineups (game_pk, team_name, lineup_status)
                VALUES (?, ?, ?)
                ''', (game_pk, team_name, status))
                
                print(f"Game {game_pk} | {team_name} Lineup: {status} ({len(lineup)} batters posted)")

    conn.commit()
    conn.close()
    print("Lineup verification status locked.")

if __name__ == "__main__":
    verify_starting_lineups()
