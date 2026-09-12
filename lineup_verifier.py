import sqlite3
import requests
from datetime import datetime

def verify_starting_lineups():
    print("Executing Extraction: Starting Lineup Verification...")
    
    today = datetime.now().strftime('%Y-%m-%d')
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today}&hydrate=lineups"
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        print(f"Error fetching schedule data: {e}")
        return

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

    for date_data in data.get('dates', []):
        for game in date_data.get('games', []):
            game_pk = game.get('gamePk')
            if not game_pk:
                continue
            
            teams = game.get('teams', {})
            game_lineups = game.get('lineups', {})
            
            for side in ['away', 'home']:
                side_data = teams.get(side, {})
                team_name = side_data.get('team', {}).get('name')
                if not team_name:
                    continue
                
                lineup = (
                    side_data.get('lineup', [])
                    or game_lineups.get(f"{side}Players", [])
                    or game_lineups.get(side, [])
                    or []
                )
                
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
