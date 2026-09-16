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
    
    for date_data in data.get('dates', []):
        for game in date_data.get('games', []):
            game_pk = game.get('gamePk')
            if not game_pk:
                continue
            
            teams = game.get('teams', {})
            game_lineups = game.get('lineups', {})
            
            away_data = teams.get('away', {})
            home_data = teams.get('home', {})
            
            away_lineup = away_data.get('lineup', []) or game_lineups.get("awayPlayers", []) or game_lineups.get("away", []) or []
            home_lineup = home_data.get('lineup', []) or game_lineups.get("homePlayers", []) or game_lineups.get("home", []) or []
            
            # A game's lineup status is only confirmed if BOTH teams have submitted their 9 batters
            if len(away_lineup) >= 9 and len(home_lineup) >= 9:
                status = "Confirmed"
            else:
                status = "Pending/TBD"
            
            cursor.execute('''
            UPDATE Daily_Lineups 
            SET lineup_status = ? 
            WHERE game_pk = ?
            ''', (status, game_pk))
            
            print(f"Game {game_pk} Lineup Status: {status} (Away: {len(away_lineup)}, Home: {len(home_lineup)})")

    conn.commit()
    conn.close()
    print("Lineup verification status locked in Daily_Lineups table.")

if __name__ == "__main__":
    verify_starting_lineups()
