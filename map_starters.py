import sqlite3
import requests
from datetime import datetime

def map_probable_starters():
    print("Initiating ALV Probable Starter Mapping...")
    today = datetime.now().strftime('%Y-%m-%d')
    
    # The 'hydrate' parameter forces the MLB API to include pitcher data
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today}&hydrate=probablePitcher"
    
    response = requests.get(url).json()
    
    conn = sqlite3.connect('mlb_engine.db')
    cursor = conn.cursor()
    
    # Safely inject new columns for the starting pitchers
    try:
        cursor.execute("ALTER TABLE Daily_Lineups ADD COLUMN away_pitcher TEXT")
        cursor.execute("ALTER TABLE Daily_Lineups ADD COLUMN home_pitcher TEXT")
    except sqlite3.OperationalError:
        pass # Columns already exist
        
    for date_data in response.get('dates', []):
        for game in date_data.get('games', []):
            game_pk = game['gamePk']
            teams = game.get('teams', {})
            
            # Extract pitcher names (defaults to 'TBD' if unannounced)
            away_pitcher = teams.get('away', {}).get('probablePitcher', {}).get('fullName', 'TBD')
            home_pitcher = teams.get('home', {}).get('probablePitcher', {}).get('fullName', 'TBD')
            
            # Update the existing ALV game records
            cursor.execute('''
                UPDATE Daily_Lineups 
                SET away_pitcher = ?, home_pitcher = ? 
                WHERE game_pk = ?
            ''', (away_pitcher, home_pitcher, game_pk))
            
            print(f"Matchup {game_pk} Mapped: {away_pitcher} (Away) vs {home_pitcher} (Home)")

    conn.commit()
    print("-" * 50)
    print("Starting pitchers successfully bolted into mlb_engine.db")
    conn.close()

if __name__ == "__main__":
    map_probable_starters()
