import sqlite3
import requests
from datetime import datetime, timedelta

def calculate_bullpen_fatigue():
    print("Executing Extraction: Bullpen Availability & Fatigue...")
    
    conn = sqlite3.connect('mlb_engine.db')
    cursor = conn.cursor()
    
    cursor.executescript('''
    CREATE TABLE IF NOT EXISTS Bullpen_Fatigue (
        team_name TEXT PRIMARY KEY,
        fatigue_multiplier REAL
    );
    ''')
    
    # Check the last 3 days of games for reliever usage
    today = datetime.now()
    fatigue_scores = {}
    
    for i in range(1, 4):
        check_date = (today - timedelta(days=i)).strftime('%Y-%m-%d')
        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={check_date}&hydrate=boxscore"
        
        try:
            response = requests.get(url).json()
            for date_data in response.get('dates', []):
                for game in date_data.get('games', []):
                    boxscore = game.get('boxscore', {})
                    for side in ['home', 'away']:
                        team_data = boxscore.get('teams', {}).get(side, {})
                        team_name = team_data.get('team', {}).get('name')
                        pitchers = team_data.get('pitchers', [])
                        
                        # Relievers are pitchers who did not start (index > 0 roughly, or check appearances)
                        # If a reliever pitched in the last 2 days, add fatigue weight
                        for p_id in pitchers[1:]:
                            fatigue_scores[team_name] = fatigue_scores.get(team_name, 1.0) + 0.05
        except Exception:
            continue

    for team, score in fatigue_scores.items():
        # Cap maximum fatigue multiplier at 1.25 (+25% run liability)
        final_fatigue = min(score, 1.25)
        cursor.execute('''
        INSERT OR REPLACE INTO Bullpen_Fatigue (team_name, fatigue_multiplier)
        VALUES (?, ?)
        ''', (team, final_fatigue))
        print(f"Logged Fatigue | {team}: {final_fatigue:.2f}x penalty")

    conn.commit()
    conn.close()
    print("Bullpen fatigue metrics locked.")

if __name__ == "__main__":
    calculate_bullpen_fatigue()
