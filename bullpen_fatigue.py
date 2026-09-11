import sqlite3
import requests
from datetime import datetime, timedelta

MLB_TEAMS = [
    "Arizona Diamondbacks", "Atlanta Braves", "Baltimore Orioles", "Boston Red Sox",
    "Chicago Cubs", "Chicago White Sox", "Cincinnati Reds", "Cleveland Guardians",
    "Colorado Rockies", "Detroit Tigers", "Houston Astros", "Kansas City Royals",
    "Los Angeles Angels", "Los Angeles Dodgers", "Miami Marlins", "Milwaukee Brewers",
    "Minnesota Twins", "New York Mets", "New York Yankees", "Oakland Athletics",
    "Philadelphia Phillies", "Pittsburgh Pirates", "San Diego Padres", "San Francisco Giants",
    "Seattle Mariners", "St. Louis Cardinals", "Tampa Bay Rays", "Texas Rangers",
    "Toronto Blue Jays", "Washington Nationals"
]

def calculate_bullpen_fatigue():
    """Calculates rolling reliever workload from MLB boxscores over the last 3 days."""
    print("Calculating Rolling Bullpen Fatigue Multipliers...")
    
    conn = sqlite3.connect('mlb_engine.db', timeout=30)
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA busy_timeout=10000;")

    cursor.executescript('''
    CREATE TABLE IF NOT EXISTS Bullpen_Fatigue (
        team_name TEXT PRIMARY KEY,
        fatigue_multiplier REAL
    );
    ''')
    
    today = datetime.now()
    fatigue_scores = {team: 1.0 for team in MLB_TEAMS}
    
    for i in range(1, 4):
        check_date = (today - timedelta(days=i)).strftime('%Y-%m-%d')
        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={check_date}&hydrate=boxscore"
        
        try:
            res = requests.get(url, timeout=10).json()
            for date_data in res.get('dates', []):
                for game in date_data.get('games', []):
                    boxscore = game.get('boxscore', {})
                    for side in ['home', 'away']:
                        team_data = boxscore.get('teams', {}).get(side, {})
                        team_name = team_data.get('team', {}).get('name')
                        pitchers = team_data.get('pitchers', [])
                        
                        # Pitchers after starter are bullpen arms
                        if len(pitchers) > 1 and team_name in fatigue_scores:
                            decay = 1.0 / (i ** 0.5)
                            fatigue_scores[team_name] += (len(pitchers) - 1) * 0.035 * decay
        except Exception:
            continue

    for team, score in fatigue_scores.items():
        final_fatigue = round(min(score, 1.25), 3)
        cursor.execute('''
        INSERT OR REPLACE INTO Bullpen_Fatigue (team_name, fatigue_multiplier)
        VALUES (?, ?)
        ''', (team, final_fatigue))

    conn.commit()
    conn.close()
    print("Bullpen fatigue multipliers committed.")

if __name__ == "__main__":
    calculate_bullpen_fatigue()
