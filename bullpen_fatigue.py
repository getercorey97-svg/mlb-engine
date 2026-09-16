import sqlite3
import requests
from datetime import datetime, timedelta

MLB_TEAMS = [
    "Arizona Diamondbacks", "Atlanta Braves", "Baltimore Orioles", "Boston Red Sox",
    "Chicago Cubs", "Chicago White Sox", "Cincinnati Reds", "Cleveland Guardians",
    "Colorado Rockies", "Detroit Tigers", "Houston Astros", "Kansas City Royals",
    "Los Angeles Angels", "Los Angeles Dodgers", "Miami Marlins", "Milwaukee Brewers",
    "Minnesota Twins", "New York Mets", "New York Yankees", "Oakland Athletics",
    "Athletics", "Philadelphia Phillies", "Pittsburgh Pirates", "San Diego Padres",
    "San Francisco Giants", "Seattle Mariners", "St. Louis Cardinals", "Tampa Bay Rays",
    "Texas Rangers", "Toronto Blue Jays", "Washington Nationals"
]

def calculate_bullpen_fatigue():
    """Calculates rolling reliever workload from MLB boxscores over the last 3 days."""
    print("=" * 65)
    print("Calculating Rolling Bullpen Fatigue Multipliers...")
    print("=" * 65)
    
    conn = sqlite3.connect('mlb_engine.db', timeout=30)
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA busy_timeout=10000;")

    cursor.executescript('''
    CREATE TABLE IF NOT EXISTS Bullpen_Fatigue (
        team_name TEXT PRIMARY KEY,
        fatigue_multiplier REAL,
        last_updated TEXT
    );
    ''')
    
    now_ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    today = datetime.now()
    fatigue_scores = {team: 1.0 for team in MLB_TEAMS}
    
    # Analyze rolling 3-day workload
    for i in range(1, 4):
        check_date = (today - timedelta(days=i)).strftime('%Y-%m-%d')
        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={check_date}&hydrate=boxscore"
        
        try:
            res = requests.get(url, timeout=12).json()
            for date_data in res.get('dates', []):
                for game in date_data.get('games', []):
                    boxscore = game.get('boxscore', {})
                    teams_data = boxscore.get('teams', {})
                    
                    for side in ['home', 'away']:
                        side_info = teams_data.get(side, {})
                        team_name = side_info.get('team', {}).get('name')
                        pitchers = side_info.get('pitchers', [])
                        
                        # Pitchers after the starter are relievers
                        if len(pitchers) > 1 and team_name in fatigue_scores:
                            decay = 1.0 / (i ** 0.5)
                            reliever_count = len(pitchers) - 1
                            fatigue_scores[team_name] += reliever_count * 0.035 * decay
        except Exception as e:
            print(f"Warning: Failed retrieving workload for {check_date}: {e}")
            continue

    # Commit normalized values (capped at 1.25x max fatigue)
    for team, score in fatigue_scores.items():
        final_fatigue = round(min(score, 1.25), 3)
        cursor.execute('''
        INSERT INTO Bullpen_Fatigue (team_name, fatigue_multiplier, last_updated)
        VALUES (?, ?, ?)
        ON CONFLICT(team_name) DO UPDATE SET
            fatigue_multiplier = excluded.fatigue_multiplier,
            last_updated = excluded.last_updated
        ''', (team, final_fatigue, now_ts))

    conn.commit()
    conn.close()
    print("[SUCCESS] Rolling bullpen fatigue multipliers successfully committed.")

if __name__ == "__main__":
    calculate_bullpen_fatigue()
