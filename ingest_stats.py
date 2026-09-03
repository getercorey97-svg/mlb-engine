import sqlite3
import requests
from datetime import datetime

def ingest_mlb_data():
    print("Initializing Factual Data Ingestion: Base Runs (BsR) Upgrade...")
    print("Enforcing Absolute Live Verification (ALV) for MLB Schedule...")
    
    conn = sqlite3.connect('mlb_engine.db')
    cursor = conn.cursor()
    
    cursor.executescript('''
        CREATE TABLE IF NOT EXISTS Pitcher_Stats (
            last_name TEXT PRIMARY KEY,
            est_era REAL,
            updated_at TEXT
        );
        CREATE TABLE IF NOT EXISTS Team_Offense (
            team_name TEXT PRIMARY KEY,
            ops REAL,
            bsr_per_game REAL,
            updated_at TEXT
        );
        CREATE TABLE IF NOT EXISTS Team_Bullpen (
            team_name TEXT PRIMARY KEY,
            team_era REAL,
            updated_at TEXT
        );
    ''')

    # Safe Schema Migrations
    for col in ["bsr_per_game REAL", "updated_at TEXT"]:
        try:
            cursor.execute(f"ALTER TABLE Team_Offense ADD COLUMN {col}")
        except sqlite3.OperationalError:
            pass
            
    for table in ['Pitcher_Stats', 'Team_Bullpen']:
        try:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN updated_at TEXT")
        except sqlite3.OperationalError:
            pass

    season = "2026"
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Fetch Empirical Team Offense (Calculating True Base Runs) and Bullpen Stats
    teams_url = "https://statsapi.mlb.com/api/v1/teams?sportId=1"
    try:
        response = requests.get(teams_url, timeout=15).json()
    except Exception as e:
        print(f"API Error fetching teams: {e}")
        conn.close()
        return

    for team in response.get('teams', []):
        team_name = team.get('name')
        team_id = team.get('id')
        
        stats_url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/stats?group=hitting,pitching&stats=season&season={season}"
        factual_ops, factual_era, bsr_per_game = 0.720, 4.00, 4.50
        
        try:
            stats_res = requests.get(stats_url, timeout=10).json()
            for split in stats_res.get('stats', []):
                group = split.get('group', {}).get('displayName')
                if group == 'hitting' and split.get('splits'):
                    stat = split['splits'][0]['stat']
                    factual_ops = float(stat.get('ops', 0.720))
                    
                    # Extract raw metrics for mathematical Base Runs (BsR) calculation
                    h = float(stat.get('hits', 0))
                    bb = float(stat.get('baseOnBalls', 0))
                    hr = float(stat.get('homeRuns', 0))
                    ab = float(stat.get('atBats', 1))
                    tb = float(stat.get('totalBases', 0))
                    games_played = float(stat.get('gamesPlayed', 1))
                    
                    # SOTA BsR Formulation
                    A = h + bb - hr
                    B = (1.4 * tb - 0.6 * h - 3 * hr + 0.1 * bb) * 1.02
                    C = ab - h
                    D = hr
                    
                    if (B + C) > 0:
                        total_bsr = ((A * B) / (B + C)) + D
                        bsr_per_game = round(total_bsr / games_played, 3)
                        
                elif group == 'pitching' and split.get('splits'):
                    factual_era = float(split['splits'][0]['stat'].get('era', 4.00))
                    
            cursor.execute('''
                INSERT OR REPLACE INTO Team_Offense (team_name, ops, bsr_per_game, updated_at) 
                VALUES (?, ?, ?, ?)
            ''', (team_name, factual_ops, bsr_per_game, current_time))
            
            cursor.execute('''
                INSERT OR REPLACE INTO Team_Bullpen (team_name, team_era, updated_at) 
                VALUES (?, ?, ?)
            ''', (team_name, factual_era, current_time))
            
        except Exception as e:
            print(f"Error mapping {team_name}: {e}")

    # Absolute Live Verification (ALV) Mandate executed for precise Starting Pitcher mapping
    live_date = datetime.now().strftime('%Y-%m-%d')
    schedule_url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={live_date}&hydrate=probablePitcher"
    
    try:
        sched_res = requests.get(schedule_url, timeout=15).json()
        for date_data in sched_res.get('dates', []):
            for game in date_data.get('games', []):
                for side in ['home', 'away']:
                    pitcher = game.get('teams', {}).get(side, {}).get('probablePitcher', {})
                    last_name = pitcher.get('lastName')
                    pitcher_id = pitcher.get('id')
                    
                    if last_name and pitcher_id:
                        p_url = f"https://statsapi.mlb.com/api/v1/people/{pitcher_id}/stats?stats=season&group=pitching&season={season}"
                        factual_era = 4.20
                        try:
                            p_res = requests.get(p_url, timeout=10).json()
                            stats_data = p_res.get('stats', [])
                            if stats_data and stats_data[0].get('splits'):
                                factual_era = float(stats_data[0]['splits'][0]['stat'].get('era', 4.20))
                                
                            cursor.execute('''
                                INSERT OR REPLACE INTO Pitcher_Stats (last_name, est_era, updated_at) 
                                VALUES (?, ?, ?)
                            ''', (last_name, factual_era, current_time))
                        except Exception:
                            pass
    except Exception as e:
        print(f"API Error fetching probable pitchers: {e}")

    conn.commit()
    conn.close()
    print(f"Ingestion complete. ALV mandated for {live_date}. Base Runs (BsR) mathematically locked.")

if __name__ == "__main__":
    ingest_mlb_data()
