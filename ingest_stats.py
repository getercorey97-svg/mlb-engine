import sqlite3
import requests
from datetime import datetime

def ingest_mlb_data():
    print("=" * 65)
    print("Initializing Factual Data Ingestion: Base Runs (BsR) & Bullpen Metrics...")
    print("=" * 65)
    
    conn = sqlite3.connect('mlb_engine.db', timeout=30)
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA busy_timeout=10000;")
    
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
            team_whip REAL DEFAULT 1.25,
            updated_at TEXT
        );
    ''')

    # Safe Schema Migrations
    for col in ["bsr_per_game REAL", "updated_at TEXT"]:
        try:
            cursor.execute(f"ALTER TABLE Team_Offense ADD COLUMN {col}")
        except sqlite3.OperationalError:
            pass
            
    for col in ["team_whip REAL DEFAULT 1.25", "updated_at TEXT"]:
        try:
            cursor.execute(f"ALTER TABLE Team_Bullpen ADD COLUMN {col}")
        except sqlite3.OperationalError:
            pass

    season = str(datetime.now().year)
    current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # Ingest Team Offense and Pitching/Bullpen Metrics
    teams_url = "https://statsapi.mlb.com/api/v1/teams?sportId=1"
    try:
        res = requests.get(teams_url, timeout=15)
        res.raise_for_status()
        response = res.json()
    except Exception as e:
        print(f"API Error fetching teams: {e}")
        conn.close()
        return

    for team in response.get('teams', []):
        team_name = team.get('name')
        team_id = team.get('id')
        if not team_name or not team_id:
            continue
        
        stats_url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/stats?group=hitting,pitching&stats=season&season={season}"
        factual_ops, factual_era, factual_whip, bsr_per_game = 0.720, 4.00, 1.25, 4.50
        
        try:
            stats_req = requests.get(stats_url, timeout=10)
            stats_req.raise_for_status()
            stats_res = stats_req.json()
            
            for split in stats_res.get('stats', []):
                group = split.get('group', {}).get('displayName')
                if group == 'hitting' and split.get('splits'):
                    stat = split['splits'][0]['stat']
                    factual_ops = float(stat.get('ops') or 0.720)
                    
                    h = float(stat.get('hits') or 0)
                    bb = float(stat.get('baseOnBalls') or 0)
                    hr = float(stat.get('homeRuns') or 0)
                    ab = float(stat.get('atBats') or 1)
                    tb = float(stat.get('totalBases') or 0)
                    games_played = float(stat.get('gamesPlayed') or 1)
                    
                    # Base Runs (BsR) Formulation
                    A = h + bb - hr
                    B = (1.4 * tb - 0.6 * h - 3 * hr + 0.1 * bb) * 1.02
                    C = ab - h
                    D = hr
                    
                    if (B + C) > 0 and games_played > 0:
                        total_bsr = ((A * B) / (B + C)) + D
                        bsr_per_game = round(total_bsr / games_played, 3)
                        
                elif group == 'pitching' and split.get('splits'):
                    pstat = split['splits'][0]['stat']
                    factual_era = float(pstat.get('era') or 4.00)
                    factual_whip = float(pstat.get('whip') or 1.25)
                    
            cursor.execute('''
                INSERT INTO Team_Offense (team_name, ops, bsr_per_game, updated_at) 
                VALUES (?, ?, ?, ?)
                ON CONFLICT(team_name) DO UPDATE SET
                    ops = excluded.ops,
                    bsr_per_game = excluded.bsr_per_game,
                    updated_at = excluded.updated_at
            ''', (team_name, factual_ops, bsr_per_game, current_time))
            
            cursor.execute('''
                INSERT INTO Team_Bullpen (team_name, team_era, team_whip, updated_at) 
                VALUES (?, ?, ?, ?)
                ON CONFLICT(team_name) DO UPDATE SET
                    team_era = excluded.team_era,
                    team_whip = excluded.team_whip,
                    updated_at = excluded.updated_at
            ''', (team_name, factual_era, factual_whip, current_time))
            
        except Exception as e:
            print(f"Error mapping {team_name}: {e}")

    # Fetch and Map Probable Starter Performance
    live_date = datetime.now().strftime('%Y-%m-%d')
    schedule_url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={live_date}&hydrate=probablePitcher"
    
    try:
        sched_req = requests.get(schedule_url, timeout=15)
        sched_req.raise_for_status()
        sched_res = sched_req.json()
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
                            p_req = requests.get(p_url, timeout=10)
                            p_req.raise_for_status()
                            p_res = p_req.json()
                            stats_data = p_res.get('stats', [])
                            if stats_data and stats_data[0].get('splits'):
                                factual_era = float(stats_data[0]['splits'][0]['stat'].get('era') or 4.20)
                                
                            cursor.execute('''
                                INSERT INTO Pitcher_Stats (last_name, est_era, updated_at) 
                                VALUES (?, ?, ?)
                                ON CONFLICT(last_name) DO UPDATE SET
                                    est_era = excluded.est_era,
                                    updated_at = excluded.updated_at
                            ''', (last_name, factual_era, current_time))
                        except Exception:
                            pass
    except Exception as e:
        print(f"API Error fetching probable pitchers: {e}")

    conn.commit()
    conn.close()
    print(f"[SUCCESS] Ingestion completed for {live_date}. Offense, Bullpen, and Starter tables synced.")

if __name__ == "__main__":
    ingest_mlb_data()
