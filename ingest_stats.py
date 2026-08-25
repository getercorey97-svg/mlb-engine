import sqlite3
import requests

def ingest_mlb_data():
    print("Initializing Live Data Ingestion from MLB Stats API...")
    
    conn = sqlite3.connect('mlb_engine.db')
    cursor = conn.cursor()
    
    # 1. Ensure tables exist with correct schemas
    cursor.executescript('''
        CREATE TABLE IF NOT EXISTS Pitcher_Stats (
            last_name TEXT PRIMARY KEY,
            est_era REAL,
            updated_at TEXT
        );
        CREATE TABLE IF NOT EXISTS Team_Offense (
            team_name TEXT PRIMARY KEY,
            ops REAL,
            updated_at TEXT
        );
        CREATE TABLE IF NOT EXISTS Team_Bullpen (
            team_name TEXT PRIMARY KEY,
            team_era REAL,
            updated_at TEXT
        );
    ''')

    # 2. Fetch Teams and Default/Live Stats from MLB API
    teams_url = "https://statsapi.mlb.com/api/v1/teams?sportId=1"
    try:
        response = requests.get(teams_url, timeout=15).json()
    except Exception as e:
        print(f"API Error fetching teams: {e}")
        conn.close()
        return

    # Populate Team Offense and Bullpen with team-specific identifiers
    for team in response.get('teams', []):
        team_name = team.get('name')
        
        # Insert baseline/current metrics (can be updated with live stats endpoint if desired)
        cursor.execute('''
            INSERT OR IGNORE INTO Team_Offense (team_name, ops, updated_at)
            VALUES (?, ?, DATETIME('now'))
        ''', (team_name, 0.720)) # Replace 0.720 with dynamic API parse if available
        
        cursor.execute('''
            INSERT OR IGNORE INTO Team_Bullpen (team_name, team_era, updated_at)
            VALUES (?, ?, DATETIME('now'))
        ''', (team_name, 4.00))

    # 3. Extract active probable pitchers from today's schedule and populate Pitcher_Stats
    schedule_url = "https://statsapi.mlb.com/api/v1/schedule?sportId=1&hydrate=probablePitcher"
    try:
        sched_res = requests.get(schedule_url, timeout=15).json()
        for date_data in sched_res.get('dates', []):
            for game in date_data.get('games', []):
                for side in ['home', 'away']:
                    pitcher = game.get('teams', {}).get(side, {}).get('probablePitcher', {})
                    last_name = pitcher.get('lastName')
                    if last_name:
                        cursor.execute('''
                            INSERT OR IGNORE INTO Pitcher_Stats (last_name, est_era, updated_at)
                            VALUES (?, ?, DATETIME('now'))
                        ''', (last_name, 4.00))
    except Exception as e:
        print(f"API Error fetching probable pitchers: {e}")

    conn.commit()
    conn.close()
    print("Data ingestion complete. Database tables populated with active identifiers.")

if __name__ == "__main__":
    ingest_mlb_data()
