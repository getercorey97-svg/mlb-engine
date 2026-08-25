import sqlite3
import requests

def fetch_team_offense():
    print("Executing Option A: Fetching Official Team Offensive Metrics...")
    
    # Tap directly into the official MLB API for 2026 team hitting stats
    url = "https://statsapi.mlb.com/api/v1/teams/stats?season=2026&group=hitting&stats=season&sportIds=1"
    response = requests.get(url).json()

    conn = sqlite3.connect('mlb_engine.db')
    cursor = conn.cursor()

    # Create the centralized offensive table
    cursor.executescript('''
    CREATE TABLE IF NOT EXISTS Team_Offense (
        team_name TEXT PRIMARY KEY,
        ops REAL,
        runs_per_game REAL
    );
    ''')

    print("Mapping offensive data...")
    
    for split in response['stats'][0]['splits']:
        team_name = split['team']['name']
        ops = float(split['stat']['ops'])
        games = int(split['stat']['gamesPlayed'])
        runs = int(split['stat']['runs'])
        
        # Calculate true runs per game to weigh against the opposing pitcher's xERA
        rpg = runs / games if games > 0 else 4.5

        cursor.execute('''
        INSERT OR REPLACE INTO Team_Offense (team_name, ops, runs_per_game)
        VALUES (?, ?, ?)
        ''', (team_name, ops, rpg))
        
        print(f"Logged {team_name}: {rpg:.2f} RPG | {ops:.3f} OPS")

    conn.commit()
    print("-" * 50)
    print("Team offensive firepower locked into mlb_engine.db")
    conn.close()

if __name__ == "__main__":
    fetch_team_offense()
