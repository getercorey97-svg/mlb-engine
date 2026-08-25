import sqlite3
import requests

def fetch_platoon_splits():
    print("Executing Extraction: Platoon Splits (LHP vs RHP)...")
    
    conn = sqlite3.connect('mlb_engine.db')
    cursor = conn.cursor()
    
    cursor.executescript('''
    CREATE TABLE IF NOT EXISTS Team_Platoon_Splits (
        team_name TEXT,
        vs_hand TEXT,
        ops REAL,
        PRIMARY KEY(team_name, vs_hand)
    );
    ''')

    # Fetch hitting stats split by opposing pitcher handedness
    for hand, code in [('LHP', 'left'), ('RHP', 'right')]:
        url = f"https://statsapi.mlb.com/api/v1/teams/stats?season=2026&group=hitting&stats=statSplits&sitCodes=vs{code.capitalize()}&sportIds=1"
        try:
            response = requests.get(url).json()
            for split in response.get('stats', [{}])[0].get('splits', []):
                team_name = split['team']['name']
                ops = float(split['stat'].get('ops', 0.720))
                
                cursor.execute('''
                INSERT OR REPLACE INTO Team_Platoon_Splits (team_name, vs_hand, ops)
                VALUES (?, ?, ?)
                ''', (team_name, hand, ops))
                
                print(f"Logged Platoon | {team_name} vs {hand}: {ops:.3f} OPS")
        except Exception as e:
            print(f"Skipped split {hand} due to API format: {e}")

    conn.commit()
    conn.close()
    print("Platoon splits locked.")

if __name__ == "__main__":
    fetch_platoon_splits()
