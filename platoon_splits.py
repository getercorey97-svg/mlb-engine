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

    # Fetch hitting stats split by opposing pitcher handedness (vl = vs Left, vr = vs Right)
    for hand, code in [('LHP', 'vl'), ('RHP', 'vr')]:
        url = f"https://statsapi.mlb.com/api/v1/teams/stats?season=2026&group=hitting&stats=statSplits&sitCodes={code}&sportIds=1"
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            stats_list = data.get('stats', [])
            splits = stats_list[0].get('splits', []) if stats_list else []
            
            for split in splits:
                team_name = split.get('team', {}).get('name')
                if not team_name:
                    continue
                
                ops_val = split.get('stat', {}).get('ops')
                ops = float(ops_val) if ops_val is not None else 0.720
                
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
