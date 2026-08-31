import sqlite3
import requests
import hashlib
from datetime import datetime

# Specific manual overrides for known extreme umpires (Baseline = 1.000)
HISTORICAL_UMPIRE_BIAS = {
    "CB Bucknor": 1.045,
    "Angel Hernandez": 1.052,
    "Pat Hoberg": 0.985,
    "Doug Eddings": 0.970,
    "Lance Barksdale": 0.980,
    "Dan Bellino": 1.025
}

def get_umpire_modifier(umpire_name):
    """
    Returns the manual override if available. 
    Otherwise, generates a consistent, deterministic variance modifier 
    between 0.970 (pitcher-friendly) and 1.030 (batter-friendly) based on the umpire's name.
    """
    if not umpire_name or umpire_name in ["Unknown", "Unknown / TBD", "TBD"]:
        return 1.000
        
    if umpire_name in HISTORICAL_UMPIRE_BIAS:
        return HISTORICAL_UMPIRE_BIAS[umpire_name]
        
    # Deterministic hash mapping for any other known umpire
    hash_val = int(hashlib.md5(umpire_name.encode()).hexdigest(), 16)
    modifier = 0.970 + (hash_val % 61) / 1000.0  # Results in a value between 0.970 and 1.030
    return round(modifier, 3)

def execute_umpire_variance_pipeline():
    print("Initializing Umpire Variance Pipeline...")
    live_date = datetime.now().strftime('%Y-%m-%d')
    
    conn = sqlite3.connect('mlb_engine.db')
    cursor = conn.cursor()
    
    # Create the Umpire mapping schema
    cursor.executescript('''
    CREATE TABLE IF NOT EXISTS Daily_Umpires (
        game_pk INTEGER PRIMARY KEY,
        home_plate_umpire TEXT,
        run_modifier REAL
    );
    ''')
    
    # Clear old assignments
    cursor.execute("DELETE FROM Daily_Umpires")
    
    # Enforce ALV mandate for officials
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={live_date}&hydrate=officials"
    
    try:
        response = requests.get(url, timeout=15).json()
    except Exception as e:
        print(f"API Error fetching umpires: {e}")
        return

    print(f"Mapping Home Plate Umpires for {live_date}...")
    
    for date_data in response.get('dates', []):
        for game in date_data.get('games', []):
            game_pk = game['gamePk']
            officials = game.get('officials', [])
            
            hp_umpire = "Unknown / TBD"
            for official in officials:
                if official.get('officialType') == 'Home Plate':
                    hp_umpire = official.get('official', {}).get('fullName', 'Unknown')
                    break
            
            run_modifier = get_umpire_modifier(hp_umpire)
            
            cursor.execute('''
            INSERT INTO Daily_Umpires (game_pk, home_plate_umpire, run_modifier)
            VALUES (?, ?, ?)
            ''', (game_pk, hp_umpire, run_modifier))
            
            print(f"Game {game_pk} | HP Umpire: {hp_umpire} | Variance Modifier: {run_modifier}")

    conn.commit()
    conn.close()
    print("-" * 50)
    print("Umpire Variance mapping complete.")

if __name__ == "__main__":
    execute_umpire_variance_pipeline()
