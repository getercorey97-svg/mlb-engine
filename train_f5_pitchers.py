import sqlite3
import requests
from datetime import datetime, timedelta

def run_f5_historical_training():
    print("Initializing Phase 0: F5 Historical Momentum Trainer (2026 Season-to-Date)...")
    
    conn = sqlite3.connect('mlb_engine.db')
    cursor = conn.cursor()
    
    # Ensure Pitcher_Modifiers table exists
    cursor.executescript('''
        CREATE TABLE IF NOT EXISTS Pitcher_Modifiers (
            pitcher_name TEXT PRIMARY KEY, 
            k_modifier REAL DEFAULT 1.0, 
            f5_run_modifier REAL DEFAULT 1.0, 
            last_updated TEXT
        );
    ''')

    # Define 2026 Season start to today
    start_date = datetime(2026, 3, 20)
    end_date = datetime.now()
    
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={start_str}&endDate={end_str}&gameType=R&hydrate=probablePitcher,linescore"
    
    print(f"Fetching Historical F5 Linescores: {start_str} to {end_str}...")
    try:
        response = requests.get(url, timeout=20).json()
    except Exception as e:
        print(f"API Error: {e}")
        return

    processed_games = 0
    pitcher_updates = 0

    for date_data in response.get('dates', []):
        for game in date_data.get('games', []):
            if game['status']['abstractGameState'] != 'Final':
                continue
                
            teams = game.get('teams', {})
            home_team = teams['home']['team']['name']
            away_team = teams['away']['team']['name']
            
            home_p = teams['home'].get('probablePitcher', {}).get('fullName', 'Unknown')
            away_p = teams['away'].get('probablePitcher', {}).get('fullName', 'Unknown')
            
            if home_p == 'Unknown' and away_p == 'Unknown':
                continue

            linescore = game.get('linescore', {}).get('innings', [])
            h_f5, a_f5 = 0, 0
            for inning in linescore[:5]:
                h_f5 += inning.get('home', {}).get('runs', 0)
                a_f5 += inning.get('away', {}).get('runs', 0)

            # Empirical baseline expected runs for 5 innings (approximate league average 2.25)
            baseline_exp_runs = 2.25 
            
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            base_lr = 0.015

            # Apply exact Micro-Evolution constraints to each pitcher
            for pitcher, actual_runs_allowed in [(home_p, a_f5), (away_p, h_f5)]:
                if pitcher == 'Unknown':
                    continue
                    
                error_delta = actual_runs_allowed - baseline_exp_runs
                adaptive_lr = min(0.08, base_lr + (abs(error_delta) * 0.005))
                
                cursor.execute('SELECT f5_run_modifier FROM Pitcher_Modifiers WHERE pitcher_name = ?', (pitcher,))
                result = cursor.fetchone()
                mod = result[0] if result else 1.0
                
                # 0.47 Volatility Ceiling
                new_mod = max(0.53, min(1.47, mod + (error_delta * adaptive_lr)))
                
                cursor.execute('''
                    INSERT OR REPLACE INTO Pitcher_Modifiers (pitcher_name, f5_run_modifier, last_updated) 
                    VALUES (?, ?, ?)
                ''', (pitcher, new_mod, current_time))
                pitcher_updates += 1

            processed_games += 1

    conn.commit()
    conn.close()
    print("-" * 65)
    print(f"F5 Historical Training Complete.")
    print(f"Processed {processed_games} games. Locked in {pitcher_updates} pitcher momentum updates.")
    print("-" * 65)

if __name__ == "__main__":
    run_f5_historical_training()
