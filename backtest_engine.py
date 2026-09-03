import sqlite3
import requests
import numpy as np
from datetime import datetime, timedelta
from engine import run_ultimate_monte_carlo
from engine_f5_props import run_f5_and_props_engine

# --- SOTA CONSTANTS ---
HISTORICAL_UMPIRE_BIAS = {
    "CB Bucknor": 1.045, "Angel Hernandez": 1.052, "Pat Hoberg": 0.985,
    "Doug Eddings": 0.970, "Lance Barksdale": 0.980, "Dan Bellino": 1.025,
    "Default": 1.000
}

STADIUMS = {
    "Arizona Diamondbacks": (33.4453, -112.0667), "Atlanta Braves": (33.8907, -84.4677),
    "Baltimore Orioles": (39.2839, -76.6216), "Boston Red Sox": (42.3467, -71.0972),
    "Chicago Cubs": (41.9484, -87.6553), "Chicago White Sox": (41.8299, -87.6338),
    "Cincinnati Reds": (39.0974, -84.5071), "Cleveland Guardians": (41.4962, -81.6852),
    "Colorado Rockies": (39.7559, -104.9942), "Detroit Tigers": (42.3390, -83.0485),
    "Houston Astros": (29.7569, -95.3555), "Kansas City Royals": (39.0517, -94.4803),
    "Los Angeles Angels": (33.8003, -117.8827), "Los Angeles Dodgers": (34.0739, -118.2400),
    "Miami Marlins": (25.7781, -80.2197), "Milwaukee Brewers": (43.0280, -87.9712),
    "Minnesota Twins": (44.9817, -93.2778), "New York Mets": (40.7571, -73.8458),
    "New York Yankees": (40.8296, -73.9262), "Oakland Athletics": (37.7516, -122.2005),
    "Philadelphia Phillies": (39.9061, -75.1665), "Pittsburgh Pirates": (40.4469, -80.0057),
    "San Diego Padres": (32.7076, -117.1570), "San Francisco Giants": (37.7786, -122.3893),
    "Seattle Mariners": (47.5914, -122.3325), "St. Louis Cardinals": (38.6226, -90.1928),
    "Tampa Bay Rays": (27.7682, -82.6534), "Texas Rangers": (32.7473, -97.0845),
    "Toronto Blue Jays": (43.6414, -79.3894), "Washington Nationals": (38.8730, -77.0074),
    "Default": (39.8283, -98.5795)
}

def get_historical_atmosphere(team_name, date_str):
    """Pulls true historical Thermodynamic Air Density and UV Contrast via Open-Meteo."""
    coords = STADIUMS.get(team_name, STADIUMS["Default"])
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": coords[0], "longitude": coords[1],
        "start_date": date_str, "end_date": date_str,
        "hourly": "surface_pressure,temperature_2m,cloud_cover"
    }
    try:
        res = requests.get(url, params=params, timeout=10).json()
        temp_c = res['hourly']['temperature_2m'][12] 
        pressure_hpa = res['hourly']['surface_pressure'][12]
        cloud_cover = res['hourly']['cloud_cover'][12]
        
        temp_k = temp_c + 273.15
        pressure_pa = pressure_hpa * 100
        density = round(pressure_pa / (287.05 * temp_k), 4)
        uv_modifier = 1.03 if cloud_cover > 70 else 1.00
        return density, uv_modifier
    except Exception:
        return 1.225, 1.00

def update_dynamic_weights(cursor, name, predicted_runs, actual_runs, is_offense=True, is_pitcher=False):
    """Executes the SOTA Micro-Evolution loop to update Pitcher/Team momentum mid-backtest."""
    error_delta = actual_runs - predicted_runs
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    base_lr = 0.03
    adaptive_lr = min(0.15, base_lr + (abs(error_delta) * 0.015))
    
    if is_pitcher:
        cursor.execute('SELECT f5_run_modifier FROM Pitcher_Modifiers WHERE pitcher_name = ?', (name,))
        result = cursor.fetchone()
        mod = result[0] if result else 1.0
        new_mod = max(0.53, min(1.47, mod + (error_delta * adaptive_lr)))
        cursor.execute('INSERT OR REPLACE INTO Pitcher_Modifiers (pitcher_name, k_modifier, f5_run_modifier, last_updated) VALUES (?, 1.0, ?, ?)', (name, new_mod, current_time))
        return

    cursor.execute('SELECT offensive_modifier, pitching_modifier FROM Dynamic_Modifiers WHERE team_name = ?', (name,))
    result = cursor.fetchone()
    if not result:
        cursor.execute('INSERT OR IGNORE INTO Dynamic_Modifiers (team_name, offensive_modifier, pitching_modifier, last_updated) VALUES (?, 1.0, 1.0, ?)', (name, current_time))
        off_mod, pitch_mod = 1.0, 1.0
    else:
        off_mod, pitch_mod = result
    
    if is_offense:
        new_off_mod = max(0.53, min(1.47, off_mod + (error_delta * adaptive_lr)))
        cursor.execute('UPDATE Dynamic_Modifiers SET offensive_modifier = ?, last_updated = ? WHERE team_name = ?', (new_off_mod, current_time, name))
    else:
        new_pitch_mod = max(0.53, min(1.47, pitch_mod + (error_delta * adaptive_lr)))
        cursor.execute('UPDATE Dynamic_Modifiers SET pitching_modifier = ?, last_updated = ? WHERE team_name = ?', (new_pitch_mod, current_time, name))

def run_backtest_engine():
    current_year = datetime.now().year
    print(f"Initializing Dual-Engine Backtesting Framework (Full {current_year} Season)...")
    print("Strict Adherence to SOTA Mechanics: ALV Atmosphere, Umpires & Adaptive Evolution Active.")
    
    end_date = datetime.now()
    start_date = datetime(current_year, 3, 20)
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    conn = sqlite3.connect('mlb_engine.db', timeout=30)
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA busy_timeout=10000;")
    
    cursor.executescript('''
        CREATE TABLE IF NOT EXISTS Model_Forecasts (
            game_pk INTEGER PRIMARY KEY, home_team TEXT, away_team TEXT, 
            home_prob REAL, away_prob REAL, predicted_edge REAL, 
            predicted_home_runs REAL, predicted_away_runs REAL, timestamp TEXT
        );
        CREATE TABLE IF NOT EXISTS F5_Forecasts (
            game_pk INTEGER PRIMARY KEY, away_team TEXT, home_team TEXT, 
            away_starter TEXT, home_starter TEXT, f5_away_prob REAL, 
            f5_home_prob REAL, f5_tie_prob REAL, f5_exp_away_runs REAL, 
            f5_exp_home_runs REAL, f5_total_runs REAL
        );
        CREATE TABLE IF NOT EXISTS Post_Match_Analysis (
            game_pk INTEGER PRIMARY KEY, actual_winner TEXT, home_score INTEGER, 
            away_score INTEGER, home_f5_score INTEGER, away_f5_score INTEGER, model_correct INTEGER, processed_at TEXT
        );
        CREATE TABLE IF NOT EXISTS Daily_Lineups (
            game_pk INTEGER PRIMARY KEY, game_date TEXT, away_team TEXT, home_team TEXT, 
            away_pitcher TEXT, home_pitcher TEXT, lineup_status TEXT, air_density REAL, uv_modifier REAL, status TEXT
        );
        CREATE TABLE IF NOT EXISTS Daily_Umpires (
            game_pk INTEGER PRIMARY KEY, home_plate_umpire TEXT, run_modifier REAL
        );
        CREATE TABLE IF NOT EXISTS Pitcher_Modifiers (
            pitcher_name TEXT PRIMARY KEY, k_modifier REAL DEFAULT 1.0, f5_run_modifier REAL DEFAULT 1.0, last_updated TEXT
        );
        CREATE TABLE IF NOT EXISTS Dynamic_Modifiers (
            team_name TEXT PRIMARY KEY, offensive_modifier REAL DEFAULT 1.0, pitching_modifier REAL DEFAULT 1.0, last_updated TEXT
        );
    ''')
    
    for col in ["predicted_edge REAL", "predicted_home_runs REAL", "predicted_away_runs REAL"]:
        try: cursor.execute(f"ALTER TABLE Model_Forecasts ADD COLUMN {col}")
        except sqlite3.OperationalError: pass

    # Clear tracking tables ONCE at the start so historical data accumulates accurately
    cursor.execute("DELETE FROM Model_Forecasts")
    cursor.execute("DELETE FROM Post_Match_Analysis")
    cursor.execute("DELETE FROM F5_Forecasts")
    
    # Initialize Teams and Pitchers at baseline 1.0 before the historical sweep starts
    cursor.execute("DELETE FROM Pitcher_Modifiers")
    cursor.execute("DELETE FROM Dynamic_Modifiers")
    conn.commit()

    total_games = 0
    correct_predictions = 0
    brier_score_sum = 0.0
    squared_error_sum = 0.0
    units_won = 0.0
    
    f5_wins = 0
    f5_losses = 0
    f5_pushes = 0
    f5_units = 0.0
    f5_total_games = 0
    
    print("-" * 60)
    
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        current_date += timedelta(days=1)
        
        # Hydrate officials (umpires), probablePitcher, and linescore
        day_url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}&hydrate=probablePitcher,linescore,officials"
        try:
            day_res = requests.get(day_url, timeout=10).json()
        except Exception:
            continue
            
        day_games_count = 0
        games_data = []
        
        cursor.execute("DELETE FROM Daily_Lineups")
        cursor.execute("DELETE FROM Daily_Umpires")
        
        for date_data in day_res.get('dates', []):
            for game in date_data.get('games', []):
                if game.get('status', {}).get('abstractGameState') != 'Final':
                    continue
                    
                game_pk = game['gamePk']
                home_team = game['teams']['home']['team']['name']
                away_team = game['teams']['away']['team']['name']
                home_score = game['teams']['home'].get('score', 0)
                away_score = game['teams']['away'].get('score', 0)
                
                home_pitcher = game['teams']['home'].get('probablePitcher', {}).get('fullName', 'Unknown Pitcher')
                away_pitcher = game['teams']['away'].get('probablePitcher', {}).get('fullName', 'Unknown Pitcher')
                
                # Fetch Umpires
                officials = game.get('officials', [])
                hp_umpire = "Unknown / TBD"
                for official in officials:
                    if official.get('officialType') == 'Home Plate':
                        hp_umpire = official.get('official', {}).get('fullName', 'Unknown')
                        break
                ump_modifier = HISTORICAL_UMPIRE_BIAS.get(hp_umpire, HISTORICAL_UMPIRE_BIAS["Default"])
                
                # Fetch historical Weather/Air Density
                air_density, uv_modifier = get_historical_atmosphere(home_team, date_str)
                
                # F5 Linescore Extraction
                linescore = game.get('linescore', {}).get('innings', [])
                h_f5, a_f5 = 0, 0
                for inning in linescore[:5]:
                    h_f5 += inning.get('home', {}).get('runs', 0)
                    a_f5 += inning.get('away', {}).get('runs', 0)
                
                cursor.execute('''
                    INSERT OR REPLACE INTO Daily_Lineups (game_pk, game_date, away_team, home_team, away_pitcher, home_pitcher, lineup_status, air_density, uv_modifier, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (game_pk, date_str, away_team, home_team, away_pitcher, home_pitcher, 'Confirmed', air_density, uv_modifier, 'Final'))
                
                cursor.execute('''
                    INSERT OR REPLACE INTO Daily_Umpires (game_pk, home_plate_umpire, run_modifier)
                    VALUES (?, ?, ?)
                ''', (game_pk, hp_umpire, ump_modifier))
                
                cursor.execute('''
                    INSERT OR REPLACE INTO Model_Forecasts (game_pk, home_team, away_team, timestamp)
                    VALUES (?, ?, ?, 'BACKTEST_INIT')
                ''', (game_pk, home_team, away_team))
                
                cursor.execute('''
                    INSERT OR REPLACE INTO F5_Forecasts (game_pk, away_team, home_team, away_starter, home_starter)
                    VALUES (?, ?, ?, ?, ?)
                ''', (game_pk, away_team, home_team, away_pitcher, home_pitcher))
                
                games_data.append({
                    'game_pk': game_pk, 'home_team': home_team, 'away_team': away_team,
                    'home_score': home_score, 'away_score': away_score, 'h_f5': h_f5, 'a_f5': a_f5,
                    'home_pitcher': home_pitcher, 'away_pitcher': away_pitcher
                })
                day_games_count += 1
        
        if day_games_count == 0:
            continue
            
        conn.commit()
        
        # Execute the Dual-Engine simulation for this specific batch date
        try:
            run_ultimate_monte_carlo()
            run_f5_and_props_engine()
        except Exception as e:
            print(f"Simulation error on {date_str}: {e}")
            pass
        
        cursor = conn.cursor()
        
        for g in games_data:
            game_pk, home_team, away_team = g['game_pk'], g['home_team'], g['away_team']
            home_score, away_score = g['home_score'], g['away_score']
            h_f5, a_f5 = g['h_f5'], g['a_f5']
            home_pitcher, away_pitcher = g['home_pitcher'], g['away_pitcher']
            
            # --- FULL GAME EVALUATION ---
            actual_winner = home_team if home_score > away_score else away_team
            actual_home_win = 1 if actual_winner == home_team else 0
            
            cursor.execute("SELECT home_prob, away_prob, predicted_home_runs, predicted_away_runs FROM Model_Forecasts WHERE game_pk = ?", (game_pk,))
            forecast = cursor.fetchone()
            
            is_correct = 0
            pred_home_runs, pred_away_runs = 4.0, 4.0
            
            if forecast and forecast[0] is not None:
                home_prob, away_prob, pred_home_runs, pred_away_runs = forecast
                pred_home_runs = pred_home_runs if pred_home_runs is not None else 4.0
                pred_away_runs = pred_away_runs if pred_away_runs is not None else 4.0
                
                predicted_winner = home_team if home_prob > away_prob else away_team
                is_correct = 1 if predicted_winner == actual_winner else 0
                
                total_games += 1
                correct_predictions += is_correct
                brier_score_sum += (home_prob - actual_home_win) ** 2
                
                total_actual_runs = home_score + away_score
                total_pred_runs = pred_home_runs + pred_away_runs
                squared_error_sum += (total_actual_runs - total_pred_runs) ** 2
                
                if is_correct: units_won += 0.909
                else: units_won -= 1.000

            # --- F5 EVALUATION & PITCHER MICRO-EVOLUTION ---
            cursor.execute("SELECT f5_home_prob, f5_away_prob, f5_exp_home_runs, f5_exp_away_runs FROM F5_Forecasts WHERE game_pk = ?", (game_pk,))
            f5_fc = cursor.fetchone()
            
            if f5_fc and f5_fc[0] is not None:
                f5_h_prob, f5_a_prob, f5_h_runs, f5_a_runs = f5_fc
                
                f5_pick = home_team if f5_h_prob > f5_a_prob else away_team
                if h_f5 > a_f5: f5_actual = home_team
                elif a_f5 > h_f5: f5_actual = away_team
                else: f5_actual = "Tie"
                
                f5_total_games += 1
                if f5_actual == "Tie": f5_pushes += 1
                elif f5_pick == f5_actual: f5_wins += 1; f5_units += 0.909
                else: f5_losses += 1; f5_units -= 1.000
                
                # ACTIVE LEARNING: Evolve Pitcher F5 Modifiers!
                update_dynamic_weights(cursor, home_pitcher, f5_a_runs, a_f5, is_pitcher=True)
                update_dynamic_weights(cursor, away_pitcher, f5_h_runs, h_f5, is_pitcher=True)
                
            # ACTIVE LEARNING: Evolve Team Modifiers!
            update_dynamic_weights(cursor, home_team, pred_home_runs, home_score, is_offense=True)
            update_dynamic_weights(cursor, away_team, pred_home_runs, home_score, is_offense=False)
            update_dynamic_weights(cursor, away_team, pred_away_runs, away_score, is_offense=True)
            update_dynamic_weights(cursor, home_team, pred_away_runs, away_score, is_offense=False)
            
            cursor.execute('''
                INSERT OR REPLACE INTO Post_Match_Analysis 
                (game_pk, actual_winner, home_score, away_score, home_f5_score, away_f5_score, model_correct, processed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (game_pk, actual_winner, home_score, away_score, h_f5, a_f5, is_correct, 'BACKTEST'))

    conn.commit()
    conn.close()
    
    if total_games == 0:
        print("No finalized historical games processed successfully.")
        return
        
    win_rate = correct_predictions / total_games
    brier_score = brier_score_sum / total_games
    rmse = np.sqrt(squared_error_sum / total_games)
    f5_win_rate = f5_wins / (f5_wins + f5_losses) if (f5_wins + f5_losses) > 0 else 0
    
    print(f"\nSOTA Engine-Linked Backtest Output: {start_str} to {end_str}")
    print(f"Total Matchups Verified & Logged: {total_games}")
    print("-" * 60)
    print(f"--- FULL-GAME ENGINE (9 Innings) ---")
    print(f"Baseline Accuracy:        {win_rate:.2%}")
    print(f"Brier Score (0 = Exact):  {brier_score:.4f}")
    print(f"RMSE (Run Variance):      {rmse:.2f} runs")
    print(f"Simulated ROI Projection: {units_won:+.2f} Units (Flat 1u @ -110)")
    print("-" * 60)
    print(f"--- FIRST 5 INNINGS ENGINE (F5) ---")
    print(f"F5 Record:                {f5_wins}W - {f5_losses}L - {f5_pushes}P")
    print(f"F5 Win Rate (w/o pushes): {f5_win_rate:.2%}")
    print(f"F5 ROI Projection:        {f5_units:+.2f} Units (Flat 1u @ -110)")
    print("-" * 60)

if __name__ == "__main__":
    run_backtest_engine()
