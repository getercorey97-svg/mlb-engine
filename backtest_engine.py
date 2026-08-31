import sqlite3
import requests
import numpy as np
from datetime import datetime, timedelta
from engine import run_ultimate_monte_carlo

# Sample Baseline Umpire Bias Mapping
HISTORICAL_UMPIRE_BIAS = {
    "CB Bucknor": 1.045,
    "Angel Hernandez": 1.052,
    "Pat Hoberg": 0.985,
    "Doug Eddings": 0.970,
    "Lance Barksdale": 0.980,
    "Dan Bellino": 1.025,
    "Default": 1.000
}

# Centralized coordinates for Air Density & UV
STADIUMS = {
    "Atlanta Braves": (33.8907, -84.4677),
    "Colorado Rockies": (39.7559, -104.9942),
    "San Diego Padres": (32.7076, -117.1570),
    "Chicago Cubs": (41.9484, -87.6553),
    "New York Yankees": (40.8296, -73.9262),
    "Default": (39.8283, -98.5795)
}

def get_historical_atmosphere(team_name, date_str):
    """Calculates historical stadium air density using Open-Meteo Archive API[span_1](start_span)[span_1](end_span)."""
    coords = STADIUMS.get(team_name, STADIUMS["Default"])
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": coords[0],
        "longitude": coords[1],
        "start_date": date_str,
        "end_date": date_str,
        "hourly": "surface_pressure,temperature_2m,cloud_cover"
    }
    try:
        res = requests.get(url, params=params, timeout=10).json()
        # Extract mid-day atmospheric conditions[span_2](start_span)[span_2](end_span)
        temp_c = res['hourly']['temperature_2m'][12] 
        pressure_hpa = res['hourly']['surface_pressure'][12]
        clouds = res['hourly']['cloud_cover'][12]
        
        temp_k = temp_c + 273.15
        pressure_pa = pressure_hpa * 100
        density = round(pressure_pa / (287.05 * temp_k), 4)
        
        uv_modifier = 1.03 if clouds > 70 else 1.00
        return density, uv_modifier
    except Exception:
        return 1.225, 1.00

def run_backtest_engine(days_back=14):
    print(f"Initializing Historical Seeding Framework with Air Density & Umpires (Past {days_back} days)...")
    
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=days_back - 1)
    
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
        
    conn = sqlite3.connect('mlb_engine.db')
    cursor = conn.cursor()
    
    # Securely build the required schemas before simulating to prevent crashes[span_3](start_span)[span_3](end_span)
    cursor.executescript('''
        CREATE TABLE IF NOT EXISTS Model_Forecasts (
            game_pk INTEGER PRIMARY KEY, home_team TEXT, away_team TEXT, 
            home_prob REAL, away_prob REAL, predicted_edge REAL, 
            predicted_home_runs REAL, predicted_away_runs REAL, timestamp TEXT
        );
        CREATE TABLE IF NOT EXISTS Post_Match_Analysis (
            game_pk INTEGER PRIMARY KEY, actual_winner TEXT, home_score INTEGER, 
            away_score INTEGER, model_correct INTEGER, processed_at TEXT
        );
        CREATE TABLE IF NOT EXISTS Daily_Lineups (
            game_pk INTEGER PRIMARY KEY, away_team TEXT, home_team TEXT, 
            away_pitcher TEXT, home_pitcher TEXT, lineup_status TEXT, 
            air_density REAL, uv_modifier REAL, status TEXT
        );
        CREATE TABLE IF NOT EXISTS Daily_Umpires (
            game_pk INTEGER PRIMARY KEY, home_plate_umpire TEXT, run_modifier REAL
        );
        CREATE TABLE IF NOT EXISTS Biological_Modifiers (
            team_name TEXT PRIMARY KEY, jet_lag_runs_penalty REAL
        );
        CREATE TABLE IF NOT EXISTS Dynamic_Modifiers (
            team_name TEXT PRIMARY KEY,
            offensive_modifier REAL DEFAULT 1.0,
            pitching_modifier REAL DEFAULT 1.0,
            last_updated TEXT
        );
    ''')
    
    cursor.execute("DELETE FROM Post_Match_Analysis")
    conn.commit()

    total_games = 0
    correct_predictions = 0
    brier_score_sum = 0.0
    squared_error_sum = 0.0
    units_won = 0.0
    
    print("-" * 60)
    
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        
        # Pull historical lineup AND official umpire assignments[span_4](start_span)[span_4](end_span)
        day_url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}&hydrate=probablePitcher,officials"
        try:
            day_res = requests.get(day_url, timeout=10).json()
        except Exception:
            current_date += timedelta(days=1)
            continue
            
        cursor.execute("DELETE FROM Daily_Lineups")
        cursor.execute("DELETE FROM Model_Forecasts")
        cursor.execute("DELETE FROM Daily_Umpires")
        conn.commit()
        
        day_games_count = 0
        games_data = []
        
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
                
                officials = game.get('officials', [])
                hp_umpire = "Unknown / TBD"
                for official in officials:
                    if official.get('officialType') == 'Home Plate':
                        hp_umpire = official.get('official', {}).get('fullName', 'Unknown')
                        break
                run_modifier = HISTORICAL_UMPIRE_BIAS.get(hp_umpire, HISTORICAL_UMPIRE_BIAS["Default"])
                
                air_density, uv_mod = get_historical_atmosphere(home_team, date_str)
                
                cursor.execute('''
                    INSERT OR REPLACE INTO Daily_Lineups (game_pk, away_team, home_team, away_pitcher, home_pitcher, lineup_status, air_density, uv_modifier, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (game_pk, away_team, home_team, away_pitcher, home_pitcher, "Confirmed", air_density, uv_mod, "Final"))
                
                cursor.execute('''
                    INSERT OR REPLACE INTO Daily_Umpires (game_pk, home_plate_umpire, run_modifier)
                    VALUES (?, ?, ?)
                ''', (game_pk, hp_umpire, run_modifier))
                
                cursor.execute('''
                    INSERT OR REPLACE INTO Model_Forecasts (game_pk, home_team, away_team, timestamp)
                    VALUES (?, ?, ?, 'BACKTEST_INIT')
                ''', (game_pk, home_team, away_team))
                
                games_data.append({
                    'game_pk': game_pk,
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_score': home_score,
                    'away_score': away_score
                })
                day_games_count += 1
        
        if day_games_count == 0:
            current_date += timedelta(days=1)
            continue
            
        conn.commit()
        
        # Triggers engine.py directly[span_5](start_span)[span_5](end_span)
        run_ultimate_monte_carlo()
        
        cursor = conn.cursor()
        
        for g in games_data:
            game_pk = g['game_pk']
            home_score = g['home_score']
            away_score = g['away_score']
            actual_winner = "Home" if home_score > away_score else "Away"
            actual_home_win = 1 if actual_winner == "Home" else 0
            
            cursor.execute("SELECT home_prob, away_prob, predicted_home_runs, predicted_away_runs FROM Model_Forecasts WHERE game_pk = ?", (game_pk,))
            forecast = cursor.fetchone()
            
            if not forecast:
                continue
                
            home_prob, away_prob, pred_home_runs, pred_away_runs = forecast
            pred_home_runs = pred_home_runs if pred_home_runs is not None else 4.0
            pred_away_runs = pred_away_runs if pred_away_runs is not None else 4.0
            
            predicted_winner = "Home" if home_prob > away_prob else "Away"
            is_correct = 1 if predicted_winner == actual_winner else 0
            
            cursor.execute('''
                INSERT OR REPLACE INTO Post_Match_Analysis 
                (game_pk, actual_winner, home_score, away_score, model_correct, processed_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (game_pk, actual_winner, home_score, away_score, is_correct, 'BACKTEST'))
            
            total_games += 1
            correct_predictions += is_correct
            brier_score_sum += (home_prob - actual_home_win) ** 2
            
            total_actual_runs = home_score + away_score
            total_pred_runs = pred_home_runs + pred_away_runs
            squared_error_sum += (total_actual_runs - total_pred_runs) ** 2
            
            if is_correct:
                units_won += 0.909
            else:
                units_won -= 1.000

        current_date += timedelta(days=1)

    conn.commit()
    conn.close()
    
    if total_games == 0:
        print("No finalized historical games processed successfully.")
        return
        
    win_rate = correct_predictions / total_games
    brier_score = brier_score_sum / total_games
    rmse = np.sqrt(squared_error_sum / total_games)
    
    print(f"Historical Seeding & Backtest Output: {start_str} to {end_str}")
    print(f"Total Matchups Verified & Logged: {total_games}")
    print(f"Baseline Accuracy:        {win_rate:.2%}")
    print(f"Brier Score (0 = Exact):  {brier_score:.4f}")
    print(f"RMSE (Run Variance):      {rmse:.2f} runs")
    print(f"Simulated ROI Projection: {units_won:+.2f} Units (Flat 1u @ -110)")
    print("-" * 60)

if __name__ == "__main__":
    run_backtest_engine(days_back=14)
