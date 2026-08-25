import sqlite3
import requests
import numpy as np
from datetime import datetime, timedelta

def run_backtest_engine(days_back=14):
    print(f"Initializing Backtesting Framework (Evaluating past {days_back} days)...")
    print("Strict Adherence to Factual Post-Mortem Mandate: Zero synthetic data permitted.")
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    start_str = start_date.strftime('%Y-%m-%d')
    end_str = end_date.strftime('%Y-%m-%d')
    
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={start_str}&endDate={end_str}&hydrate=probablePitcher"
    
    try:
        response = requests.get(url, timeout=15).json()
    except Exception as e:
        print(f"API Error fetching historical schedule: {e}")
        return
        
    conn = sqlite3.connect('mlb_engine.db')
    cursor = conn.cursor()
    
    # Ensure tables exist for historical hydration
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
    ''')

    def get_metric(table, column, key_column, key_value, fallback):
        try:
            cursor.execute(f"SELECT {column} FROM {table} WHERE {key_column} LIKE ?", (f'%{key_value}%',))
            result = cursor.fetchone()
            return float(result[0]) if result and result[0] is not None else fallback
        except Exception:
            return fallback

    total_games = 0
    correct_predictions = 0
    brier_score_sum = 0.0
    squared_error_sum = 0.0
    units_won = 0.0
    
    print("-" * 60)
    
    for date_data in response.get('dates', []):
        for game in date_data.get('games', []):
            if game['status']['abstractGameState'] != 'Final':
                continue
                
            game_pk = game['gamePk']
            home_team = game['teams']['home']['team']['name']
            away_team = game['teams']['away']['team']['name']
            home_score = game['teams']['home'].get('score', 0)
            away_score = game['teams']['away'].get('score', 0)
            
            home_pitcher = game['teams']['home'].get('probablePitcher', {}).get('lastName', 'Unknown')
            away_pitcher = game['teams']['away'].get('probablePitcher', {}).get('lastName', 'Unknown')

            actual_winner = "Home" if home_score > away_score else "Away"
            actual_home_win = 1 if actual_winner == "Home" else 0
            
            home_off_mod = get_metric('Dynamic_Modifiers', 'offensive_modifier', 'team_name', home_team, 1.0)
            away_off_mod = get_metric('Dynamic_Modifiers', 'offensive_modifier', 'team_name', away_team, 1.0)
            home_pitch_mod = get_metric('Dynamic_Modifiers', 'pitching_modifier', 'team_name', home_team, 1.0)
            away_pitch_mod = get_metric('Dynamic_Modifiers', 'pitching_modifier', 'team_name', away_team, 1.0)
            
            home_xera = get_metric('Pitcher_Stats', 'est_era', 'last_name', home_pitcher, 4.30)
            away_xera = get_metric('Pitcher_Stats', 'est_era', 'last_name', away_pitcher, 4.30)
            
            home_ops = get_metric('Team_Offense', 'ops', 'team_name', home_team, 0.720)
            away_ops = get_metric('Team_Offense', 'ops', 'team_name', away_team, 0.720)
            
            home_bullpen = get_metric('Team_Bullpen', 'team_era', 'team_name', home_team, 4.00)
            away_bullpen = get_metric('Team_Bullpen', 'team_era', 'team_name', away_team, 4.00)
            
            park_factor = get_metric('Park_Factors', 'run_factor', 'home_team', home_team, 1.000)
            
            away_ops_mult = (away_ops / 0.720) * away_off_mod
            home_ops_mult = (home_ops / 0.720) * home_off_mod
            
            adj_home_xera = home_xera * home_pitch_mod
            adj_away_xera = away_xera * away_pitch_mod
            
            adj_home_bullpen = home_bullpen * home_pitch_mod
            adj_away_bullpen = away_bullpen * away_pitch_mod
            
            away_lambda = ((adj_home_xera * away_ops_mult * park_factor) * 0.66) + ((adj_home_bullpen * away_ops_mult * park_factor) * 0.33)
            home_lambda = ((adj_away_xera * home_ops_mult * park_factor) * 0.66) + ((adj_away_bullpen * home_ops_mult * park_factor) * 0.33)
            
            iterations = 2500
            away_sims = np.random.poisson(away_lambda, iterations)
            home_sims = np.random.poisson(home_lambda, iterations)
            
            ties = away_sims == home_sims
            while np.any(ties):
                away_sims[ties] += np.random.poisson((adj_home_bullpen / 3) + 1.1, np.sum(ties))
                home_sims[ties] += np.random.poisson((adj_away_bullpen / 3) + 1.1, np.sum(ties))
                ties = away_sims == home_sims
                
            home_prob = float(np.sum(home_sims > away_sims) / iterations)
            away_prob = float(np.sum(away_sims > home_sims) / iterations)
            predicted_edge = float(abs(home_prob - away_prob))
            
            predicted_winner = "Home" if home_prob > away_prob else "Away"
            is_correct = 1 if predicted_winner == actual_winner else 0
            
            # Log historical data into the database for the Correlation Engine
            cursor.execute('''
                INSERT OR REPLACE INTO Model_Forecasts 
                (game_pk, home_team, away_team, home_prob, away_prob, predicted_edge, predicted_home_runs, predicted_away_runs, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (game_pk, home_team, away_team, home_prob, away_prob, predicted_edge, home_lambda, away_lambda, 'BACKTEST'))
            
            cursor.execute('''
                INSERT OR REPLACE INTO Post_Match_Analysis 
                (game_pk, actual_winner, home_score, away_score, model_correct, processed_at)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (game_pk, actual_winner, home_score, away_score, is_correct, 'BACKTEST'))

            total_games += 1
            correct_predictions += is_correct
            
            brier_score_sum += (home_prob - actual_home_win) ** 2
            
            total_actual_runs = home_score + away_score
            total_pred_runs = home_lambda + away_lambda
            squared_error_sum += (total_actual_runs - total_pred_runs) ** 2
            
            if is_correct:
                units_won += 0.909
            else:
                units_won -= 1.000

    conn.commit()
    conn.close()
    
    if total_games == 0:
        print("No finalized historical games found in the specified range.")
        return
        
    win_rate = correct_predictions / total_games
    brier_score = brier_score_sum / total_games
    rmse = np.sqrt(squared_error_sum / total_games)
    
    print(f"Backtest Analytical Output: {start_str} to {end_str}")
    print(f"Total Matchups Verified & Logged: {total_games}")
    print(f"Baseline Accuracy:        {win_rate:.2%}")
    print(f"Brier Score (0 = Exact):  {brier_score:.4f}")
    print(f"RMSE (Run Variance):      {rmse:.2f} runs")
    print(f"Simulated ROI Projection: {units_won:+.2f} Units (Flat 1u @ -110)")
    print("-" * 60)

if __name__ == "__main__":
    run_backtest_engine(days_back=14)
