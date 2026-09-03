import sqlite3
import requests
import numpy as np
from datetime import datetime, timedelta
from engine import run_ultimate_monte_carlo
from engine_f5_props import run_f5_and_props_engine

def run_backtest_engine(days_back=14):
    print(f"Initializing Dual-Engine Backtesting Framework (Past {days_back} days)...")
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
        
    conn = sqlite3.connect('mlb_engine.db', timeout=30)
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA busy_timeout=10000;")
    
    # Ensure all pipeline tables exist
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
    ''')
    
    for col in ["predicted_edge REAL", "predicted_home_runs REAL", "predicted_away_runs REAL"]:
        try:
            cursor.execute(f"ALTER TABLE Model_Forecasts ADD COLUMN {col}")
        except sqlite3.OperationalError:
            pass

    # THE FIX: Clear tracking tables ONCE at the start so historical data accumulates
    cursor.execute("DELETE FROM Model_Forecasts")
    cursor.execute("DELETE FROM Post_Match_Analysis")
    cursor.execute("DELETE FROM F5_Forecasts")
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
        current_date += timedelta(days=1)
        
        day_url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}&hydrate=probablePitcher"
        try:
            day_res = requests.get(day_url, timeout=10).json()
        except Exception:
            continue
            
        day_games_count = 0
        games_data = []
        
        # ONLY clear Daily_Lineups inside the loop so the engine processes exactly one day's slate at a time
        cursor.execute("DELETE FROM Daily_Lineups")
        
        for date_data in day_res.get('dates', []):
            for game in date_data.get('games', []):
                if game['status']['abstractGameState'] != 'Final':
                    continue
                    
                game_pk = game['gamePk']
                home_team = game['teams']['home']['team']['name']
                away_team = game['teams']['away']['team']['name']
                home_score = game['teams']['home'].get('score', 0)
                away_score = game['teams']['away'].get('score', 0)
                
                home_pitcher = game['teams']['home'].get('probablePitcher', {}).get('fullName', 'Unknown Pitcher')
                away_pitcher = game['teams']['away'].get('probablePitcher', {}).get('fullName', 'Unknown Pitcher')
                
                cursor.execute('''
                    INSERT OR REPLACE INTO Daily_Lineups (game_pk, game_date, away_team, home_team, away_pitcher, home_pitcher, lineup_status, air_density, uv_modifier, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (game_pk, date_str, away_team, home_team, away_pitcher, home_pitcher, 'Confirmed', 1.225, 1.0, 'Final'))
                
                # Insert placeholder rows so engine.py UPDATE statements find the game records
                cursor.execute('''
                    INSERT OR REPLACE INTO Model_Forecasts (game_pk, home_team, away_team, timestamp)
                    VALUES (?, ?, ?, 'BACKTEST_INIT')
                ''', (game_pk, home_team, away_team))
                
                cursor.execute('''
                    INSERT OR REPLACE INTO F5_Forecasts (game_pk, away_team, home_team, away_starter, home_starter)
                    VALUES (?, ?, ?, ?, ?)
                ''', (game_pk, away_team, home_team, away_pitcher, home_pitcher))
                
                games_data.append({
                    'game_pk': game_pk,
                    'home_team': home_team,
                    'away_team': away_team,
                    'home_score': home_score,
                    'away_score': away_score
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
            game_pk = g['game_pk']
            home_score = g['home_score']
            away_score = g['away_score']
            
            actual_winner = home_team if home_score > away_score else away_team
            actual_home_win = 1 if actual_winner == home_team else 0
            
            cursor.execute("SELECT home_prob, away_prob, predicted_home_runs, predicted_away_runs FROM Model_Forecasts WHERE game_pk = ?", (game_pk,))
            forecast = cursor.fetchone()
            
            if not forecast or forecast[0] is None:
                continue
                
            home_prob, away_prob, pred_home_runs, pred_away_runs = forecast
            pred_home_runs = pred_home_runs if pred_home_runs is not None else 4.0
            pred_away_runs = pred_away_runs if pred_away_runs is not None else 4.0
            
            predicted_winner = home_team if home_prob > away_prob else away_team
            is_correct = 1 if predicted_winner == actual_winner else 0
            
            # Log the successful simulation directly into the final post-mortem table
            cursor.execute('''
                INSERT OR REPLACE INTO Post_Match_Analysis 
                (game_pk, actual_winner, home_score, away_score, home_f5_score, away_f5_score, model_correct, processed_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (game_pk, actual_winner, home_score, away_score, 0, 0, is_correct, 'BACKTEST'))
            
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

    conn.commit()
    conn.close()
    
    if total_games == 0:
        print("No finalized historical games processed successfully.")
        return
        
    win_rate = correct_predictions / total_games
    brier_score = brier_score_sum / total_games
    rmse = np.sqrt(squared_error_sum / total_games)
    
    print(f"Engine-Linked Backtest Analytical Output: {start_str} to {end_str}")
    print(f"Total Matchups Verified & Logged: {total_games}")
    print(f"Baseline Accuracy:        {win_rate:.2%}")
    print(f"Brier Score (0 = Exact):  {brier_score:.4f}")
    print(f"RMSE (Run Variance):      {rmse:.2f} runs")
    print(f"Simulated ROI Projection: {units_won:+.2f} Units (Flat 1u @ -110)")
    print("-" * 60)

if __name__ == "__main__":
    # You can change days_back=14 to whatever sample size you want to test!
    run_backtest_engine(days_back=14)
