import sqlite3
import requests
import numpy as np
from datetime import datetime, timedelta
from engine import run_ultimate_monte_carlo
from factual_post_mortem import update_dynamic_weights # Newly imported from the renamed post-mortem script

def run_backtest_engine(days_back=220): # Expanded to naturally sweep the 1600-game dataset
    print(f"Initializing Engine-Linked Backtesting Framework (Past {days_back} days)...")
    
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
            away_pitcher TEXT, home_pitcher TEXT
        );
    ''')
    
    for col in ["predicted_edge REAL", "predicted_home_runs REAL", "predicted_away_runs REAL"]:
        try:
            cursor.execute(f"ALTER TABLE Model_Forecasts ADD COLUMN {col}")
        except sqlite3.OperationalError:
            pass
    try:
        cursor.execute("ALTER TABLE Post_Match_Analysis ADD COLUMN processed_at TEXT")
    except sqlite3.OperationalError:
        pass

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
            
        cursor.execute("DELETE FROM Daily_Lineups")
        cursor.execute("DELETE FROM Model_Forecasts")
        
        day_games_count = 0
        games_data = []
        
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
                    INSERT OR REPLACE INTO Daily_Lineups (game_pk, away_team, home_team, away_pitcher, home_pitcher)
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
            
            # ACTIVE LEARNING: Dynamically shift weights based on the freshly widened learning rate
            update_dynamic_weights(cursor, home_team, pred_home_runs, home_score, is_offense=True)
            update_dynamic_weights(cursor, away_team, pred_home_runs, home_score, is_offense=False)
            update_dynamic_weights(cursor, away_team, pred_away_runs, away_score, is_offense=True)
            update_dynamic_weights(cursor, home_team, pred_away_runs, away_score, is_offense=False)
            
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
    run_backtest_engine(days_back=220)
