import sqlite3
import requests
import numpy as np
from datetime import datetime, timedelta
from engine import run_ultimate_monte_carlo
from engine_f5_props import run_f5_and_props_engine
from post_match_analysis import update_dynamic_weights  # Corrected Import

def run_backtest_engine(days_back=220):
    print(f"Initializing Dual-Engine Backtesting Framework (Past {days_back} days)...")
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days_back)
    start_str, end_str = start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d')
    
    try:
        response = requests.get(f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={start_str}&endDate={end_str}&hydrate=probablePitcher,linescore", timeout=15).json()
    except Exception as e:
        return print(f"API Error fetching historical schedule: {e}")
        
    conn = sqlite3.connect('mlb_engine.db')
    cursor = conn.cursor()
    
    # Ensure ALL required tables exist before running the backtest sweep
    cursor.executescript('''
        CREATE TABLE IF NOT EXISTS Model_Forecasts (game_pk INTEGER PRIMARY KEY, home_team TEXT, away_team TEXT, home_prob REAL, away_prob REAL, predicted_edge REAL, predicted_home_runs REAL, predicted_away_runs REAL, timestamp TEXT);
        CREATE TABLE IF NOT EXISTS F5_Forecasts (game_pk INTEGER PRIMARY KEY, away_team TEXT, home_team TEXT, away_starter TEXT, home_starter TEXT, f5_away_prob REAL, f5_home_prob REAL, f5_tie_prob REAL, f5_exp_away_runs REAL, f5_exp_home_runs REAL, f5_total_runs REAL);
        CREATE TABLE IF NOT EXISTS Post_Match_Analysis (game_pk INTEGER PRIMARY KEY, actual_winner TEXT, home_score INTEGER, away_score INTEGER, home_f5_score INTEGER, away_f5_score INTEGER, model_correct INTEGER, processed_at TEXT);
        CREATE TABLE IF NOT EXISTS Daily_Lineups (game_pk INTEGER PRIMARY KEY, away_team TEXT, home_team TEXT, away_pitcher TEXT, home_pitcher TEXT);
        CREATE TABLE IF NOT EXISTS Pitcher_Modifiers (pitcher_name TEXT PRIMARY KEY, k_modifier REAL DEFAULT 1.0, f5_run_modifier REAL DEFAULT 1.0, last_updated TEXT);
        CREATE TABLE IF NOT EXISTS Dynamic_Modifiers (team_name TEXT PRIMARY KEY, offensive_modifier REAL DEFAULT 1.0, pitching_modifier REAL DEFAULT 1.0, last_updated TEXT);
    ''')

    total_games, correct_predictions = 0, 0
    fg_brier_sum, f5_brier_sum, squared_error_sum, units_won = 0.0, 0.0, 0.0, 0.0
    
    print("-" * 60)
    current_date = start_date
    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        current_date += timedelta(days=1)
        
        try: day_res = requests.get(f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}&hydrate=probablePitcher,linescore", timeout=10).json()
        except Exception: continue
            
        cursor.execute("DELETE FROM Daily_Lineups")
        cursor.execute("DELETE FROM Model_Forecasts")
        cursor.execute("DELETE FROM F5_Forecasts")
        
        games_data, day_games_count = [], 0
        
        for date_data in day_res.get('dates', []):
            for game in date_data.get('games', []):
                if game['status']['abstractGameState'] != 'Final': continue
                    
                game_pk, home_team, away_team = game['gamePk'], game['teams']['home']['team']['name'], game['teams']['away']['team']['name']
                home_score, away_score = game['teams']['home'].get('score', 0), game['teams']['away'].get('score', 0)
                home_pitcher, away_pitcher = game['teams']['home'].get('probablePitcher', {}).get('fullName', 'Unknown'), game['teams']['away'].get('probablePitcher', {}).get('fullName', 'Unknown')
                
                h_f5, a_f5 = 0, 0
                for inning in game.get('linescore', {}).get('innings', [])[:5]:
                    h_f5 += inning.get('home', {}).get('runs', 0)
                    a_f5 += inning.get('away', {}).get('runs', 0)
                
                cursor.execute('INSERT OR REPLACE INTO Daily_Lineups (game_pk, away_team, home_team, away_pitcher, home_pitcher) VALUES (?, ?, ?, ?, ?)', (game_pk, away_team, home_team, away_pitcher, home_pitcher))
                
                # Added pitchers to games_data dictionary for downstream F5 learning loops
                games_data.append({'game_pk': game_pk, 'home': home_team, 'away': away_team, 'h_score': home_score, 'a_score': away_score, 'h_f5': h_f5, 'a_f5': a_f5, 'h_pitcher': home_pitcher, 'a_pitcher': away_pitcher})
                day_games_count += 1
        
        if day_games_count == 0: continue
            
        conn.commit()
        run_ultimate_monte_carlo()
        run_f5_and_props_engine()
        cursor = conn.cursor()
        
        for g in games_data:
            actual_winner = "Home" if g['h_score'] > g['a_score'] else "Away"
            actual_home_win = 1 if actual_winner == "Home" else 0
            actual_f5_home_win = 1 if g['h_f5'] > g['a_f5'] else (0.5 if g['h_f5'] == g['a_f5'] else 0)
            
            cursor.execute("SELECT home_prob, away_prob, predicted_home_runs, predicted_away_runs FROM Model_Forecasts WHERE game_pk = ?", (g['game_pk'],))
            forecast = cursor.fetchone()
            
            # Fetch F5 Expected Runs for SOTA Pitcher Micro-Evolution
            cursor.execute("SELECT f5_home_prob, f5_exp_home_runs, f5_exp_away_runs FROM F5_Forecasts WHERE game_pk = ?", (g['game_pk'],))
            f5_forecast = cursor.fetchone()
            
            if not forecast: continue
                
            home_prob, away_prob, pred_home_runs, pred_away_runs = forecast
            predicted_winner = "Home" if home_prob > away_prob else "Away"
            is_correct = 1 if predicted_winner == actual_winner else 0
            
            cursor.execute('''INSERT OR REPLACE INTO Post_Match_Analysis (game_pk, actual_winner, home_score, away_score, home_f5_score, away_f5_score, model_correct, processed_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?)''', (g['game_pk'], actual_winner, g['h_score'], g['a_score'], g['h_f5'], g['a_f5'], is_correct, 'BACKTEST'))
            
            # 1. Macro-Evolution (Team Level)
            update_dynamic_weights(cursor, g['home'], pred_home_runs or 4.0, g['h_score'], is_offense=True)
            update_dynamic_weights(cursor, g['away'], pred_home_runs or 4.0, g['h_score'], is_offense=False)
            update_dynamic_weights(cursor, g['away'], pred_away_runs or 4.0, g['a_score'], is_offense=True)
            update_dynamic_weights(cursor, g['home'], pred_away_runs or 4.0, g['a_score'], is_offense=False)
            
            # 2. Micro-Evolution (Pitcher Level)
            if f5_forecast:
                f5_home_prob, f5_exp_home_runs, f5_exp_away_runs = f5_forecast
                
                # The Home Pitcher's job is preventing the Away Team's expected runs
                update_dynamic_weights(cursor, g['h_pitcher'], f5_exp_away_runs, g['a_f5'], is_pitcher=True)
                
                # The Away Pitcher's job is preventing the Home Team's expected runs
                update_dynamic_weights(cursor, g['a_pitcher'], f5_exp_home_runs, g['h_f5'], is_pitcher=True)
                
                f5_brier_sum += (f5_home_prob - actual_f5_home_win) ** 2
            
            total_games += 1
            correct_predictions += is_correct
            fg_brier_sum += (home_prob - actual_home_win) ** 2
            squared_error_sum += ((g['h_score'] + g['a_score']) - ((pred_home_runs or 4.0) + (pred_away_runs or 4.0))) ** 2
            units_won += 0.909 if is_correct else -1.000

    conn.commit()
    conn.close()
    
    if total_games == 0: return print("No finalized historical games processed.")
        
    print(f"Dual-Engine Backtest Output: {start_str} to {end_str}")
    print(f"Total Matchups Verified & Logged: {total_games}")
    print(f"Baseline Accuracy:        {correct_predictions / total_games:.2%}")
    print(f"Full-Game Brier Score:    {fg_brier_sum / total_games:.4f}")
    print(f"F5 Signal Brier Score:    {f5_brier_sum / total_games:.4f}")
    print(f"RMSE (Run Variance):      {np.sqrt(squared_error_sum / total_games):.2f} runs")
    print(f"Simulated ROI Projection: {units_won:+.2f} Units (Flat 1u @ -110)")
    print("-" * 60)

if __name__ == "__main__":
    run_backtest_engine(days_back=220)
