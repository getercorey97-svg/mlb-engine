import sqlite3
import requests
from datetime import datetime, timedelta

def execute_factual_post_mortem():
    print("Executing Phase 1: Factual Post-Mortem and Model Evolution...")
    
    conn = sqlite3.connect('mlb_engine.db')
    cursor = conn.cursor()
    
    # Target yesterday's date for empirical outcomes
    target_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    schedule_url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={target_date}"
    
    try:
        response = requests.get(schedule_url, timeout=15).json()
        games = response.get('dates', [])[0].get('games', []) if response.get('dates') else []
    except Exception as e:
        print(f"API Error fetching completed games: {e}")
        conn.close()
        return

    for game in games:
        # Process only finalized, non-simulated outcomes
        if game['status']['statusCode'] == 'F':
            away_team = game['teams']['away']['team']['name']
            home_team = game['teams']['home']['team']['name']
            actual_away_runs = game['teams']['away']['score']
            actual_home_runs = game['teams']['home']['score']
            
            # Retrieve initial forecast
            cursor.execute('''
                SELECT predicted_away_runs, predicted_home_runs 
                FROM Model_Forecasts 
                WHERE away_team = ? AND home_team = ?
                ORDER BY ROWID DESC LIMIT 1
            ''', (away_team, home_team))
            
            forecast = cursor.fetchone()
            
            if forecast:
                pred_away_runs, pred_home_runs = forecast
                
                # Identify discrepancies (Actual vs. Expected)
                away_delta = actual_away_runs - pred_away_runs
                home_delta = actual_home_runs - pred_home_runs
                
                # Define learning rate for evolution (e.g., 0.002 shift in OPS per run differential)
                learning_rate = 0.002
                away_ops_adjustment = round(away_delta * learning_rate, 4)
                home_ops_adjustment = round(home_delta * learning_rate, 4)
                
                # Evolve team offensive baselines in the database
                cursor.execute('''
                    UPDATE Team_Offense 
                    SET ops = ops + ?, updated_at = DATETIME('now')
                    WHERE team_name = ?
                ''', (away_ops_adjustment, away_team))
                
                cursor.execute('''
                    UPDATE Team_Offense 
                    SET ops = ops + ?, updated_at = DATETIME('now')
                    WHERE team_name = ?
                ''', (home_ops_adjustment, home_team))
                
                print(f"Evolved {away_team}: {away_delta:+.2f} runs -> OPS {away_ops_adjustment:+.4f}")
                print(f"Evolved {home_team}: {home_delta:+.2f} runs -> OPS {home_ops_adjustment:+.4f}")

    conn.commit()
    conn.close()
    print("Post-match analysis and evolution completed simulation-free.")

if __name__ == "__main__":
    execute_factual_post_mortem()
