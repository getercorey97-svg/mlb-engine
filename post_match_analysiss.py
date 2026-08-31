import sqlite3
import requests
from datetime import datetime, timedelta

def update_dynamic_weights(cursor, team_name, predicted_runs, actual_runs, is_offense=True):
    """
    Calculates the Factual Error Delta and updates the Dynamic_Modifiers table.
    Learning rate temporarily widened to 0.05 to rapidly ingest the 1600-game dataset.
    """
    LEARNING_RATE = 0.05
    error_delta = actual_runs - predicted_runs
    
    cursor.execute('SELECT offensive_modifier, pitching_modifier FROM Dynamic_Modifiers WHERE team_name = ?', (team_name,))
    result = cursor.fetchone()
    
    if not result:
        return
        
    off_mod, pitch_mod = result
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if is_offense:
        new_off_mod = off_mod + (error_delta * LEARNING_RATE)
        new_off_mod = max(0.80, min(1.20, new_off_mod)) # Widened clamps to allow adaptation
        
        cursor.execute('''
            UPDATE Dynamic_Modifiers 
            SET offensive_modifier = ?, last_updated = ? 
            WHERE team_name = ?
        ''', (new_off_mod, current_time, team_name))
        print(f"  [Dynamic Weight Update] {team_name} Offensive Modifier: {off_mod:.3f} -> {new_off_mod:.3f} (Delta: {error_delta:+.2f})")
    else:
        new_pitch_mod = pitch_mod + (error_delta * LEARNING_RATE)
        new_pitch_mod = max(0.80, min(1.20, new_pitch_mod))
        
        cursor.execute('''
            UPDATE Dynamic_Modifiers 
            SET pitching_modifier = ?, last_updated = ? 
            WHERE team_name = ?
        ''', (new_pitch_mod, current_time, team_name))
        print(f"  [Dynamic Weight Update] {team_name} Pitching Modifier: {pitch_mod:.3f} -> {new_pitch_mod:.3f} (Delta: {error_delta:+.2f})")

def run_post_match_analysis():
    print("Executing Factual Post-Mortem: Pulling daily outcomes...")
    
    dates_to_check = [
        (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
        datetime.now().strftime('%Y-%m-%d')
    ]
    
    conn = sqlite3.connect('mlb_engine.db')
    cursor = conn.cursor()
    
    cursor.executescript('''
    CREATE TABLE IF NOT EXISTS Post_Match_Analysis (
        game_pk INTEGER PRIMARY KEY,
        actual_winner TEXT,
        home_score INTEGER,
        away_score INTEGER,
        model_correct INTEGER,
        processed_at TEXT
    );
    ''')
    
    for date_str in dates_to_check:
        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}"
        try:
            response = requests.get(url, timeout=10).json()
        except Exception as e:
            print(f"Could not retrieve schedule for {date_str}: {e}")
            continue

        for date_data in response.get('dates', []):
            for game in date_data.get('games', []):
                game_pk = game['gamePk']
                status = game['status']['abstractGameState']
                
                if status == 'Final':
                    cursor.execute("SELECT game_pk FROM Post_Match_Analysis WHERE game_pk = ?", (game_pk,))
                    if cursor.fetchone():
                        continue

                    teams = game.get('teams', {})
                    home_team = teams['home']['team']['name']
                    away_team = teams['away']['team']['name']
                    home_score = teams['home'].get('score', 0)
                    away_score = teams['away'].get('score', 0)
                    
                    actual_winner = home_team if home_score > away_score else away_team
                    
                    try:
                        cursor.execute('''
                            SELECT home_prob, away_prob, predicted_home_runs, predicted_away_runs 
                            FROM Model_Forecasts 
                            WHERE game_pk = ?
                        ''', (game_pk,))
                        forecast = cursor.fetchone()
                    except sqlite3.OperationalError:
                        forecast = None
                    
                    if forecast:
                        home_prob, away_prob, pred_home_runs, pred_away_runs = forecast
                        predicted_winner = home_team if home_prob > away_prob else away_team
                        model_correct = 1 if predicted_winner == actual_winner else 0
                        
                        cursor.execute('''
                        INSERT OR REPLACE INTO Post_Match_Analysis (game_pk, actual_winner, home_score, away_score, model_correct, processed_at)
                        VALUES (?, ?, ?, ?, ?, ?)
                        ''', (game_pk, actual_winner, home_score, away_score, model_correct, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                        
                        print(f"Post-Mortem Game {game_pk}: Winner: {actual_winner} ({home_score}-{away_score}) | Model Correct: {bool(model_correct)}")
                        
                        if pred_home_runs is not None and pred_away_runs is not None:
                            update_dynamic_weights(cursor, home_team, pred_home_runs, home_score, is_offense=True)
                            update_dynamic_weights(cursor, away_team, pred_home_runs, home_score, is_offense=False)
                            
                            update_dynamic_weights(cursor, away_team, pred_away_runs, away_score, is_offense=True)
                            update_dynamic_weights(cursor, home_team, pred_away_runs, away_score, is_offense=False)

    conn.commit()
    conn.close()
    print("Post-match analysis completed.")

if __name__ == "__main__":
    run_post_match_analysis()
