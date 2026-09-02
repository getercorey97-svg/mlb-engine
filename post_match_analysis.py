import sqlite3
import requests
from datetime import datetime, timedelta

def update_dynamic_weights(cursor, name, predicted_runs, actual_runs, is_offense=True, is_pitcher=False):
    """Calculates Error Delta and applies an Adaptive Learning Rate with a 0.47 Volatility Ceiling."""
    error_delta = actual_runs - predicted_runs
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # Adaptive Learning Rate: Scales dynamically based on error magnitude
    base_lr = 0.015
    adaptive_lr = min(0.08, base_lr + (abs(error_delta) * 0.005))
    
    if is_pitcher:
        cursor.execute('SELECT f5_run_modifier FROM Pitcher_Modifiers WHERE pitcher_name = ?', (name,))
        result = cursor.fetchone()
        mod = result[0] if result else 1.0
        
        # 0.47 Volatility Ceiling (Limits modifiers to +/- 47% from baseline 1.0)
        new_mod = max(0.53, min(1.47, mod + (error_delta * adaptive_lr)))
        
        cursor.execute('''
            INSERT OR REPLACE INTO Pitcher_Modifiers (pitcher_name, f5_run_modifier, last_updated) 
            VALUES (?, ?, ?)
        ''', (name, new_mod, current_time))
        print(f"  [Micro-Evolution] {name} F5 SP Modifier: {mod:.3f} -> {new_mod:.3f} (LR: {adaptive_lr:.3f})")
        return

    cursor.execute('SELECT offensive_modifier, pitching_modifier FROM Dynamic_Modifiers WHERE team_name = ?', (name,))
    result = cursor.fetchone()
    if not result: return
    off_mod, pitch_mod = result
    
    if is_offense:
        new_off_mod = max(0.53, min(1.47, off_mod + (error_delta * adaptive_lr)))
        cursor.execute('UPDATE Dynamic_Modifiers SET offensive_modifier = ?, last_updated = ? WHERE team_name = ?', (new_off_mod, current_time, name))
        print(f"  [Dynamic Update] {name} Offense: {off_mod:.3f} -> {new_off_mod:.3f} (LR: {adaptive_lr:.3f})")
    else:
        new_pitch_mod = max(0.53, min(1.47, pitch_mod + (error_delta * adaptive_lr)))
        cursor.execute('UPDATE Dynamic_Modifiers SET pitching_modifier = ?, last_updated = ? WHERE team_name = ?', (new_pitch_mod, current_time, name))
        print(f"  [Dynamic Update] {name} Pitching: {pitch_mod:.3f} -> {new_pitch_mod:.3f} (LR: {adaptive_lr:.3f})")

def run_post_match_analysis():
    print("Executing Factual Post-Mortem (Full Game & F5 Linescores)...")
    dates_to_check = [(datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'), datetime.now().strftime('%Y-%m-%d')]
    
    conn = sqlite3.connect('mlb_engine.db')
    cursor = conn.cursor()
    
    cursor.executescript('''
    CREATE TABLE IF NOT EXISTS Post_Match_Analysis (
        game_pk INTEGER PRIMARY KEY, actual_winner TEXT, home_score INTEGER, away_score INTEGER, 
        home_f5_score INTEGER, away_f5_score INTEGER, model_correct INTEGER, processed_at TEXT
    );
    CREATE TABLE IF NOT EXISTS Pitcher_Modifiers (
        pitcher_name TEXT PRIMARY KEY, k_modifier REAL DEFAULT 1.0, f5_run_modifier REAL DEFAULT 1.0, last_updated TEXT
    );
    ''')

    for col in ["home_f5_score INTEGER", "away_f5_score INTEGER"]:
        try:
            cursor.execute(f"ALTER TABLE Post_Match_Analysis ADD COLUMN {col}")
        except sqlite3.OperationalError:
            pass
    
    for date_str in dates_to_check:
        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}&hydrate=linescore,probablePitcher"
        try: response = requests.get(url, timeout=10).json()
        except Exception as e: continue

        for date_data in response.get('dates', []):
            for game in date_data.get('games', []):
                game_pk = game['gamePk']
                if game['status']['abstractGameState'] != 'Final': continue
                
                cursor.execute("SELECT game_pk FROM Post_Match_Analysis WHERE game_pk = ?", (game_pk,))
                if cursor.fetchone(): continue

                teams = game.get('teams', {})
                home_team, away_team = teams['home']['team']['name'], teams['away']['team']['name']
                home_score, away_score = teams['home'].get('score', 0), teams['away'].get('score', 0)
                home_p = teams['home'].get('probablePitcher', {}).get('fullName', 'Unknown')
                away_p = teams['away'].get('probablePitcher', {}).get('fullName', 'Unknown')
                
                actual_winner = home_team if home_score > away_score else away_team
                
                # F5 Linescore Extraction
                linescore = game.get('linescore', {}).get('innings', [])
                h_f5, a_f5 = 0, 0
                for inning in linescore[:5]:
                    h_f5 += inning.get('home', {}).get('runs', 0)
                    a_f5 += inning.get('away', {}).get('runs', 0)
                
                try:
                    cursor.execute('SELECT home_prob, away_prob, predicted_home_runs, predicted_away_runs FROM Model_Forecasts WHERE game_pk = ?', (game_pk,))
                    fg_fc = cursor.fetchone()
                except sqlite3.OperationalError: fg_fc = None
                
                try:
                    cursor.execute('SELECT f5_exp_home_runs, f5_exp_away_runs FROM F5_Forecasts WHERE game_pk = ?', (game_pk,))
                    f5_fc = cursor.fetchone()
                except sqlite3.OperationalError: f5_fc = None
                
                model_correct = 0
                if fg_fc:
                    predicted_winner = home_team if fg_fc[0] > fg_fc[1] else away_team
                    model_correct = 1 if predicted_winner == actual_winner else 0
                    update_dynamic_weights(cursor, home_team, fg_fc[2], home_score, is_offense=True)
                    update_dynamic_weights(cursor, away_team, fg_fc[2], home_score, is_offense=False)
                    update_dynamic_weights(cursor, away_team, fg_fc[3], away_score, is_offense=True)
                    update_dynamic_weights(cursor, home_team, fg_fc[3], away_score, is_offense=False)
                
                if f5_fc:
                    update_dynamic_weights(cursor, home_p, f5_fc[0], away_f5, is_pitcher=True)
                    update_dynamic_weights(cursor, away_p, f5_fc[1], home_f5, is_pitcher=True)

                cursor.execute('''
                    INSERT OR REPLACE INTO Post_Match_Analysis (game_pk, actual_winner, home_score, away_score, home_f5_score, away_f5_score, model_correct, processed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (game_pk, actual_winner, home_score, away_score, h_f5, a_f5, model_correct, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    conn.commit()
    conn.close()
    print("Post-match analysis completed.")

if __name__ == "__main__":
    run_post_match_analysis()
