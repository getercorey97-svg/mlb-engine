import sqlite3
import requests
from datetime import datetime, timedelta

def update_bullpen_fatigue(cursor, team_name, predicted_late, actual_late):
    """Updates bullpen fatigue strictly against late-inning errors using EWMA."""
    if not team_name or team_name == 'Unknown':
        return
    error_delta = actual_late - predicted_late
    cursor.execute('SELECT fatigue_multiplier FROM Bullpen_Fatigue WHERE team_name = ?', (team_name,))
    res = cursor.fetchone()
    fatigue = res[0] if res else 1.00
    
    alpha = min(0.30, 0.10 + (abs(error_delta) * 0.02))
    target_fatigue = fatigue + (error_delta * 0.15)
    new_fatigue = max(0.70, min(1.30, alpha * target_fatigue + (1.0 - alpha) * fatigue))
    
    cursor.execute('''
        INSERT OR REPLACE INTO Bullpen_Fatigue (team_name, fatigue_multiplier)
        VALUES (?, ?)
    ''', (team_name, new_fatigue))
    print(f"  [EWMA Bullpen] {team_name} Fatigue: {fatigue:.3f} -> {new_fatigue:.3f}")

def update_dynamic_weights(cursor, name, predicted_runs, actual_runs, is_offense=True, is_pitcher=False):
    """Calculates Error Delta and applies an Adaptive EWMA Learning Rate with Bayesian Shrinkage tracking."""
    if not name or name == 'Unknown':
        return

    error_delta = actual_runs - predicted_runs
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    # EWMA smoothing factor alpha (higher error delta increases responsiveness)
    alpha = min(0.30, 0.10 + (abs(error_delta) * 0.02))
    
    if is_pitcher:
        cursor.execute('SELECT f5_run_modifier, k_modifier, appearance_count FROM Pitcher_Modifiers WHERE pitcher_name = ?', (name,))
        result = cursor.fetchone()
        mod = result[0] if result else 1.0
        k_mod = result[1] if result else 1.0
        count = result[2] if result else 0
        
        target_mod = mod + (error_delta * 0.15)
        new_mod = max(0.60, min(1.40, alpha * target_mod + (1.0 - alpha) * mod))
        new_count = count + 1
        
        cursor.execute('''
            INSERT OR REPLACE INTO Pitcher_Modifiers (pitcher_name, k_modifier, f5_run_modifier, appearance_count, last_updated) 
            VALUES (?, ?, ?, ?, ?)
        ''', (name, k_mod, new_mod, new_count, current_time))
        print(f"  [EWMA Pitcher] {name} F5 SP Modifier: {mod:.3f} -> {new_mod:.3f} (N={new_count}, alpha={alpha:.3f})")
        return

    cursor.execute('SELECT offensive_modifier, pitching_modifier, appearance_count FROM Dynamic_Modifiers WHERE team_name = ?', (name,))
    result = cursor.fetchone()
    if not result:
        off_mod, pitch_mod, count = 1.0, 1.0, 0
        cursor.execute('''
            INSERT OR IGNORE INTO Dynamic_Modifiers (team_name, offensive_modifier, pitching_modifier, appearance_count, last_updated)
            VALUES (?, 1.0, 1.0, 0, ?)
        ''', (name, current_time))
    else:
        off_mod, pitch_mod, count = result
    
    new_count = count + 1
    if is_offense:
        target_mod = off_mod + (error_delta * 0.15)
        new_off_mod = max(0.60, min(1.40, alpha * target_mod + (1.0 - alpha) * off_mod))
        cursor.execute('UPDATE Dynamic_Modifiers SET offensive_modifier = ?, appearance_count = ?, last_updated = ? WHERE team_name = ?', (new_off_mod, new_count, current_time, name))
        print(f"  [EWMA Team] {name} Offense: {off_mod:.3f} -> {new_off_mod:.3f} (N={new_count}, alpha={alpha:.3f})")
    else:
        target_mod = pitch_mod + (error_delta * 0.15)
        new_pitch_mod = max(0.60, min(1.40, alpha * target_mod + (1.0 - alpha) * pitch_mod))
        cursor.execute('UPDATE Dynamic_Modifiers SET pitching_modifier = ?, appearance_count = ?, last_updated = ? WHERE team_name = ?', (new_pitch_mod, new_count, current_time, name))
        print(f"  [EWMA Team] {name} Pitching: {pitch_mod:.3f} -> {new_pitch_mod:.3f} (N={new_count}, alpha={alpha:.3f})")

def run_post_match_analysis():
    print("Executing Factual Post-Mortem (Full Game & F5 Linescores)...")
    dates_to_check = [(datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'), datetime.now().strftime('%Y-%m-%d')]
    
    conn = sqlite3.connect('mlb_engine.db', timeout=30)
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA busy_timeout=10000;")
    
    cursor.executescript('''
    CREATE TABLE IF NOT EXISTS Post_Match_Analysis (
        game_pk INTEGER PRIMARY KEY, actual_winner TEXT, home_score INTEGER, away_score INTEGER, 
        home_f5_score INTEGER, away_f5_score INTEGER, model_correct INTEGER, processed_at TEXT
    );
    CREATE TABLE IF NOT EXISTS Pitcher_Modifiers (
        pitcher_name TEXT PRIMARY KEY, k_modifier REAL DEFAULT 1.0, f5_run_modifier REAL DEFAULT 1.0, appearance_count INTEGER DEFAULT 0, last_updated TEXT
    );
    CREATE TABLE IF NOT EXISTS Dynamic_Modifiers (
        team_name TEXT PRIMARY KEY, offensive_modifier REAL DEFAULT 1.0, pitching_modifier REAL DEFAULT 1.0, appearance_count INTEGER DEFAULT 0, last_updated TEXT
    );
    ''')

    for col in ["home_f5_score INTEGER", "away_f5_score INTEGER"]:
        try:
            cursor.execute(f"ALTER TABLE Post_Match_Analysis ADD COLUMN {col}")
        except sqlite3.OperationalError:
            pass
    
    for date_str in dates_to_check:
        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}&hydrate=linescore,probablePitcher"
        try:
            res = requests.get(url, timeout=10)
            res.raise_for_status()
            response = res.json()
        except Exception:
            continue

        for date_data in response.get('dates', []):
            for game in date_data.get('games', []):
                game_pk = game['gamePk']
                if game.get('status', {}).get('abstractGameState') != 'Final':
                    continue
                
                cursor.execute("SELECT game_pk FROM Post_Match_Analysis WHERE game_pk = ?", (game_pk,))
                if cursor.fetchone():
                    continue

                teams = game.get('teams', {})
                home_team = teams.get('home', {}).get('team', {}).get('name', 'Unknown Home Team')
                away_team = teams.get('away', {}).get('team', {}).get('name', 'Unknown Away Team')
                home_score = teams.get('home', {}).get('score', 0)
                away_score = teams.get('away', {}).get('score', 0)
                home_p = teams.get('home', {}).get('probablePitcher', {}).get('fullName', 'Unknown')
                away_p = teams.get('away', {}).get('probablePitcher', {}).get('fullName', 'Unknown')
                
                actual_winner = home_team if home_score > away_score else away_team
                
                # F5 Linescore Extraction
                linescore = game.get('linescore', {}).get('innings', [])
                h_f5, a_f5 = 0, 0
                for inning in linescore[:5]:
                    h_f5 += inning.get('home', {}).get('runs') or 0
                    a_f5 += inning.get('away', {}).get('runs') or 0
                
                # Late Inning (6th through 9th) Extraction
                h_late, a_late = 0, 0
                for inning in linescore[5:9]:
                    h_late += inning.get('home', {}).get('runs') or 0
                    a_late += inning.get('away', {}).get('runs') or 0
                
                try:
                    cursor.execute('SELECT home_prob, away_prob, predicted_home_runs, predicted_away_runs FROM Model_Forecasts WHERE game_pk = ?', (game_pk,))
                    fg_fc = cursor.fetchone()
                except sqlite3.OperationalError:
                    fg_fc = None
                
                try:
                    cursor.execute('SELECT f5_exp_home_runs, f5_exp_away_runs FROM F5_Forecasts WHERE game_pk = ?', (game_pk,))
                    f5_fc = cursor.fetchone()
                except sqlite3.OperationalError:
                    f5_fc = None
                
                model_correct = 0
                if fg_fc:
                    predicted_winner = home_team if fg_fc[0] > fg_fc[1] else away_team
                    model_correct = 1 if predicted_winner == actual_winner else 0
                    
                    # Decoupled Multi-Target Loss Optimization:
                    # Team offensive modifiers and bullpen fatigue weights optimize strictly against late-inning (6th through 9th) errors.
                    pred_home_late = fg_fc[2] * 0.45
                    pred_away_late = fg_fc[3] * 0.45
                    
                    update_dynamic_weights(cursor, home_team, pred_home_late, h_late, is_offense=True)
                    update_dynamic_weights(cursor, away_team, pred_home_late, h_late, is_offense=False)
                    update_dynamic_weights(cursor, away_team, pred_away_late, a_late, is_offense=True)
                    update_dynamic_weights(cursor, home_team, pred_away_late, a_late, is_offense=False)
                    
                    update_bullpen_fatigue(cursor, away_team, pred_home_late, h_late)
                    update_bullpen_fatigue(cursor, home_team, pred_away_late, a_late)
                
                if f5_fc:
                    # Starting pitcher run modifiers strictly optimize against First 5 (F5) inning scoring errors
                    update_dynamic_weights(cursor, home_p, f5_fc[1], a_f5, is_pitcher=True)
                    update_dynamic_weights(cursor, away_p, f5_fc[0], h_f5, is_pitcher=True)
                else:
                    if fg_fc:
                        pred_home_f5 = fg_fc[2] * 0.55
                        pred_away_f5 = fg_fc[3] * 0.55
                        update_dynamic_weights(cursor, home_p, pred_away_f5, a_f5, is_pitcher=True)
                        update_dynamic_weights(cursor, away_p, pred_home_f5, h_f5, is_pitcher=True)

                cursor.execute('''
                    INSERT OR REPLACE INTO Post_Match_Analysis (game_pk, actual_winner, home_score, away_score, home_f5_score, away_f5_score, model_correct, processed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (game_pk, actual_winner, home_score, away_score, h_f5, a_f5, model_correct, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))

    conn.commit()
    cursor.execute("PRAGMA wal_checkpoint(TRUNCATE);")
    conn.close()
    print("Post-match analysis & F5 micro-evolution completed.")

if __name__ == "__main__":
    run_post_match_analysis()
