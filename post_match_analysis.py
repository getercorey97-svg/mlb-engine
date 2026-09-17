import sqlite3
import requests
import numpy as np
from datetime import datetime, timedelta

def ensure_post_match_schema(conn):
    cursor = conn.cursor()
    cursor.executescript('''
        CREATE TABLE IF NOT EXISTS Post_Match_Analysis (
            game_pk INTEGER PRIMARY KEY,
            actual_winner TEXT,
            home_score INTEGER,
            away_score INTEGER,
            home_f5_score INTEGER,
            away_f5_score INTEGER,
            model_correct INTEGER,
            processed_at TEXT
        );
        CREATE TABLE IF NOT EXISTS Dynamic_Modifiers (
            team_name TEXT PRIMARY KEY,
            offensive_modifier REAL DEFAULT 1.0,
            pitching_modifier REAL DEFAULT 1.0,
            sample_count INTEGER DEFAULT 1,
            last_updated TEXT
        );
        CREATE TABLE IF NOT EXISTS Pitcher_Modifiers (
            pitcher_name TEXT PRIMARY KEY,
            k_modifier REAL DEFAULT 1.0,
            f5_run_modifier REAL DEFAULT 1.0,
            sample_count INTEGER DEFAULT 1,
            last_updated TEXT
        );
        CREATE TABLE IF NOT EXISTS Bullpen_Modifiers (
            team_name TEXT PRIMARY KEY,
            fatigue_modifier REAL DEFAULT 1.0,
            last_updated TEXT
        );
    ''')
    conn.commit()

def calculate_ewma(current_val, target_val, n_games, base_alpha=0.10):
    # Dynamic alpha based on sample size and error delta
    alpha = min(0.20, max(base_alpha, 2.0 / (n_games + 1) + 0.05))
    updated_val = (alpha * target_val) + ((1.0 - alpha) * current_val)
    return round(float(np.clip(updated_val, 0.50, 1.50)), 3), round(alpha, 3)

def run_post_match_analysis():
    print("Executing Factual Post-Mortem (Full Game & F5 Linescores)...")
    conn = sqlite3.connect('mlb_engine.db', timeout=30)
    cursor = conn.cursor()
    ensure_post_match_schema(conn)

    # Inspect games from the last 48 hours to capture completed linescores
    target_dates = [
        (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d'),
        datetime.now().strftime('%Y-%m-%d')
    ]
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for date_str in target_dates:
        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}&hydrate=probablePitcher,linescore"
        try:
            res = requests.get(url, timeout=15).json()
        except Exception:
            continue

        for date_data in res.get('dates', []):
            for game in date_data.get('games', []):
                if game.get('status', {}).get('abstractGameState') != 'Final':
                    continue

                game_pk = game['gamePk']
                home_team = game['teams']['home']['team']['name']
                away_team = game['teams']['away']['team']['name']
                home_score = game['teams']['home'].get('score', 0)
                away_score = game['teams']['away'].get('score', 0)

                actual_winner = home_team if home_score > away_score else away_team

                # Calculate F5 scores from linescore innings
                innings = game.get('linescore', {}).get('innings', [])
                h_f5 = sum(inn.get('home', {}).get('runs', 0) for inn in innings[:5])
                a_f5 = sum(inn.get('away', {}).get('runs', 0) for inn in innings[:5])

                home_sp = game['teams']['home'].get('probablePitcher', {}).get('fullName', 'Unknown Pitcher')
                away_sp = game['teams']['away'].get('probablePitcher', {}).get('fullName', 'Unknown Pitcher')

                # Winner Resolution with String Normalization
                cursor.execute("SELECT home_prob, away_prob, home_team, away_team FROM Model_Forecasts WHERE game_pk = ?", (game_pk,))
                forecast = cursor.fetchone()
                
                is_correct = 0
                if forecast and forecast[0] is not None:
                    h_prob, a_prob, f_home, f_away = forecast
                    pred_winner = f_home if h_prob >= a_prob else f_away
                    is_correct = 1 if pred_winner.strip().lower() == actual_winner.strip().lower() else 0

                cursor.execute('''
                    INSERT OR REPLACE INTO Post_Match_Analysis 
                    (game_pk, actual_winner, home_score, away_score, home_f5_score, away_f5_score, model_correct, processed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (game_pk, actual_winner, home_score, away_score, h_f5, a_f5, is_correct, now_str))

                # Team Offense & Pitching EWMA Evolution
                for team, scored, allowed in [(home_team, home_score, away_score), (away_team, away_score, home_score)]:
                    cursor.execute("SELECT offensive_modifier, pitching_modifier, sample_count FROM Dynamic_Modifiers WHERE team_name = ?", (team,))
                    row = cursor.fetchone()
                    off_mod, pitch_mod, n_count = row if row else (1.000, 1.000, 1)
                    n_count += 1

                    target_off = 1.0 + ((scored - 4.3) * 0.04)
                    target_pitch = 1.0 + ((allowed - 4.3) * 0.04)

                    new_off, a_off = calculate_ewma(off_mod, target_off, n_count)
                    new_pitch, a_pitch = calculate_ewma(pitch_mod, target_pitch, n_count)

                    cursor.execute('''
                        INSERT OR REPLACE INTO Dynamic_Modifiers 
                        (team_name, offensive_modifier, pitching_modifier, sample_count, last_updated)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (team, new_off, new_pitch, n_count, now_str))

                    print(f"  [EWMA Team] {team} Offense: {off_mod:.3f} -> {new_off:.3f} (N={n_count}, alpha={a_off})")
                    print(f"  [EWMA Team] {team} Pitching: {pitch_mod:.3f} -> {new_pitch:.3f} (N={n_count}, alpha={a_pitch})")

                # Pitcher F5 SP Modifier Evolution
                for pitcher, runs_allowed in [(home_sp, a_f5), (away_sp, h_f5)]:
                    if pitcher == 'Unknown Pitcher':
                        continue
                    cursor.execute("SELECT f5_run_modifier, sample_count FROM Pitcher_Modifiers WHERE pitcher_name = ?", (pitcher,))
                    p_row = cursor.fetchone()
                    sp_mod, p_count = p_row if p_row else (1.000, 1)
                    p_count += 1

                    target_sp = 1.0 + ((runs_allowed - 2.2) * 0.06)
                    new_sp, sp_alpha = calculate_ewma(sp_mod, target_sp, p_count)

                    cursor.execute('''
                        INSERT OR REPLACE INTO Pitcher_Modifiers 
                        (pitcher_name, k_modifier, f5_run_modifier, sample_count, last_updated)
                        VALUES (?, 1.0, ?, ?, ?)
                    ''', (pitcher, new_sp, p_count, now_str))
                    print(f"  [EWMA Pitcher] {pitcher} F5 SP Modifier: {sp_mod:.3f} -> {new_sp:.3f} (N={p_count}, alpha={sp_alpha})")

    conn.commit()

    # --- Aggregation and Status Output ---
    cursor.execute("SELECT COUNT(*), AVG(f5_run_modifier) FROM Pitcher_Modifiers")
    pitchers_count, avg_f5_mod = cursor.fetchone()
    avg_f5_mod = avg_f5_mod or 1.000

    cursor.execute("SELECT COUNT(*), AVG(offensive_modifier), AVG(pitching_modifier) FROM Dynamic_Modifiers")
    teams_count, avg_off, avg_pitch = cursor.fetchone()
    avg_off, avg_pitch = avg_off or 1.000, avg_pitch or 1.000

    cursor.execute("SELECT COUNT(*), AVG(fatigue_modifier) FROM Bullpen_Modifiers")
    bp_row = cursor.fetchone()
    bullpens_count = bp_row[0] if bp_row else 30
    avg_bp = bp_row[1] if bp_row and bp_row[1] else 1.000

    # Strict non-negative accuracy calculation
    cursor.execute('''
        SELECT 
            SUM(CASE WHEN model_correct = 1 THEN 1 ELSE 0 END),
            COUNT(*)
        FROM Post_Match_Analysis
        WHERE actual_winner IS NOT NULL
    ''')
    correct_row = cursor.fetchone()
    correct_games = correct_row[0] if correct_row and correct_row[0] is not None else 0
    total_games = correct_row[1] if correct_row and correct_row[1] is not None else 0
    accuracy = (correct_games / total_games) if total_games > 0 else 0.0

    print("\n=== ENGINE LEARNING & MEMORY STATUS ===")
    print(f"  Pitchers Remembered: {pitchers_count} (Avg F5 Modifier: {avg_f5_mod:.3f})")
    print(f"  Teams Remembered:    {teams_count} (Avg Offense: {avg_off:.3f}, Avg Pitching: {avg_pitch:.3f})")
    print(f"  Bullpens Tracked:    {bullpens_count} (Avg Fatigue: {avg_bp:.3f})")
    print(f"  Historical Accuracy: {accuracy:.2%} ({correct_games}/{total_games} games correct)")
    print("=======================================\n")

    conn.close()

if __name__ == "__main__":
    run_post_match_analysis()
