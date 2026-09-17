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
            appearance_count INTEGER DEFAULT 0,
            last_updated TEXT
        );
        CREATE TABLE IF NOT EXISTS Pitcher_Modifiers (
            pitcher_name TEXT PRIMARY KEY,
            k_modifier REAL DEFAULT 1.0,
            f5_run_modifier REAL DEFAULT 1.0,
            appearance_count INTEGER DEFAULT 0,
            last_updated TEXT
        );
        CREATE TABLE IF NOT EXISTS Bullpen_Fatigue (
            team_name TEXT PRIMARY KEY,
            fatigue_multiplier REAL DEFAULT 1.00,
            last_updated TEXT
        );
    ''')
    conn.commit()

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

                if home_score == away_score:
                    continue

                actual_winner = home_team if home_score > away_score else away_team

                # Calculate F5 scores from linescore innings
                innings = game.get('linescore', {}).get('innings', [])
                h_f5 = sum(inn.get('home', {}).get('runs', 0) for inn in innings[:5])
                a_f5 = sum(inn.get('away', {}).get('runs', 0) for inn in innings[:5])
                h_late = max(0, home_score - h_f5)
                a_late = max(0, away_score - a_f5)

                home_sp = game['teams']['home'].get('probablePitcher', {}).get('fullName', 'Unknown Pitcher')
                away_sp = game['teams']['away'].get('probablePitcher', {}).get('fullName', 'Unknown Pitcher')

                # Fetch Forecasts for error-based EWMA updates
                cursor.execute("""
                    SELECT home_prob, away_prob, predicted_home_runs, predicted_away_runs 
                    FROM Model_Forecasts WHERE game_pk = ?
                """, (game_pk,))
                forecast = cursor.fetchone()

                cursor.execute("""
                    SELECT f5_exp_home_runs, f5_exp_away_runs 
                    FROM F5_Forecasts WHERE game_pk = ?
                """, (game_pk,))
                f5_forecast = cursor.fetchone()

                pred_h_runs = forecast[2] if (forecast and forecast[2] is not None) else 4.45
                pred_a_runs = forecast[3] if (forecast and forecast[3] is not None) else 4.25
                pred_h_f5 = f5_forecast[0] if (f5_forecast and f5_forecast[0] is not None) else pred_h_runs * 0.55
                pred_a_f5 = f5_forecast[1] if (f5_forecast and f5_forecast[1] is not None) else pred_a_runs * 0.55
                pred_h_late = pred_h_runs * 0.45
                pred_a_late = pred_a_runs * 0.45

                is_correct = None
                if forecast and forecast[0] is not None:
                    h_prob, a_prob = forecast[0], forecast[1]
                    pred_winner = home_team if h_prob >= a_prob else away_team
                    is_correct = 1 if pred_winner.strip().lower() == actual_winner.strip().lower() else 0

                cursor.execute('''
                    INSERT OR REPLACE INTO Post_Match_Analysis 
                    (game_pk, actual_winner, home_score, away_score, home_f5_score, away_f5_score, model_correct, processed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (game_pk, actual_winner, home_score, away_score, h_f5, a_f5, is_correct, now_str))

                # 1. Starting Pitcher F5 EWMA Updates (Aligned with backtest_engine.py)
                for pitcher, pred_f5, act_f5 in [(home_sp, pred_a_f5, a_f5), (away_sp, pred_h_f5, h_f5)]:
                    if pitcher == 'Unknown Pitcher':
                        continue
                    cursor.execute("SELECT f5_run_modifier, appearance_count FROM Pitcher_Modifiers WHERE pitcher_name = ?", (pitcher,))
                    p_row = cursor.fetchone()
                    old_mod, p_count = p_row if p_row else (1.000, 0)
                    p_count += 1

                    err = act_f5 - pred_f5
                    alpha = min(0.12, 0.03 + (abs(err) * 0.01))
                    new_mod = max(0.70, min(1.30, alpha * (old_mod + err * 0.05) + (1.0 - alpha) * old_mod))

                    cursor.execute('''
                        INSERT OR REPLACE INTO Pitcher_Modifiers 
                        (pitcher_name, k_modifier, f5_run_modifier, appearance_count, last_updated)
                        VALUES (?, 1.0, ?, ?, ?)
                    ''', (pitcher, round(new_mod, 4), p_count, now_str))
                    print(f"  [EWMA Pitcher] {pitcher} F5 SP Modifier: {old_mod:.3f} -> {new_mod:.3f} (N={p_count}, alpha={alpha:.3f})")

                # 2. Team Offense & Pitching Late-Inning Updates (Aligned with backtest_engine.py)
                for team, pred_late, act_late, is_off in [
                    (home_team, pred_h_late, h_late, True),
                    (away_team, pred_h_late, h_late, False),
                    (away_team, pred_a_late, a_late, True),
                    (home_team, pred_a_late, a_late, False)
                ]:
                    cursor.execute("SELECT offensive_modifier, pitching_modifier, appearance_count FROM Dynamic_Modifiers WHERE team_name = ?", (team,))
                    row = cursor.fetchone()
                    off_mod, pitch_mod, n_count = row if row else (1.000, 1.000, 0)
                    n_count += 1

                    err = act_late - pred_late
                    alpha = min(0.10, 0.02 + (abs(err) * 0.008))

                    if is_off:
                        new_off = max(0.70, min(1.30, alpha * (off_mod + err * 0.04) + (1.0 - alpha) * off_mod))
                        new_pitch = pitch_mod
                    else:
                        new_off = off_mod
                        new_pitch = max(0.70, min(1.30, alpha * (pitch_mod + err * 0.04) + (1.0 - alpha) * pitch_mod))

                    cursor.execute('''
                        INSERT OR REPLACE INTO Dynamic_Modifiers 
                        (team_name, offensive_modifier, pitching_modifier, appearance_count, last_updated)
                        VALUES (?, ?, ?, ?, ?)
                    ''', (team, round(new_off, 4), round(new_pitch, 4), n_count, now_str))
                    print(f"  [EWMA Team] {team} {'Offense' if is_off else 'Pitching'}: {(off_mod if is_off else pitch_mod):.3f} -> {(new_off if is_off else new_pitch):.3f} (N={n_count}, alpha={alpha:.3f})")

                # 3. Bullpen Fatigue Workload Updates (Aligned with backtest_engine.py)
                for team, pred_late, act_late in [(away_team, pred_h_late, h_late), (home_team, pred_a_late, a_late)]:
                    cursor.execute("SELECT fatigue_multiplier FROM Bullpen_Fatigue WHERE team_name = ?", (team,))
                    f_row = cursor.fetchone()
                    old_fatigue = f_row[0] if f_row else 1.000

                    err = act_late - pred_late
                    alpha = min(0.10, 0.02 + (abs(err) * 0.008))
                    new_fatigue = max(0.80, min(1.25, alpha * (old_fatigue + err * 0.04) + (1.0 - alpha) * old_fatigue))

                    cursor.execute('''
                        INSERT OR REPLACE INTO Bullpen_Fatigue (team_name, fatigue_multiplier, last_updated)
                        VALUES (?, ?, ?)
                    ''', (team, round(new_fatigue, 4), now_str))

    conn.commit()

    # --- Aggregation and Status Output ---
    cursor.execute("SELECT COUNT(*), AVG(f5_run_modifier) FROM Pitcher_Modifiers")
    pitchers_count, avg_f5_mod = cursor.fetchone()
    avg_f5_mod = avg_f5_mod or 1.000

    cursor.execute("SELECT COUNT(*), AVG(offensive_modifier), AVG(pitching_modifier) FROM Dynamic_Modifiers")
    teams_count, avg_off, avg_pitch = cursor.fetchone()
    avg_off, avg_pitch = avg_off or 1.000, avg_pitch or 1.000

    cursor.execute("SELECT COUNT(*), AVG(fatigue_multiplier) FROM Bullpen_Fatigue")
    bp_row = cursor.fetchone()
    bullpens_count = bp_row[0] if bp_row else 30
    avg_bp = bp_row[1] if bp_row and bp_row[1] else 1.000

    # Strict non-negative accuracy calculation
    cursor.execute('''
        SELECT 
            SUM(CASE WHEN model_correct = 1 THEN 1 ELSE 0 END),
            COUNT(model_correct)
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
