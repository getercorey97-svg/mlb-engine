import sqlite3
import math
import numpy as np
from scipy.stats import poisson

LEAGUE_K_RATE = 0.224

def get_db_connection():
    conn = sqlite3.connect("mlb_engine.db")
    conn.row_factory = sqlite3.Row
    return conn

def calculate_log5_k(pitcher_k_rate, lineup_k_rate, lg_k=LEAGUE_K_RATE):
    p = float(pitcher_k_rate)
    b = float(lineup_k_rate)
    num = (p * b) / lg_k
    denom = num + (((1.0 - p) * (1.0 - b)) / (1.0 - lg_k))
    return float(np.clip(num / denom, 0.08, 0.45))

def run_pitcher_props():
    conn = get_db_connection()
    c = conn.cursor()

    games = c.execute("""
        SELECT game_pk, away_team, home_team, away_sp, home_sp, umpire_k_mod
        FROM Daily_Lineups
        WHERE lineup_status IN ('Confirmed', 'Pending')
    """).fetchall()

    if not games:
        print("[PITCHER PROPS] No active games found in Daily_Lineups.")
        conn.close()
        return

    c.execute("DELETE FROM Pitcher_K_Forecasts")

    total_pitchers = 0
    for g in games:
        pk = g["game_pk"]
        ump_k_mod = float(g["umpire_k_mod"]) if "umpire_k_mod" in g.keys() and g["umpire_k_mod"] else 1.000

        pairings = [
            (g["away_sp"], g["away_team"], g["home_team"]),
            (g["home_sp"], g["home_team"], g["away_team"])
        ]

        for sp_name, team, opp_team in pairings:
            if not sp_name or sp_name == "TBD":
                continue

            sp_row = c.execute("""
                SELECT k_modifier, whiff_rate, pitches_per_bf, sample_starts 
                FROM Pitcher_Stats 
                WHERE pitcher_name = ?
            """, (sp_name,)).fetchone()

            k_mod = float(sp_row["k_modifier"]) if sp_row and sp_row["k_modifier"] else 1.000
            p_bf = float(sp_row["pitches_per_bf"]) if sp_row and sp_row["pitches_per_bf"] else 3.90
            p_whiff = float(sp_row["whiff_rate"]) if sp_row and sp_row["whiff_rate"] else 0.245

            sp_base_k = np.clip(p_whiff * 0.92, 0.12, 0.38) * k_mod

            lineup_rows = c.execute("""
                SELECT AVG(k_rate) as avg_k 
                FROM Daily_Batters 
                WHERE game_pk = ? AND team_name = ? AND is_starter = 1
            """, (pk, opp_team)).fetchone()
            
            opp_k_rate = float(lineup_rows["avg_k"]) if lineup_rows and lineup_rows["avg_k"] else LEAGUE_K_RATE

            matchup_k_rate = calculate_log5_k(sp_base_k, opp_k_rate) * ump_k_mod

            expected_pitches = 88.0
            expected_bf = expected_pitches / p_bf

            # Kinematic fatigue TTOP penalty after pitch 65
            fatigue_penalty = -0.0015 * max(0.0, expected_pitches - 65.0)
            adjusted_k_rate = max(0.05, matchup_k_rate + fatigue_penalty)

            exp_k = round(expected_bf * adjusted_k_rate, 2)
            k_line = math.floor(exp_k) + 0.5

            p_under = float(poisson.cdf(math.floor(k_line), exp_k))
            p_over = float(1.0 - p_under)

            c.execute("""
                INSERT OR REPLACE INTO Pitcher_K_Forecasts (
                    game_pk, pitcher_name, team_name, opponent_team,
                    projected_pitches, projected_bf, expected_k,
                    k_line, over_prob, under_prob
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                pk, sp_name, team, opp_team,
                round(expected_pitches, 1), round(expected_bf, 1), exp_k,
                k_line, round(p_over, 4), round(p_under, 4)
            ))
            total_pitchers += 1

    conn.commit()
    conn.close()
    print(f"[SUCCESS] Synthesized {total_pitchers} Pitcher Strikeout Projections into Pitcher_K_Forecasts.")

if __name__ == "__main__":
    run_pitcher_props()
