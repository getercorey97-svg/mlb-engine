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
    return float(np.clip(num / denom, 0.08, 0.48))

def run_pitcher_props():
    conn = get_db_connection()
    c = conn.cursor()

    dl_cols = [r[1] for r in c.execute("PRAGMA table_info(Daily_Lineups)").fetchall()]
    if not dl_cols:
        print("[PITCHER PROPS] Daily_Lineups table not found.")
        conn.close()
        return

    away_col = "away_pitcher" if "away_pitcher" in dl_cols else ("away_sp" if "away_sp" in dl_cols else None)
    home_col = "home_pitcher" if "home_pitcher" in dl_cols else ("home_sp" if "home_sp" in dl_cols else None)
    status_col = "status" if "status" in dl_cols else ("lineup_status" if "lineup_status" in dl_cols else None)
    ump_col = "umpire_k_mod" if "umpire_k_mod" in dl_cols else None

    where_clause = f"WHERE {status_col} != 'Final'" if status_col else ""
    query = f"""
        SELECT game_pk, away_team, home_team, 
               {away_col if away_col else "''"} AS away_sp, 
               {home_col if home_col else "''"} AS home_sp,
               {ump_col if ump_col else "1.000"} AS umpire_k_mod
        FROM Daily_Lineups
        {where_clause}
    """
    games = c.execute(query).fetchall()

    if not games:
        print("[PITCHER PROPS] No scheduled matchups found in Daily_Lineups.")
        conn.close()
        return

    c.execute("DELETE FROM Pitcher_K_Forecasts")

    ps_cols = [r[1] for r in c.execute("PRAGMA table_info(Pitcher_Stats)").fetchall()]
    p_name_col = "pitcher_name" if "pitcher_name" in ps_cols else ("player_name" if "player_name" in ps_cols else "last_name")

    total_pitchers = 0
    for g in games:
        pk = g["game_pk"]
        ump_k_mod = float(g["umpire_k_mod"]) if g["umpire_k_mod"] else 1.000

        pairings = [
            (g["away_sp"], g["away_team"], g["home_team"]),
            (g["home_sp"], g["home_team"], g["away_team"])
        ]

        for sp_name, team, opp_team in pairings:
            if not sp_name or sp_name in ("TBD", "Unknown", ""):
                continue

            # Ingest Latent State from Kalman Filter Table
            c.execute("INSERT OR IGNORE INTO Pitcher_Kalman_State (pitcher_name, latent_k_modifier, variance_p, process_noise_q) VALUES (?, 1.000, 0.040, 0.0025)", (sp_name,))
            kalman_row = c.execute("SELECT latent_k_modifier, variance_p, process_noise_q FROM Pitcher_Kalman_State WHERE pitcher_name = ?", (sp_name,)).fetchone()
            
            # Prior State Projection: Variance increases with time/gap
            latent_k_mod = float(kalman_row["latent_k_modifier"])
            p_prior = float(kalman_row["variance_p"]) + float(kalman_row["process_noise_q"])

            sp_row = c.execute(f"""
                SELECT whiff_rate, pitches_per_bf 
                FROM Pitcher_Stats 
                WHERE {p_name_col} = ? OR ? LIKE '%' || {p_name_col}
                LIMIT 1
            """, (sp_name, sp_name)).fetchone()

            p_bf = float(sp_row["pitches_per_bf"]) if sp_row and sp_row["pitches_per_bf"] else 3.90
            p_whiff = float(sp_row["whiff_rate"]) if sp_row and sp_row["whiff_rate"] else 0.245

            # Base strikeout capability modified by latent state
            sp_base_k = np.clip(p_whiff * 0.92, 0.12, 0.38) * latent_k_mod

            opp_k_rate = LEAGUE_K_RATE
            tables = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
            if "Daily_Batters" in tables:
                db_cols = [r[1] for r in c.execute("PRAGMA table_info(Daily_Batters)").fetchall()]
                if "k_rate" in db_cols:
                    lineup_rows = c.execute("""
                        SELECT AVG(k_rate) as avg_k 
                        FROM Daily_Batters 
                        WHERE game_pk = ? AND team_name = ?
                    """, (pk, opp_team)).fetchone()
                    if lineup_rows and lineup_rows["avg_k"]:
                        opp_k_rate = float(lineup_rows["avg_k"])

            matchup_k_rate = calculate_log5_k(sp_base_k, opp_k_rate) * ump_k_mod

            expected_pitches = 88.0
            expected_bf = expected_pitches / p_bf

            # Kinematic fatigue curve: TTOP degradation past pitch 65
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
    print(f"[KALMAN ENGINE] Successfully forecasted {total_pitchers} starting pitcher strikeout lines.")

if __name__ == "__main__":
    run_pitcher_props()
