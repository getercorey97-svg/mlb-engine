import sqlite3
import math
import requests
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

def fetch_pitcher_live_k_stats(sp_name):
    """Fallback to MLB Stats API to fetch true season K% and pitch workload if local DB record is missing."""
    try:
        search_url = f"https://statsapi.mlb.com/api/v1/people/search?names={requests.utils.quote(sp_name)}&sportIds=1"
        res = requests.get(search_url, timeout=4).json()
        people = res.get("people", [])
        if not people:
            return None
        pid = people[0]["id"]
        
        stat_url = f"https://statsapi.mlb.com/api/v1/people/{pid}/stats?stats=season&group=pitching"
        s_res = requests.get(stat_url, timeout=4).json()
        splits = s_res.get("stats", [{}])[0].get("splits", [])
        if not splits:
            return None
        s = splits[0].get("stat", {})
        
        bf = int(s.get("battersFaced", 0))
        so = int(s.get("strikeOuts", 0))
        pitches = int(s.get("numberOfPitches", 0))
        games_started = int(s.get("gamesStarted", 0)) or int(s.get("gamesPitched", 1))

        if bf > 30 and so > 5:
            k_pct = so / bf
            p_per_bf = round(pitches / bf, 2) if bf > 0 else 3.90
            p_per_start = round(pitches / games_started, 1) if games_started > 0 else 88.0
            return {
                "k_rate": float(np.clip(k_pct, 0.12, 0.40)),
                "pitches_per_bf": float(np.clip(p_per_bf, 3.2, 4.6)),
                "expected_pitches": float(np.clip(p_per_start, 65.0, 105.0))
            }
    except Exception:
        pass
    return None

def fetch_team_k_rate(team_name):
    """Derive team strikeout rate to model lineup vulnerability."""
    team_k_map = {
        "Rockies": 0.262, "Athletics": 0.254, "White Sox": 0.252, "Mariners": 0.250,
        "Pirates": 0.245, "Angels": 0.238, "Nationals": 0.228, "Marlins": 0.230,
        "Reds": 0.235, "Cardinals": 0.220, "Tigers": 0.228, "Giants": 0.226,
        "Red Sox": 0.225, "Rays": 0.232, "Cubs": 0.222, "Brewers": 0.228,
        "Guardians": 0.198, "Padres": 0.202, "Astros": 0.195, "Diamondbacks": 0.208,
        "Yankees": 0.218, "Dodgers": 0.212, "Phillies": 0.214, "Orioles": 0.216,
        "Braves": 0.220, "Blue Jays": 0.205, "Twins": 0.222, "Royals": 0.200,
        "Mets": 0.218, "Rangers": 0.224
    }
    for t_key, rate in team_k_map.items():
        if t_key.lower() in team_name.lower():
            return rate
    return LEAGUE_K_RATE

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

            # 1. Ingest Kalman Prior
            c.execute("INSERT OR IGNORE INTO Pitcher_Kalman_State (pitcher_name, latent_k_modifier, variance_p, process_noise_q) VALUES (?, 1.000, 0.040, 0.0025)", (sp_name,))
            kalman_row = c.execute("SELECT latent_k_modifier, variance_p, process_noise_q FROM Pitcher_Kalman_State WHERE pitcher_name = ?", (sp_name,)).fetchone()
            latent_k_mod = float(kalman_row["latent_k_modifier"])

            # 2. Match Pitcher Kinematics from Pitcher_Stats
            sp_row = c.execute(f"""
                SELECT whiff_rate, pitches_per_bf 
                FROM Pitcher_Stats 
                WHERE {p_name_col} = ? OR ? LIKE '%' || {p_name_col}
                LIMIT 1
            """, (sp_name, sp_name)).fetchone()

            # Dynamic retrieval if local row is unpopulated
            live_stats = None
            if not sp_row or not sp_row["whiff_rate"] or float(sp_row["whiff_rate"]) == 0.245:
                live_stats = fetch_pitcher_live_k_stats(sp_name)

            if live_stats:
                sp_base_k = live_stats["k_rate"] * latent_k_mod
                p_bf = live_stats["pitches_per_bf"]
                expected_pitches = live_stats["expected_pitches"]
            else:
                p_bf = float(sp_row["pitches_per_bf"]) if sp_row and sp_row["pitches_per_bf"] else 3.90
                p_whiff = float(sp_row["whiff_rate"]) if sp_row and sp_row["whiff_rate"] else 0.245
                sp_base_k = np.clip(p_whiff * 0.92, 0.12, 0.38) * latent_k_mod
                expected_pitches = 88.0

            # 3. Opponent Lineup Strikeout Rate
            opp_k_rate = fetch_team_k_rate(opp_team)

            # 4. Synthesize Matchup K% via Log5
            matchup_k_rate = calculate_log5_k(sp_base_k, opp_k_rate) * ump_k_mod

            # 5. Batters Faced & Kinematic Fatigue past pitch 65
            expected_bf = expected_pitches / p_bf
            fatigue_penalty = -0.0015 * max(0.0, expected_pitches - 65.0)
            adjusted_k_rate = max(0.05, matchup_k_rate + fatigue_penalty)

            # 6. Poisson Expectations
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
    print(f"[SUCCESS] Synthesized {total_pitchers} Pitcher Strikeout Projections with dynamic variations.")

if __name__ == "__main__":
    run_pitcher_props()
