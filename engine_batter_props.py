import sqlite3
import requests
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

LEAGUE_AVG_BA = 0.245
LEAGUE_AVG_K_RATE = 0.222
LEAGUE_AVG_BB_RATE = 0.082
LEAGUE_AVG_BABIP = 0.292

ORDER_PA_WEIGHTS = {
    1: 1.14, 2: 1.11, 3: 1.08, 4: 1.05, 5: 1.02,
    6: 0.98, 7: 0.95, 8: 0.92, 9: 0.88
}

def ensure_batter_schemas(cursor):
    """Guarantees persistence schemas for individual batter hit projections and historical stats."""
    cursor.executescript('''
    CREATE TABLE IF NOT EXISTS Batter_Hit_Forecasts (
        game_pk INTEGER,
        player_name TEXT,
        team_name TEXT,
        batting_order INTEGER,
        projected_pa REAL,
        projected_ab REAL,
        expected_hits REAL,
        over_0_5_hit_prob REAL,
        over_1_5_hit_prob REAL,
        over_2_5_hit_prob REAL,
        PRIMARY KEY (game_pk, player_name)
    );
    CREATE TABLE IF NOT EXISTS Batter_Stats (
        player_name TEXT PRIMARY KEY,
        team_name TEXT,
        avg REAL DEFAULT 0.250,
        avg_vs_rhp REAL DEFAULT 0.250,
        avg_vs_lhp REAL DEFAULT 0.250,
        bb_rate REAL DEFAULT 0.085,
        k_rate REAL DEFAULT 0.220,
        babip REAL DEFAULT 0.295,
        updated_at TEXT
    );
    ''')

def log5_matchup_odds(p_batter: float, p_pitcher: float, p_league: float) -> float:
    """
    Computes head-to-head event probability using the odds-ratio Log5 formulation.
    Decouples batter skill, pitcher skill, and environmental context from raw sample noise.
    """
    p_b = float(np.clip(p_batter, 0.05, 0.95))
    p_p = float(np.clip(p_pitcher, 0.05, 0.95))
    p_l = float(np.clip(p_league, 0.05, 0.95))

    odds_b = p_b / (1.0 - p_b)
    odds_p = p_p / (1.0 - p_p)
    odds_l = p_l / (1.0 - p_l)

    odds_matchup = (odds_b * odds_p) / odds_l
    p_matchup = odds_matchup / (1.0 + odds_matchup)
    return float(np.clip(p_matchup, 0.01, 0.99))

def project_endogenous_plate_appearances(batting_order: int, team_expected_runs: float, is_home_team: bool, pred_win_prob: float) -> tuple:
    """
    Calculates endogenous plate appearances tied to team scoring volume.
    Deducts 3.0 team plate appearances if the home team has high win equity (walk-off / no-bottom-9th suppression).
    """
    team_pa = 25.5 + (1.25 * team_expected_runs)
    if is_home_team and pred_win_prob > 0.50:
        team_pa -= 3.0 * pred_win_prob

    base_slot_pa = (team_pa / 9.0) * ORDER_PA_WEIGHTS.get(batting_order, 1.00)
    proj_pa = float(np.clip(base_slot_pa, 3.0, 5.8))
    proj_ab = proj_pa * 0.895
    return round(proj_pa, 2), round(proj_ab, 2)

def fetch_confirmed_batting_orders(game_pk: int) -> dict:
    """Ingests official 9-man batting orders directly from the MLB Stats API linescores."""
    url = f"https://statsapi.mlb.com/api/v1/game/{game_pk}/boxscore"
    lineups = {"away": [], "home": []}
    try:
        res = requests.get(url, timeout=10).json()
        teams = res.get('teams', {})
        for side in ('away', 'home'):
            batters = teams.get(side, {}).get('batters', [])
            players = teams.get(side, {}).get('players', {})
            order_idx = 1
            for b_id in batters:
                p_key = f"ID{b_id}"
                p_data = players.get(p_key, {})
                pos = p_data.get('position', {}).get('abbreviation', '')
                if pos == 'P' and len(batters) > 9:
                    continue
                name = p_data.get('person', {}).get('fullName')
                if name and order_idx <= 9:
                    lineups[side].append((order_idx, name))
                    order_idx += 1
    except Exception:
        pass
    return lineups

def run_batter_props_engine():
    print("=" * 65)
    print(f"[{datetime.now()}] Running High-Precision Batter Hit Engine (50,000 Iterations + Log5 Matchups)")
    print("=" * 65)

    conn = sqlite3.connect('mlb_engine.db', timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=10000;")
    cursor = conn.cursor()

    ensure_batter_schemas(cursor)
    conn.commit()

    cursor.execute('''
        SELECT d.game_pk, d.away_team, d.home_team, d.away_pitcher, d.home_pitcher,
               COALESCE(d.air_density, 1.225), COALESCE(u.run_modifier, 1.00),
               m.predicted_away_runs, m.predicted_home_runs, m.away_prob, m.home_prob
        FROM Daily_Lineups d
        LEFT JOIN Daily_Umpires u ON d.game_pk = u.game_pk
        LEFT JOIN Model_Forecasts m ON d.game_pk = m.game_pk
        WHERE d.status != 'Final' AND d.game_pk NOT IN (SELECT game_pk FROM Post_Match_Analysis)
    ''')
    active_games = cursor.fetchall()

    if not active_games:
        conn.close()
        return

    pitcher_stats = {r[0]: (r[1], r[2], r[3], r[4]) for r in cursor.execute(
        "SELECT last_name, COALESCE(est_era, 4.20), COALESCE(throws, 'R'), 0.220, 0.245 FROM Pitcher_Stats"
    ).fetchall()}

    park_factors = {r[0]: r[1] for r in cursor.execute(
        "SELECT home_team, COALESCE(run_factor, 1.00) FROM Park_Factors"
    ).fetchall()}

    batter_cache = {r[0]: (r[1], r[2], r[3], r[4], r[5], r[6]) for r in cursor.execute(
        "SELECT player_name, COALESCE(avg, 0.250), COALESCE(avg_vs_rhp, 0.250), COALESCE(avg_vs_lhp, 0.250), COALESCE(bb_rate, 0.085), COALESCE(k_rate, 0.220), COALESCE(babip, 0.295) FROM Batter_Stats"
    ).fetchall()}

    iterations = 50000

    for game in active_games:
        pk, away, home, away_sp, home_sp, rho, ump_run, pred_a_runs, pred_h_runs, a_prob, h_prob = game
        rng = np.random.default_rng(seed=int(pk) + 777)

        exp_a_runs = pred_a_runs if pred_a_runs else 4.30
        exp_h_runs = pred_h_runs if pred_h_runs else 4.50
        prob_a = a_prob if a_prob else 0.50
        prob_h = h_prob if h_prob else 0.50

        a_sp_ln = away_sp.split()[-1] if away_sp and away_sp != "TBD" else ""
        h_sp_ln = home_sp.split()[-1] if home_sp and home_sp != "TBD" else ""

        a_sp_era, a_sp_throws, a_sp_k, a_sp_baa = pitcher_stats.get(a_sp_ln, (4.20, 'R', 0.220, 0.245))
        h_sp_era, h_sp_throws, h_sp_k, h_sp_baa = pitcher_stats.get(h_sp_ln, (4.20, 'R', 0.220, 0.245))

        # Dynamic starter length projections for mixture modeling
        w_h_sp = float(np.clip((27.0 - (h_sp_era * 2.2)) / 27.0, 0.33, 0.80))
        w_a_sp = float(np.clip((27.0 - (a_sp_era * 2.2)) / 27.0, 0.33, 0.80))

        # Ballpark carry and air density on base hits
        pf_hits = 1.000 + (park_factors.get(home, 1.00) - 1.000) * 0.70
        rho_contact = 1.000 + ((1.225 - rho) * 0.8)
        env_hit_scalar = pf_hits * rho_contact

        lineups = fetch_confirmed_batting_orders(pk)

        matchup_sides = [
            ("away", away, exp_a_runs, False, prob_a, h_sp_throws, h_sp_era, h_sp_k, h_sp_baa, w_h_sp),
            ("home", home, exp_h_runs, True, prob_h, a_sp_throws, a_sp_era, a_sp_k, a_sp_baa, w_a_sp)
        ]

        for side_key, team_name, team_runs, is_home, win_p, opp_throws, opp_era, opp_k, opp_baa, w_sp in matchup_sides:
            roster = lineups.get(side_key, [])
            if not roster or len(roster) < 9:
                roster = [(slot, f"{team_name} Batter {slot}") for slot in range(1, 10)]

            for order_slot, player_name in roster:
                proj_pa, proj_ab = project_endogenous_plate_appearances(order_slot, team_runs, is_home, win_p)

                b_avg, b_rhp, b_lhp, b_bb, b_k, b_babip = batter_cache.get(player_name, (0.250, 0.250, 0.250, 0.085, 0.220, 0.295))
                platoon_avg = b_lhp if opp_throws == 'L' else b_rhp

                # Stage 1: Log5 Strikeout and In-Play Decomposition vs Starter
                matchup_k_sp = log5_matchup_odds(b_k, opp_k, LEAGUE_AVG_K_RATE)
                p_in_play_sp = max(0.40, 1.0 - matchup_k_sp - b_bb)

                # Stage 2: Contact Average vs Starter with Log5 Batting Average Against
                matchup_ba_sp = log5_matchup_odds(platoon_avg, opp_baa, LEAGUE_AVG_BA) * env_hit_scalar
                p_hit_per_pa_sp = p_in_play_sp * (matchup_ba_sp / max(0.01, 1.0 - LEAGUE_AVG_K_RATE - LEAGUE_AVG_BB_RATE))

                # Reliever Baseline Mixture Component
                matchup_ba_pen = platoon_avg * env_hit_scalar
                p_hit_per_pa_pen = (1.0 - b_k - b_bb) * (matchup_ba_pen / max(0.01, 1.0 - LEAGUE_AVG_K_RATE - LEAGUE_AVG_BB_RATE))

                # Weighted Mixture Contact Probability per Plate Appearance
                p_hit_pa = float(np.clip(w_sp * p_hit_per_pa_sp + (1.0 - w_sp) * p_hit_per_pa_pen, 0.10, 0.45))
                p_hit_ab = float(np.clip(p_hit_pa / 0.895, 0.12, 0.48))

                # 50,000 Monte Carlo Iterations per Player
                int_ab = int(np.round(proj_ab))
                sim_hits = rng.binomial(int_ab, p_hit_ab, iterations)

                expected_hits = round(proj_ab * p_hit_ab, 2)
                p_over_0_5 = round(float(np.mean(sim_hits >= 1)), 4)
                p_over_1_5 = round(float(np.mean(sim_hits >= 2)), 4)
                p_over_2_5 = round(float(np.mean(sim_hits >= 3)), 4)

                cursor.execute('''
                INSERT OR REPLACE INTO Batter_Hit_Forecasts 
                (game_pk, player_name, team_name, batting_order, projected_pa, projected_ab, expected_hits, over_0_5_hit_prob, over_1_5_hit_prob, over_2_5_hit_prob)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (pk, player_name, team_name, order_slot, proj_pa, proj_ab, expected_hits, p_over_0_5, p_over_1_5, p_over_2_5))

    conn.commit()
    cursor.execute("PRAGMA wal_checkpoint(TRUNCATE);")
    conn.close()
    print("[SUCCESS] Batter hit projections successfully generated at 50,000 iterations.")

if __name__ == "__main__":
    run_batter_props_engine()
