import os
import sys
import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

DEFAULT_PARK_FACTORS = {
    "Colorado Rockies": 1.38, "Boston Red Sox": 1.09, "Cincinnati Reds": 1.08,
    "Kansas City Royals": 1.05, "Texas Rangers": 1.04, "Arizona Diamondbacks": 1.04,
    "Philadelphia Phillies": 1.03, "Washington Nationals": 1.02, "Atlanta Braves": 1.01,
    "Baltimore Orioles": 1.01, "Chicago Cubs": 1.01, "Los Angeles Angels": 1.00,
    "Milwaukee Brewers": 1.00, "Minnesota Twins": 1.00, "Toronto Blue Jays": 1.00,
    "Chicago White Sox": 0.99, "Houston Astros": 0.99, "Pittsburgh Pirates": 0.98,
    "St. Louis Cardinals": 0.98, "Detroit Tigers": 0.97, "New York Yankees": 0.97,
    "Cleveland Guardians": 0.96, "Miami Marlins": 0.95, "Oakland Athletics": 0.95,
    "San Francisco Giants": 0.95, "Tampa Bay Rays": 0.94, "New York Mets": 0.94,
    "Los Angeles Dodgers": 0.93, "San Diego Padres": 0.92, "Seattle Mariners": 0.91
}

STADIUM_RHO_BASELINES = {
    "Colorado Rockies": 1.050, "Arizona Diamondbacks": 1.075, "Texas Rangers": 1.135,
    "Atlanta Braves": 1.145, "Minnesota Twins": 1.150, "Cincinnati Reds": 1.160,
    "Detroit Tigers": 1.162, "Milwaukee Brewers": 1.163, "Chicago Cubs": 1.166,
    "Chicago White Sox": 1.167, "St. Louis Cardinals": 1.168, "Washington Nationals": 1.172,
    "Tampa Bay Rays": 1.175, "Miami Marlins": 1.185, "New York Yankees": 1.188,
    "Boston Red Sox": 1.195, "Baltimore Orioles": 1.198, "San Francisco Giants": 1.205,
    "Los Angeles Dodgers": 1.210, "Los Angeles Angels": 1.212, "New York Mets": 1.215,
    "Philadelphia Phillies": 1.218, "San Diego Padres": 1.225, "Seattle Mariners": 1.225,
    "Oakland Athletics": 1.220, "Athletics": 1.220, "Houston Astros": 1.180,
    "Kansas City Royals": 1.155, "Pittsburgh Pirates": 1.170, "Cleveland Guardians": 1.165,
    "Toronto Blue Jays": 1.190, "Default": 1.225
}

LEAGUE_AVG_BA = 0.245
LEAGUE_AVG_K_RATE = 0.222
LEAGUE_AVG_BB_RATE = 0.082

ORDER_PA_WEIGHTS = {
    1: 1.14, 2: 1.11, 3: 1.08, 4: 1.05, 5: 1.02,
    6: 0.98, 7: 0.95, 8: 0.92, 9: 0.88
}

def log5_matchup_odds(p_batter: float, p_pitcher: float, p_league: float) -> float:
    p_b = float(np.clip(p_batter, 0.05, 0.95))
    p_p = float(np.clip(p_pitcher, 0.05, 0.95))
    p_l = float(np.clip(p_league, 0.05, 0.95))
    odds_b = p_b / (1.0 - p_b)
    odds_p = p_p / (1.0 - p_p)
    odds_l = p_l / (1.0 - p_l)
    odds_matchup = (odds_b * odds_p) / odds_l
    return float(np.clip(odds_matchup / (1.0 + odds_matchup), 0.01, 0.99))

def apply_bayesian_hit_shrinkage(raw_prob_over_0_5: float, ab_sample: int = 40) -> float:
    p_clipped = float(np.clip(raw_prob_over_0_5, 0.05, 0.95))
    logit_raw = np.log(p_clipped / (1.0 - p_clipped))
    prior_p = 0.605
    logit_prior = np.log(prior_p / (1.0 - prior_p))
    w = float(np.clip(ab_sample / (ab_sample + 80), 0.20, 0.85))
    shrunk_logit = (w * logit_raw) + ((1.0 - w) * logit_prior)
    shrunk_p = 1.0 / (1.0 + np.exp(-shrunk_logit))
    return float(np.clip(shrunk_p, 0.20, 0.82))

def project_endogenous_plate_appearances(batting_order: int, team_expected_runs: float, is_home: bool, win_prob: float) -> tuple:
    team_pa = 25.5 + (1.25 * team_expected_runs)
    if is_home and win_prob > 0.50:
        team_pa -= 3.0 * win_prob
    base_slot_pa = (team_pa / 9.0) * ORDER_PA_WEIGHTS.get(batting_order, 1.00)
    proj_pa = float(np.clip(base_slot_pa, 3.0, 5.8))
    proj_ab = proj_pa * 0.895
    return round(proj_pa, 2), round(proj_ab, 2)

def project_starter_innings(effective_metric: float) -> tuple:
    projected_outs = float(np.clip(27.0 - (effective_metric * 2.2), 9.0, 21.6))
    ip_projected = projected_outs / 3.0
    sp_weight = float(np.clip(ip_projected / 9.0, 0.33, 0.80))
    pen_weight = round(1.0 - sp_weight, 4)
    return sp_weight, pen_weight

def run_production_batter_props(conn, cursor):
    print("=" * 65)
    print(f"[{datetime.now()}] Synthesizing Production Batter Props (Pitch-Arsenal Log5)...")
    print("=" * 65)

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
    CREATE TABLE IF NOT EXISTS Daily_Batters (
        game_pk INTEGER,
        player_name TEXT,
        team_name TEXT,
        batting_order INTEGER,
        is_starter INTEGER DEFAULT 1,
        PRIMARY KEY (game_pk, player_name)
    );
    ''')

    # Query daily slates with game model context
    query_games = '''
    SELECT 
        d.game_pk,
        d.home_team,
        d.away_team,
        COALESCE(d.home_pitcher, 'Unknown'),
        COALESCE(d.away_pitcher, 'Unknown'),
        COALESCE(d.air_density, 1.225),
        COALESCE(m.predicted_home_runs, 4.50),
        COALESCE(m.predicted_away_runs, 4.50),
        COALESCE(m.home_prob, 0.50),
        COALESCE(m.away_prob, 0.50),
        COALESCE(ps_h.throws, 'R'),
        COALESCE(ps_a.throws, 'R'),
        COALESCE(ps_h.xfip, ps_h.est_era, 4.20),
        COALESCE(ps_a.xfip, ps_a.est_era, 4.20),
        COALESCE(ps_h.arsenal_type, 'Balanced'),
        COALESCE(ps_a.arsenal_type, 'Balanced')
    FROM Daily_Lineups d
    LEFT JOIN Model_Forecasts m ON d.game_pk = m.game_pk
    LEFT JOIN Pitcher_Stats ps_h ON d.home_pitcher LIKE '%' || ps_h.last_name
    LEFT JOIN Pitcher_Stats ps_a ON d.away_pitcher LIKE '%' || ps_a.last_name
    WHERE d.status != 'Final';
    '''
    cursor.execute(query_games)
    games = cursor.fetchall()

    if not games:
        print("[INFO] No active slates found for batter prop synthesis.")
        return

    # Cache active modifiers and stats
    cursor.execute("SELECT player_name, contact_modifier, appearance_count FROM Batter_Modifiers;")
    batter_mods = {r[0]: (r[1], r[2]) for r in cursor.fetchall()}

    cursor.execute("SELECT player_name, team_name, avg, avg_vs_rhp, avg_vs_lhp, k_rate, bb_rate FROM Batter_Stats;")
    batter_stats = {(r[0], r[1]): (r[2], r[3], r[4], r[5], r[6]) for r in cursor.fetchall()}

    prop_rows = []

    for game in games:
        (pk, home, away, home_p, away_p, rho,
         pred_home_runs, pred_away_runs, home_prob, away_prob,
         h_throws, a_throws, h_era, a_era, h_arsenal, a_arsenal) = game

        base_pf = DEFAULT_PARK_FACTORS.get(home, 1.00)
        env_hit_scalar = (1.000 + (base_pf - 1.000) * 0.70) * (1.000 + ((1.225 - rho) * 0.8))

        w_h_sp, _ = project_starter_innings(h_era)
        w_a_sp, _ = project_starter_innings(a_era)

        # Retrieve slate batters from Daily_Batters
        cursor.execute("SELECT player_name, team_name, batting_order FROM Daily_Batters WHERE game_pk = ? ORDER BY batting_order ASC;", (pk,))
        lineup_batters = cursor.fetchall()

        # Fallback to seeded top-order slots if lineup not confirmed
        if not lineup_batters:
            lineup_batters = []
            for t_name in (away, home):
                for slot in range(1, 10):
                    lineup_batters.append((f"Batter {slot}", t_name, slot))

        for b_name, b_team, b_order in lineup_batters:
            is_home = (b_team == home)
            tm_runs = pred_home_runs if is_home else pred_away_runs
            win_prob = home_prob if is_home else away_prob
            opp_throws = a_throws if is_home else h_throws
            opp_era = a_era if is_home else h_era
            opp_arsenal = a_arsenal if is_home else h_arsenal
            w_sp = w_a_sp if is_home else w_h_sp

            stats = batter_stats.get((b_name, b_team))
            if not stats:
                stats = batter_stats.get((f"Batter {b_order}", b_team), (0.250, 0.250, 0.250, 0.220, 0.085))

            b_avg, b_rhp, b_lhp, b_k, b_bb = stats
            mod_tuple = batter_mods.get(b_name, (1.000, 0))
            b_mod, app_count = mod_tuple[0], mod_tuple[1]

            base_contact = (b_lhp if opp_throws == 'L' else b_rhp) * b_mod

            # Pitch-Arsenal Matchup Decomposition
            if opp_arsenal == 'FourSeam_Sweeper':
                b_k_adj = b_k * 1.08
                contact_adj = base_contact * 0.96
            elif opp_arsenal == 'Sinker_Cutter':
                b_k_adj = b_k * 0.92
                contact_adj = base_contact * 1.03
            else:
                b_k_adj = b_k
                contact_adj = base_contact

            proj_pa, proj_ab = project_endogenous_plate_appearances(b_order, tm_runs, is_home, win_prob)

            # Matchup Odds Synthesis (Log5)
            matchup_k = log5_matchup_odds(b_k_adj, 0.220, LEAGUE_AVG_K_RATE)
            p_in_play = max(0.40, 1.0 - matchup_k - b_bb)
            matchup_ba_sp = log5_matchup_odds(contact_adj, float(np.clip(opp_era / 17.5, 0.18, 0.32)), LEAGUE_AVG_BA) * env_hit_scalar

            p_hit_pa_sp = p_in_play * (matchup_ba_sp / max(0.01, 1.0 - LEAGUE_AVG_K_RATE - LEAGUE_AVG_BB_RATE))
            p_hit_pa_pen = (1.0 - 0.220 - 0.085) * (0.250 * env_hit_scalar / max(0.01, 1.0 - LEAGUE_AVG_K_RATE - LEAGUE_AVG_BB_RATE))

            p_hit_pa = float(np.clip(w_sp * p_hit_pa_sp + (1.0 - w_sp) * p_hit_pa_pen, 0.10, 0.45))
            p_hit_ab = float(np.clip(p_hit_pa / 0.895, 0.12, 0.48))

            expected_hits = round(proj_ab * p_hit_ab, 2)

            # Binomial Simulation & Dynamic Bayesian Shrinkage
            rng = np.random.default_rng(seed=int(pk) + int(b_order) * 7)
            sim_hits = rng.binomial(int(np.round(proj_ab)), p_hit_ab, 5000)

            raw_over_0_5 = float(np.mean(sim_hits >= 1))
            raw_over_1_5 = float(np.mean(sim_hits >= 2))
            raw_over_2_5 = float(np.mean(sim_hits >= 3))

            actual_ab_sample = max(15, app_count * 4)
            shrunk_over_0_5 = round(apply_bayesian_hit_shrinkage(raw_over_0_5, ab_sample=actual_ab_sample), 4)
            shrunk_over_1_5 = round(raw_over_1_5 * (shrunk_over_0_5 / max(0.01, raw_over_0_5)), 4)
            shrunk_over_2_5 = round(raw_over_2_5 * (shrunk_over_0_5 / max(0.01, raw_over_0_5)), 4)

            prop_rows.append((
                pk, b_name, b_team, b_order,
                proj_pa, proj_ab, expected_hits,
                shrunk_over_0_5, shrunk_over_1_5, shrunk_over_2_5
            ))

    cursor.executemany('''
    INSERT OR REPLACE INTO Batter_Hit_Forecasts 
    (game_pk, player_name, team_name, batting_order, projected_pa, projected_ab, expected_hits, over_0_5_hit_prob, over_1_5_hit_prob, over_2_5_hit_prob)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
    ''', prop_rows)

    conn.commit()
    print(f"[SUCCESS] Synthesized hit props for {len(prop_rows)} batters into Batter_Hit_Forecasts.")

def main():
    conn = sqlite3.connect('mlb_engine.db', timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    cursor = conn.cursor()

    run_production_batter_props(conn, cursor)

    cursor.execute("PRAGMA wal_checkpoint(TRUNCATE);")
    conn.close()

if __name__ == "__main__":
    main()


def run_batter_props_engine(*args, **kwargs):
    conn, cursor = None, None
    for arg in args:
        if isinstance(arg, sqlite3.Connection):
            conn = arg
        elif isinstance(arg, sqlite3.Cursor):
            cursor = arg
    close_after = False
    if conn is None:
        if cursor is not None:
            conn = cursor.connection
        else:
            conn = sqlite3.connect('mlb_engine.db', timeout=30)
            conn.execute("PRAGMA journal_mode=WAL;")
            cursor = conn.cursor()
            close_after = True
    elif cursor is None:
        cursor = conn.cursor()

    run_production_batter_props(conn, cursor)

    if close_after:
        cursor.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        conn.close()
