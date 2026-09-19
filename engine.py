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

CATCHER_FRAMING_RUNS = {
    "San Francisco Giants": -0.18, "Texas Rangers": -0.15, "Milwaukee Brewers": -0.16,
    "New York Yankees": -0.14, "Los Angeles Dodgers": -0.12, "Toronto Blue Jays": -0.11,
    "Seattle Mariners": -0.10, "Philadelphia Phillies": -0.08, "Arizona Diamondbacks": -0.06,
    "Houston Astros": -0.05, "Baltimore Orioles": -0.04, "Tampa Bay Rays": -0.04,
    "Cleveland Guardians": -0.03, "Atlanta Braves": -0.02, "Chicago Cubs": -0.01,
    "San Diego Padres": 0.01, "Detroit Tigers": 0.02, "Boston Red Sox": 0.03,
    "Minnesota Twins": 0.04, "Cincinnati Reds": 0.05, "Kansas City Royals": 0.06,
    "New York Mets": 0.07, "St. Louis Cardinals": 0.08, "Pittsburgh Pirates": 0.09,
    "Los Angeles Angels": 0.10, "Washington Nationals": 0.12, "Miami Marlins": 0.14,
    "Athletics": 0.15, "Oakland Athletics": 0.15, "Chicago White Sox": 0.18,
    "Colorado Rockies": 0.20
}

TTOP_SUPPRESSION_FACTOR = 0.90
TOP_ORDER_WEIGHT = 1.03
F5_VOLUME_SCALAR = 5.0 / 9.7
PARK_REGRESSION_FACTOR = 0.90

LEAGUE_AVG_BA = 0.245
LEAGUE_AVG_K_RATE = 0.222
LEAGUE_AVG_BB_RATE = 0.082

def compute_24_state_markov_half_inning_runs(p_single, p_double, p_triple, p_hr, p_bb, p_k, p_out):
    """
    Computes exact analytical expected runs per half-inning via 24-state base-out Markov chain:
    E[Runs] = (I - Q)^(-1) * R_vec
    States: 8 base occupancies (0 to 7) x 3 out states (0, 1, 2) = 24 transient states.
    """
    Q = np.zeros((24, 24))
    R_vec = np.zeros(24)

    outcomes = {
        'single': {0: (1, 0), 1: (3, 0), 2: (1, 1), 3: (3, 1), 4: (1, 1), 5: (3, 1), 6: (1, 2), 7: (3, 2)},
        'double': {0: (2, 0), 1: (2, 1), 2: (2, 1), 3: (2, 2), 4: (2, 1), 5: (2, 2), 6: (2, 2), 7: (2, 3)},
        'triple': {0: (4, 0), 1: (4, 1), 2: (4, 1), 3: (4, 2), 4: (4, 1), 5: (4, 2), 6: (4, 2), 7: (4, 3)},
        'hr':     {0: (0, 1), 1: (0, 2), 2: (0, 2), 3: (0, 3), 4: (0, 2), 5: (0, 3), 6: (0, 3), 7: (0, 4)},
        'bb':     {0: (1, 0), 1: (3, 0), 2: (3, 0), 3: (7, 0), 4: (1, 0), 5: (3, 0), 6: (7, 0), 7: (7, 1)}
    }

    for out in range(3):
        for b in range(8):
            curr_state = out * 8 + b
            for ev, p_ev in [('single', p_single), ('double', p_double), ('triple', p_triple), ('hr', p_hr), ('bb', p_bb)]:
                next_b, runs = outcomes[ev][b]
                next_state = out * 8 + next_b
                Q[curr_state, next_state] += p_ev
                R_vec[curr_state] += p_ev * runs

            if out < 2:
                next_state_k = (out + 1) * 8 + b
                Q[curr_state, next_state_k] += p_k

            has_runner_1st = (b in (1, 3, 5, 7))
            if has_runner_1st and out == 0:
                p_dp = p_out * 0.12
                p_reg_out = p_out - p_dp
                next_b = 0 if b == 1 else (2 if b == 3 else (4 if b == 5 else 6))
                Q[curr_state, (out + 2) * 8 + next_b] += p_dp
                Q[curr_state, (out + 1) * 8 + b] += p_reg_out
            elif out < 2:
                Q[curr_state, (out + 1) * 8 + b] += p_out

    I = np.eye(24)
    try:
        N = np.linalg.inv(I - Q)
        exp_runs_from_empty = np.dot(N, R_vec)[0]
    except Exception:
        exp_runs_from_empty = 0.50

    return max(0.05, float(exp_runs_from_empty))

def project_starter_innings(effective_metric: float) -> tuple:
    projected_outs = float(np.clip(27.0 - (effective_metric * 2.2), 9.0, 21.6))
    ip_projected = projected_outs / 3.0
    sp_weight = float(np.clip(ip_projected / 9.0, 0.33, 0.80))
    pen_weight = round(1.0 - sp_weight, 4)
    return sp_weight, pen_weight, round(ip_projected, 1)

def run_production_game_simulations(conn, cursor, iterations=50000):
    print("=" * 65)
    print(f"[{datetime.now()}] Launching Production Markov Game Engine (N={iterations})...")
    print("=" * 65)

    query = '''
    SELECT 
        d.game_pk,
        d.home_team,
        d.away_team,
        COALESCE(d.home_pitcher, 'Unknown'),
        COALESCE(d.away_pitcher, 'Unknown'),
        COALESCE(d.air_density, 1.225),
        COALESCE(d.uv_modifier, 5.0),
        COALESCE(u.run_modifier, 1.00),
        COALESCE(ps_h.throws, 'R'),
        COALESCE(ps_a.throws, 'R'),
        COALESCE(ps_h.xfip, ps_h.est_era, 4.20),
        COALESCE(ps_a.xfip, ps_a.est_era, 4.20),
        COALESCE(t_h.ops_vs_rhp, 0.720),
        COALESCE(t_h.ops_vs_lhp, 0.720),
        COALESCE(t_a.ops_vs_rhp, 0.720),
        COALESCE(t_a.ops_vs_lhp, 0.720),
        COALESCE(t_h.bsr_per_game, 4.50),
        COALESCE(t_a.bsr_per_game, 4.50),
        COALESCE(dm_h.offensive_modifier, 1.00),
        COALESCE(dm_h.pitching_modifier, 1.00),
        COALESCE(dm_a.offensive_modifier, 1.00),
        COALESCE(dm_a.pitching_modifier, 1.00),
        COALESCE(pm_h.f5_run_modifier, 1.00),
        COALESCE(pm_a.f5_run_modifier, 1.00),
        COALESCE(bf_h.fatigue_multiplier, 1.00),
        COALESCE(bf_a.fatigue_multiplier, 1.00),
        COALESCE(bf_h.high_leverage_available, 1),
        COALESCE(bf_a.high_leverage_available, 1)
    FROM Daily_Lineups d
    LEFT JOIN Daily_Umpires u ON d.game_pk = u.game_pk
    LEFT JOIN Pitcher_Stats ps_h ON d.home_pitcher LIKE '%' || ps_h.last_name
    LEFT JOIN Pitcher_Stats ps_a ON d.away_pitcher LIKE '%' || ps_a.last_name
    LEFT JOIN Team_Offense t_h ON d.home_team = t_h.team_name
    LEFT JOIN Team_Offense t_a ON d.away_team = t_a.team_name
    LEFT JOIN Dynamic_Modifiers dm_h ON d.home_team = dm_h.team_name
    LEFT JOIN Dynamic_Modifiers dm_a ON d.away_team = dm_a.team_name
    LEFT JOIN Pitcher_Modifiers pm_h ON d.home_pitcher = pm_h.pitcher_name
    LEFT JOIN Pitcher_Modifiers pm_a ON d.away_pitcher = pm_a.pitcher_name
    LEFT JOIN Bullpen_Fatigue bf_h ON d.home_team = bf_h.team_name
    LEFT JOIN Bullpen_Fatigue bf_a ON d.away_team = bf_a.team_name
    WHERE d.status != 'Final';
    '''
    cursor.execute(query)
    slate = cursor.fetchall()

    if not slate:
        print("[INFO] No pending scheduled matchups found in Daily_Lineups.")
        return

    now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    dispersion = 1.35

    for match in slate:
        (pk, home, away, home_p, away_p, rho, uv_raw, ump_run,
         h_throws, a_throws, h_sp_metric, a_sp_metric,
         h_ops_rhp, h_ops_lhp, a_ops_rhp, a_ops_lhp,
         h_bsr, a_bsr, h_off_mod, h_pitch_mod, a_off_mod, a_pitch_mod,
         h_p_run_mod, a_p_run_mod, h_pen_fatigue, a_pen_fatigue,
         h_hl_avail, a_hl_avail) = match

        # Starter workloads
        w_h_sp, w_h_pen, h_ip_proj = project_starter_innings(h_sp_metric * h_p_run_mod)
        w_a_sp, w_a_pen, a_ip_proj = project_starter_innings(a_sp_metric * a_p_run_mod)

        # Environmental & Catcher Shadow-Zone Integration
        base_pf = DEFAULT_PARK_FACTORS.get(home, 1.00)
        air_drag_mult = 1.000 + ((1.225 - rho) * 1.5)
        uv_glare_mult = 1.000 + (np.clip(uv_raw, 1.0, 11.0) - 5.0) * 0.005

        h_catcher_framing = CATCHER_FRAMING_RUNS.get(home, 0.0)
        a_catcher_framing = CATCHER_FRAMING_RUNS.get(away, 0.0)
        effective_ump_h = max(0.85, ump_run + (h_catcher_framing / 9.0))
        effective_ump_a = max(0.85, ump_run + (a_catcher_framing / 9.0))

        full_env_h = base_pf * air_drag_mult * uv_glare_mult * effective_ump_h
        full_env_a = base_pf * air_drag_mult * uv_glare_mult * effective_ump_a

        # High-leverage tier penalty
        h_hl_tax = 0.18 if h_hl_avail == 0 else 0.00
        a_hl_tax = 0.18 if a_hl_avail == 0 else 0.00

        # Baseline Platoon Expectancies
        a_platoon = a_ops_lhp if h_throws == 'L' else a_ops_rhp
        h_platoon = h_ops_lhp if a_throws == 'L' else h_ops_rhp

        a_sp_matchup = a_bsr * (a_platoon / 0.720)
        h_sp_matchup = h_bsr * (h_platoon / 0.720)

        exp_away_runs = max(
            0.2, 
            (((a_sp_matchup * a_off_mod * w_h_sp * (h_sp_metric / 4.20) * h_p_run_mod) + 
              (a_bsr * a_off_mod * w_h_pen * h_pen_fatigue * h_pitch_mod)) * full_env_a) + h_hl_tax
        )
        exp_home_runs = max(
            0.2, 
            (((h_sp_matchup * h_off_mod * w_a_sp * (a_sp_metric / 4.20) * a_p_run_mod) + 
              (h_bsr * h_off_mod * w_a_pen * a_pen_fatigue * a_pitch_mod)) * full_env_h) + a_hl_tax
        )

        # 24-State Base-Out Markov Run Expectancies
        p_bb_a = float(np.clip(LEAGUE_AVG_BB_RATE * (a_platoon / 0.720), 0.05, 0.14))
        p_k_a = float(np.clip(LEAGUE_AVG_K_RATE * (h_sp_metric / 4.20), 0.12, 0.35))
        p_hit_a = float(np.clip(LEAGUE_AVG_BA * (a_platoon / 0.720), 0.18, 0.32))
        p_hr_a = p_hit_a * 0.14
        p_double_a = p_hit_a * 0.20
        p_triple_a = p_hit_a * 0.02
        p_single_a = p_hit_a - (p_hr_a + p_double_a + p_triple_a)
        p_out_a = max(0.20, 1.0 - (p_single_a + p_double_a + p_triple_a + p_hr_a + p_bb_a + p_k_a))
        markov_exp_a = compute_24_state_markov_half_inning_runs(p_single_a, p_double_a, p_triple_a, p_hr_a, p_bb_a, p_k_a, p_out_a) * 9.0

        p_bb_h = float(np.clip(LEAGUE_AVG_BB_RATE * (h_platoon / 0.720), 0.05, 0.14))
        p_k_h = float(np.clip(LEAGUE_AVG_K_RATE * (a_sp_metric / 4.20), 0.12, 0.35))
        p_hit_h = float(np.clip(LEAGUE_AVG_BA * (h_platoon / 0.720), 0.18, 0.32))
        p_hr_h = p_hit_h * 0.14
        p_double_h = p_hit_h * 0.20
        p_triple_h = p_hit_h * 0.02
        p_single_h = p_hit_h - (p_hr_h + p_double_h + p_triple_h)
        p_out_h = max(0.20, 1.0 - (p_single_h + p_double_h + p_triple_h + p_hr_h + p_bb_h + p_k_h))
        markov_exp_h = compute_24_state_markov_half_inning_runs(p_single_h, p_double_h, p_triple_h, p_hr_h, p_bb_h, p_k_h, p_out_h) * 8.65

        final_exp_away = round((0.60 * exp_away_runs) + (0.40 * markov_exp_a), 2)
        final_exp_home = round((0.60 * exp_home_runs) + (0.40 * markov_exp_h), 2)

        # Monte Carlo Iterations
        rng = np.random.default_rng(seed=int(pk))
        va = max(final_exp_away + 0.01, final_exp_away * dispersion)
        vh = max(final_exp_home + 0.01, final_exp_home * dispersion)
        pa = max(0.01, min(0.99, final_exp_away / va))
        ph = max(0.01, min(0.99, final_exp_home / vh))
        na = max(0.1, (final_exp_away ** 2) / (va - final_exp_away))
        nh = max(0.1, (final_exp_home ** 2) / (vh - final_exp_home))

        away_sim = np.clip(rng.negative_binomial(na, pa, iterations), 0, 22)
        home_sim = np.clip(rng.negative_binomial(nh, ph, iterations), 0, 22)

        p_home_reg = float(np.mean(home_sim > away_sim))
        p_tie_reg = float(np.mean(home_sim == away_sim))
        final_home_prob = round(p_home_reg + (0.53 * p_tie_reg), 4)
        final_away_prob = round(1.0 - final_home_prob, 4)
        edge = round(abs(final_home_prob - final_away_prob), 4)

        # F5 Model Synthesis
        f5_pf = 1.000 + (base_pf - 1.000) * PARK_REGRESSION_FACTOR
        f5_env_h = f5_pf * air_drag_mult * uv_glare_mult * effective_ump_h
        f5_env_a = f5_pf * air_drag_mult * uv_glare_mult * effective_ump_a

        a_xera_f5 = np.clip(a_sp_metric, 1.5, 9.0) * a_p_run_mod * TTOP_SUPPRESSION_FACTOR
        h_xera_f5 = np.clip(h_sp_metric, 1.5, 9.0) * h_p_run_mod * TTOP_SUPPRESSION_FACTOR

        a_off_f5 = (a_platoon / 0.720) * a_off_mod * TOP_ORDER_WEIGHT
        h_off_f5 = (h_platoon / 0.720) * h_off_mod * TOP_ORDER_WEIGHT

        lam_f5_a = max(0.10, (h_xera_f5 * a_off_f5 * f5_env_a) * F5_VOLUME_SCALAR)
        lam_f5_h = max(0.10, (a_xera_f5 * h_off_f5 * f5_env_h) * F5_VOLUME_SCALAR)

        f5_disp = 1.22
        f5_va, f5_vh = lam_f5_a * f5_disp, lam_f5_h * f5_disp
        f5_pa, f5_ph = max(0.01, min(0.99, lam_f5_a / f5_va)), max(0.01, min(0.99, lam_f5_h / f5_vh))
        f5_na = max(0.1, (lam_f5_a ** 2) / (f5_va - lam_f5_a))
        f5_nh = max(0.1, (lam_f5_h ** 2) / (f5_vh - lam_f5_h))

        f5_sim_a = np.clip(rng.negative_binomial(f5_na, f5_pa, iterations), 0, 15)
        f5_sim_h = np.clip(rng.negative_binomial(f5_nh, f5_ph, iterations), 0, 15)
        f5_sim_tot = f5_sim_a + f5_sim_h

        f5_median_cont = round(float(np.median(f5_sim_tot)), 2)
        f5_away_prob = round(float(np.mean(f5_sim_a > f5_sim_h)), 4)
        f5_home_prob = round(float(np.mean(f5_sim_h > f5_sim_a)), 4)
        f5_tie_prob = round(float(np.mean(f5_sim_a == f5_sim_h)), 4)

        # Store to Production Tables
        cursor.execute('''
        INSERT OR REPLACE INTO Model_Forecasts 
        (game_pk, home_team, away_team, home_prob, away_prob, predicted_edge, predicted_home_runs, predicted_away_runs, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        ''', (pk, home, away, final_home_prob, final_away_prob, edge, final_exp_home, final_exp_away, now_ts))

        cursor.execute('''
        INSERT OR REPLACE INTO F5_Forecasts 
        (game_pk, away_team, home_team, away_starter, home_starter, f5_away_prob, f5_home_prob, f5_tie_prob, f5_exp_away_runs, f5_exp_home_runs, f5_total_runs, f5_median_total)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        ''', (pk, away, home, away_p, home_p, f5_away_prob, f5_home_prob, f5_tie_prob, round(lam_f5_a, 2), round(lam_f5_h, 2), round(lam_f5_a + lam_f5_h, 2), f5_median_cont))

        print(f"[{away} @ {home}] ML: {home} {final_home_prob:.1%} | Exp Runs: {final_exp_away} - {final_exp_home} | F5 Median: {f5_median_cont}")

    conn.commit()
    print("[SUCCESS] Production forecasts generated and persisted to Model_Forecasts and F5_Forecasts.")

def main():
    conn = sqlite3.connect('mlb_engine.db', timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    cursor = conn.cursor()

    run_production_game_simulations(conn, cursor)

    cursor.execute("PRAGMA wal_checkpoint(TRUNCATE);")
    conn.close()

if __name__ == "__main__":
    main()


def run_ultimate_monte_carlo(*args, **kwargs):
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

    iterations = kwargs.get('iterations', 50000)
    run_production_game_simulations(conn, cursor, iterations=iterations)

    if close_after:
        cursor.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        conn.close()
