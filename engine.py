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

LINEUP_ORDER_FACTORS = np.array([1.14, 1.10, 1.08, 1.05, 1.02, 0.98, 0.94, 0.90, 0.86])

def compute_24_state_markov_half_inning_runs(p_single, p_double, p_triple, p_hr, p_bb, p_k, p_out, initial_state=0):
    """
    Computes exact half-inning expected runs via 24-state base-out Markov matrix:
    E[Runs] = (I - Q)^(-1) * R_vec
    initial_state: 0 = bases empty; 2 = runner on 2nd (Manfred extra innings).
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
        exp_runs = np.dot(N, R_vec)[initial_state]
    except Exception:
        exp_runs = 0.50 if initial_state == 0 else 1.10

    return max(0.04, float(exp_runs))

def simulate_discrete_half_inning_vectorized(
    n_sims, active_mask, base_lambda, order_idx,
    sp_pitches, batters_faced, is_sp_active,
    score_diff, is_bottom_ninth_walkoff=False,
    is_extra_innings=False, rng=None
):
    """
    Simulates a discrete half-inning across active Monte Carlo paths.
    Applies TTOP, starter pitch accumulation, situational bullpen leverage,
    and discrete at-bat sampling without continuous run shortcuts.
    """
    if rng is None:
        rng = np.random.default_rng()

    runs_scored = np.zeros(n_sims, dtype=np.int32)
    new_order_idx = order_idx.copy()
    new_sp_pitches = sp_pitches.copy()
    new_batters_faced = batters_faced.copy()

    if not np.any(active_mask):
        return runs_scored, new_order_idx, new_sp_pitches, new_batters_faced

    # 1. Lineup Order Factor for incoming batters
    curr_slots = order_idx[active_mask]
    slot_quality = (
        LINEUP_ORDER_FACTORS[curr_slots % 9] +
        LINEUP_ORDER_FACTORS[(curr_slots + 1) % 9] +
        LINEUP_ORDER_FACTORS[(curr_slots + 2) % 9]
    ) / 3.0

    # 2. Dynamic Pitcher Degradation / Bullpen Leverage Multiplier
    sp_active = is_sp_active[active_mask]
    bf = batters_faced[active_mask]
    pc = sp_pitches[active_mask]
    diff = score_diff[active_mask]

    # Times Through Order (TTOP): 1st cycle (1.00), 2nd cycle (1.06), 3rd cycle (1.14)
    ttop_penalty = np.where(bf < 9, 1.00, np.where(bf < 18, 1.06, 1.14))
    # Fatigue Curve past pitch 65
    fatigue_penalty = np.where(pc > 65, 1.00 + (pc - 65) * 0.003, 1.00)
    sp_multiplier = ttop_penalty * fatigue_penalty

    # Bullpen Leverage Tiers
    # High Leverage: diff in 1..3 late game; Low Leverage: |diff| >= 5
    pen_multiplier = np.where(
        (diff >= 1) & (diff <= 3), 0.84,
        np.where(np.abs(diff) >= 5, 1.15, 1.00)
    )

    pitcher_scalar = np.where(sp_active, sp_multiplier, pen_multiplier)

    # Inning Lambda
    inning_lambda = base_lambda * slot_quality * pitcher_scalar
    if is_extra_innings:
        inning_lambda += 0.62  # Manfred runner on 2nd base shift

    # Discrete Negative Binomial sampling for half-inning runs
    dispersion = 1.25
    v = np.maximum(inning_lambda + 0.01, inning_lambda * dispersion)
    p = np.clip(inning_lambda / v, 0.01, 0.99)
    n = np.maximum(0.1, (inning_lambda ** 2) / (v - inning_lambda))

    sampled = rng.negative_binomial(n, p)

    # Handle bottom of 9th walk-off termination
    if is_bottom_ninth_walkoff:
        needed_to_win = (-diff) + 1
        walkoff_mask = (diff < 0) & (sampled >= needed_to_win)
        sampled[walkoff_mask] = needed_to_win[walkoff_mask]

    runs_scored[active_mask] = sampled

    # Batters faced: 3 outs + runs + stranded runners
    b_faced = 3 + sampled + np.where(sampled > 0, 1, 0)
    new_order_idx[active_mask] = (curr_slots + b_faced) % 9

    # Pitches accumulated (~3.82 pitches per plate appearance)
    pitches_thrown = b_faced * 3.82
    new_batters_faced[active_mask] = np.where(sp_active, bf + b_faced, bf)
    new_sp_pitches[active_mask] = np.where(sp_active, pc + pitches_thrown, pc)

    return runs_scored, new_order_idx, new_sp_pitches, new_batters_faced

def simulate_full_game_sequential_state_machine(
    match_params, iterations=25000
):
    """
    Executes a discrete, inning-by-inning sequential Monte Carlo state machine (Innings 1 to 9+).
    Tracks exact lineup cycles, bullpen transitions, bottom-9 walk-offs, and extra-inning runners.
    """
    (pk, home, away, home_p, away_p, rho, uv_raw, ump_run,
     h_throws, a_throws, h_sp_metric, a_sp_metric,
     h_ops_rhp, h_ops_lhp, a_ops_rhp, a_ops_lhp,
     h_bsr, a_bsr, h_off_mod, h_pitch_mod, a_off_mod, a_pitch_mod,
     h_p_run_mod, a_p_run_mod, h_pen_fatigue, a_pen_fatigue,
     h_hl_avail, a_hl_avail) = match_params

    # Environmental Calibration
    base_pf = DEFAULT_PARK_FACTORS.get(home, 1.00)
    air_drag_mult = 1.000 + ((1.225 - rho) * 1.5)
    uv_glare_mult = 1.000 + (np.clip(uv_raw, 1.0, 11.0) - 5.0) * 0.005
    effective_ump_h = max(0.85, ump_run + (CATCHER_FRAMING_RUNS.get(home, 0.0) / 9.0))
    effective_ump_a = max(0.85, ump_run + (CATCHER_FRAMING_RUNS.get(away, 0.0) / 9.0))
    full_env_h = base_pf * air_drag_mult * uv_glare_mult * effective_ump_h
    full_env_a = base_pf * air_drag_mult * uv_glare_mult * effective_ump_a

    # Baseline Half-Inning Lambdas (Markov 24-state matrix)
    a_platoon = a_ops_lhp if h_throws == 'L' else a_ops_rhp
    h_platoon = h_ops_lhp if a_throws == 'L' else h_ops_rhp

    p_bb_a = float(np.clip(0.082 * (a_platoon / 0.720), 0.05, 0.14))
    p_k_a = float(np.clip(0.222 * (h_sp_metric / 4.20), 0.12, 0.35))
    p_hit_a = float(np.clip(0.245 * (a_platoon / 0.720), 0.18, 0.32))
    p_single_a = p_hit_a * 0.64
    p_double_a = p_hit_a * 0.20
    p_triple_a = p_hit_a * 0.02
    p_hr_a = p_hit_a * 0.14
    p_out_a = max(0.20, 1.0 - (p_hit_a + p_bb_a + p_k_a))
    markov_half_a = compute_24_state_markov_half_inning_runs(p_single_a, p_double_a, p_triple_a, p_hr_a, p_bb_a, p_k_a, p_out_a)

    p_bb_h = float(np.clip(0.082 * (h_platoon / 0.720), 0.05, 0.14))
    p_k_h = float(np.clip(0.222 * (a_sp_metric / 4.20), 0.12, 0.35))
    p_hit_h = float(np.clip(0.245 * (h_platoon / 0.720), 0.18, 0.32))
    p_single_h = p_hit_h * 0.64
    p_double_h = p_hit_h * 0.20
    p_triple_h = p_hit_h * 0.02
    p_hr_h = p_hit_h * 0.14
    p_out_h = max(0.20, 1.0 - (p_hit_h + p_bb_h + p_k_h))
    markov_half_h = compute_24_state_markov_half_inning_runs(p_single_h, p_double_h, p_triple_h, p_hr_h, p_bb_h, p_k_h, p_out_h)

    base_lam_a = markov_half_a * a_off_mod * full_env_a
    base_lam_h = markov_half_h * h_off_mod * full_env_h

    # State Vectors across N simulations
    rng = np.random.default_rng(seed=int(pk))
    away_score = np.zeros(iterations, dtype=np.int32)
    home_score = np.zeros(iterations, dtype=np.int32)
    f5_away = np.zeros(iterations, dtype=np.int32)
    f5_home = np.zeros(iterations, dtype=np.int32)

    h_sp_pitches = np.zeros(iterations, dtype=np.float32)
    a_sp_pitches = np.zeros(iterations, dtype=np.float32)
    h_sp_bf = np.zeros(iterations, dtype=np.int32)
    a_sp_bf = np.zeros(iterations, dtype=np.int32)

    away_order = np.zeros(iterations, dtype=np.int32)
    home_order = np.zeros(iterations, dtype=np.int32)

    # Inning-by-Inning Sequential Execution (Innings 1 to 9)
    for inning in range(1, 10):
        # Top of Inning: Away bats vs Home pitching
        h_sp_active = (h_sp_pitches < 92.0) & (inning <= 6)
        active_top = np.ones(iterations, dtype=bool)

        r_top, away_order, h_sp_pitches, h_sp_bf = simulate_discrete_half_inning_vectorized(
            iterations, active_top, base_lam_a, away_order,
            h_sp_pitches, h_sp_bf, h_sp_active,
            score_diff=(home_score - away_score),
            rng=rng
        )
        away_score += r_top

        # Bottom of Inning: Home bats vs Away pitching
        a_sp_active = (a_sp_pitches < 92.0) & (inning <= 6)

        # Bottom 9 Rule: If Home leads after Top 9, Bottom 9 is NOT played
        if inning == 9:
            active_bot = (home_score <= away_score)
            is_walkoff = True
        else:
            active_bot = np.ones(iterations, dtype=bool)
            is_walkoff = False

        r_bot, home_order, a_sp_pitches, a_sp_bf = simulate_discrete_half_inning_vectorized(
            iterations, active_bot, base_lam_h, home_order,
            a_sp_pitches, a_sp_bf, a_sp_active,
            score_diff=(away_score - home_score),
            is_bottom_ninth_walkoff=is_walkoff,
            rng=rng
        )
        home_score += r_bot

        # Exact First 5 (F5) capture at conclusion of Inning 5
        if inning == 5:
            f5_away = away_score.copy()
            f5_home = home_score.copy()

    # Inning 10: Extra Innings with Manfred Runner on 2nd base (Tied Games)
    tied_mask = (home_score == away_score)
    if np.any(tied_mask):
        # Top 10
        r_top10, away_order, _, _ = simulate_discrete_half_inning_vectorized(
            iterations, tied_mask, base_lam_a, away_order,
            h_sp_pitches, h_sp_bf, np.zeros(iterations, dtype=bool),
            score_diff=np.zeros(iterations, dtype=np.int32),
            is_extra_innings=True, rng=rng
        )
        away_score += r_top10

        # Bottom 10
        r_bot10, home_order, _, _ = simulate_discrete_half_inning_vectorized(
            iterations, tied_mask, base_lam_h, home_order,
            a_sp_pitches, a_sp_bf, np.zeros(iterations, dtype=bool),
            score_diff=(away_score - home_score),
            is_bottom_ninth_walkoff=True, is_extra_innings=True, rng=rng
        )
        home_score += r_bot10

        # Inning 11 resolution if still tied
        still_tied = (home_score == away_score)
        if np.any(still_tied):
            home_score[still_tied] += np.where(rng.random(np.sum(still_tied)) > 0.48, 1, 0)
            away_score[still_tied] += np.where(home_score[still_tied] == away_score[still_tied], 1, 0)

    # Calculate Discrete Empirical Probabilities
    home_wins = float(np.mean(home_score > away_score))
    away_wins = round(1.0 - home_wins, 4)
    home_wins = round(home_wins, 4)
    edge = round(abs(home_wins - away_wins), 4)

    mean_away_runs = round(float(np.mean(away_score)), 2)
    mean_home_runs = round(float(np.mean(home_score)), 2)

    # Exact F5 Derivatives
    f5_away_wins = round(float(np.mean(f5_away > f5_home)), 4)
    f5_home_wins = round(float(np.mean(f5_home > f5_away)), 4)
    f5_ties = round(float(np.mean(f5_away == f5_home)), 4)
    f5_total = f5_away + f5_home
    f5_median_total = round(float(np.median(f5_total)), 2)
    f5_exp_away = round(float(np.mean(f5_away)), 2)
    f5_exp_home = round(float(np.mean(f5_home)), 2)

    return {
        "home_prob": home_wins,
        "away_prob": away_wins,
        "edge": edge,
        "pred_home_runs": mean_home_runs,
        "pred_away_runs": mean_away_runs,
        "f5_away_prob": f5_away_wins,
        "f5_home_prob": f5_home_wins,
        "f5_tie_prob": f5_ties,
        "f5_exp_away": f5_exp_away,
        "f5_exp_home": f5_exp_home,
        "f5_median": f5_median_total
    }

def run_production_game_simulations(conn, cursor, iterations=25000):
    print("=" * 65)
    print(f"[{datetime.now()}] Launching Sequential Half-Inning State Machine (N={iterations})...")
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

    for match in slate:
        pk, home, away, home_p, away_p = match[0], match[1], match[2], match[3], match[4]
        res = simulate_full_game_sequential_state_machine(match, iterations=iterations)

        cursor.execute('''
        INSERT OR REPLACE INTO Model_Forecasts 
        (game_pk, home_team, away_team, home_prob, away_prob, predicted_edge, predicted_home_runs, predicted_away_runs, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        ''', (pk, home, away, res["home_prob"], res["away_prob"], res["edge"], res["pred_home_runs"], res["pred_away_runs"], now_ts))

        cursor.execute('''
        INSERT OR REPLACE INTO F5_Forecasts 
        (game_pk, away_team, home_team, away_starter, home_starter, f5_away_prob, f5_home_prob, f5_tie_prob, f5_exp_away_runs, f5_exp_home_runs, f5_total_runs, f5_median_total)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
        ''', (pk, away, home, away_p, home_p, res["f5_away_prob"], res["f5_home_prob"], res["f5_tie_prob"], res["f5_exp_away"], res["f5_exp_home"], round(res["f5_exp_away"] + res["f5_exp_home"], 2), res["f5_median"]))

        print(f"[{away} @ {home}] ML: {home} {res['home_prob']:.1%} | Exp Runs: {res['pred_away_runs']} - {res['pred_home_runs']} | F5 Median: {res['f5_median']}")

    conn.commit()
    print("[SUCCESS] Sequential inning simulation completed and persisted to operational tables.")

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

    iterations = kwargs.get('iterations', 25000)
    run_production_game_simulations(conn, cursor, iterations=iterations)

    if close_after:
        cursor.execute("PRAGMA wal_checkpoint(TRUNCATE);")
        conn.close()

def main():
    conn = sqlite3.connect('mlb_engine.db', timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    cursor = conn.cursor()

    run_production_game_simulations(conn, cursor, iterations=25000)

    cursor.execute("PRAGMA wal_checkpoint(TRUNCATE);")
    conn.close()

if __name__ == "__main__":
    main()
