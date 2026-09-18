import os
import sys
import sqlite3
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings('ignore')
warnings.filterwarnings('ignore', category=UserWarning)

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

CORRELATION_SIGNIFICANCE_THRESHOLD = 0.20
TTOP_SUPPRESSION_FACTOR = 0.90
TOP_ORDER_WEIGHT = 1.03
F5_VOLUME_SCALAR = 5.0 / 9.7
PARK_REGRESSION_FACTOR = 0.90

LEAGUE_AVG_BA = 0.245
LEAGUE_AVG_K_RATE = 0.222
LEAGUE_AVG_BB_RATE = 0.082

ORDER_PA_WEIGHTS = {
    1: 1.14, 2: 1.11, 3: 1.08, 4: 1.05, 5: 1.02,
    6: 0.98, 7: 0.95, 8: 0.92, 9: 0.88
}

def ensure_unified_schemas(cursor):
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
    CREATE TABLE IF NOT EXISTS Model_Forecasts (
        game_pk INTEGER PRIMARY KEY,
        home_team TEXT,
        away_team TEXT,
        home_prob REAL,
        away_prob REAL,
        predicted_edge REAL,
        predicted_home_runs REAL,
        predicted_away_runs REAL,
        timestamp TEXT
    );
    CREATE TABLE IF NOT EXISTS F5_Forecasts (
        game_pk INTEGER PRIMARY KEY,
        away_team TEXT,
        home_team TEXT,
        away_starter TEXT,
        home_starter TEXT,
        f5_away_prob REAL,
        f5_home_prob REAL,
        f5_tie_prob REAL,
        f5_exp_away_runs REAL,
        f5_exp_home_runs REAL,
        f5_total_runs REAL,
        f5_median_total REAL DEFAULT 0.0
    );
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
    CREATE TABLE IF NOT EXISTS Historical_Batter_Boxscores (
        game_pk INTEGER,
        player_name TEXT,
        team_name TEXT,
        batting_order INTEGER,
        hits INTEGER,
        ab INTEGER,
        pa INTEGER,
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
    CREATE TABLE IF NOT EXISTS Pitcher_Props (
        game_pk INTEGER,
        pitcher_name TEXT,
        team_name TEXT,
        projected_outs REAL,
        projected_strikeouts REAL,
        over_4_5_k_prob REAL,
        over_5_5_k_prob REAL,
        over_6_5_k_prob REAL,
        PRIMARY KEY (game_pk, pitcher_name)
    );
    CREATE TABLE IF NOT EXISTS Daily_Lineups (
        game_pk INTEGER PRIMARY KEY,
        game_date TEXT,
        away_team TEXT,
        home_team TEXT,
        away_pitcher TEXT,
        home_pitcher TEXT,
        lineup_status TEXT,
        air_density REAL DEFAULT 1.225,
        uv_modifier REAL DEFAULT 5.0,
        status TEXT
    );
    CREATE TABLE IF NOT EXISTS Daily_Umpires (
        game_pk INTEGER PRIMARY KEY,
        home_plate_umpire TEXT,
        run_modifier REAL DEFAULT 1.00,
        umpire_locked INTEGER DEFAULT 0,
        updated_at TEXT
    );
    CREATE TABLE IF NOT EXISTS Pitcher_Stats (
        last_name TEXT PRIMARY KEY,
        est_era REAL DEFAULT 4.20,
        xfip REAL DEFAULT 4.20,
        throws TEXT DEFAULT 'R',
        updated_at TEXT
    );
    CREATE TABLE IF NOT EXISTS Team_Offense (
        team_name TEXT PRIMARY KEY,
        ops REAL DEFAULT 0.720,
        ops_vs_rhp REAL DEFAULT 0.720,
        ops_vs_lhp REAL DEFAULT 0.720,
        bsr_per_game REAL DEFAULT 4.50,
        updated_at TEXT
    );
    CREATE TABLE IF NOT EXISTS Pitcher_Modifiers (
        pitcher_name TEXT PRIMARY KEY,
        k_modifier REAL DEFAULT 1.0,
        f5_run_modifier REAL DEFAULT 1.0,
        appearance_count INTEGER DEFAULT 0,
        last_updated TEXT
    );
    CREATE TABLE IF NOT EXISTS Dynamic_Modifiers (
        team_name TEXT PRIMARY KEY,
        offensive_modifier REAL DEFAULT 1.0,
        pitching_modifier REAL DEFAULT 1.0,
        appearance_count INTEGER DEFAULT 0,
        last_updated TEXT
    );
    CREATE TABLE IF NOT EXISTS Bullpen_Fatigue (
        team_name TEXT PRIMARY KEY,
        fatigue_multiplier REAL DEFAULT 1.00,
        rolling_ip_3d REAL DEFAULT 8.0,
        last_updated TEXT
    );
    CREATE TABLE IF NOT EXISTS Backtest_Ledger (
        run_id INTEGER PRIMARY KEY AUTOINCREMENT,
        games_evaluated INTEGER,
        brier_score REAL,
        win_accuracy REAL,
        f5_win_accuracy REAL DEFAULT 0.0,
        avg_run_error REAL,
        f5_avg_run_error REAL DEFAULT 0.0,
        batter_hit_mae REAL DEFAULT 0.0,
        batter_hit_brier REAL DEFAULT 0.0,
        executed_at TEXT
    );
    CREATE TABLE IF NOT EXISTS Feature_Weights (
        feature_name TEXT PRIMARY KEY,
        r_runs REAL,
        r_error REAL,
        beta_weight REAL,
        status TEXT,
        last_calibrated TEXT
    );
    CREATE TABLE IF NOT EXISTS Feature_Correlations (
        feature_name TEXT PRIMARY KEY,
        corr_with_total_runs REAL,
        corr_with_model_error REAL,
        sample_size INTEGER,
        anomaly_flagged INTEGER,
        last_updated TEXT
    );
    CREATE TABLE IF NOT EXISTS Park_Factors (
        home_team TEXT PRIMARY KEY,
        run_factor REAL DEFAULT 1.00
    );
    ''')

    migrations = [
        ("Pitcher_Stats", "throws", "TEXT DEFAULT 'R'"),
        ("Pitcher_Stats", "xfip", "REAL DEFAULT 4.20"),
        ("Team_Offense", "ops_vs_rhp", "REAL DEFAULT 0.720"),
        ("Team_Offense", "ops_vs_lhp", "REAL DEFAULT 0.720"),
        ("Team_Offense", "bsr_per_game", "REAL DEFAULT 4.50"),
        ("Bullpen_Fatigue", "rolling_ip_3d", "REAL DEFAULT 8.0"),
        ("Pitcher_Modifiers", "appearance_count", "INTEGER DEFAULT 0"),
        ("Dynamic_Modifiers", "appearance_count", "INTEGER DEFAULT 0"),
        ("Daily_Lineups", "uv_modifier", "REAL DEFAULT 5.0"),
        ("Daily_Umpires", "umpire_locked", "INTEGER DEFAULT 0"),
        ("Backtest_Ledger", "f5_win_accuracy", "REAL DEFAULT 0.0"),
        ("Backtest_Ledger", "f5_avg_run_error", "REAL DEFAULT 0.0"),
        ("Backtest_Ledger", "batter_hit_mae", "REAL DEFAULT 0.0"),
        ("Backtest_Ledger", "batter_hit_brier", "REAL DEFAULT 0.0"),
        ("F5_Forecasts", "f5_median_total", "REAL DEFAULT 0.0")
    ]
    for table, col, col_def in migrations:
        cursor.execute(f"PRAGMA table_info({table});")
        cols = [c[1] for c in cursor.fetchall()]
        if col not in cols:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_def};")

    cursor.execute("SELECT COUNT(*) FROM Park_Factors;")
    if cursor.fetchone()[0] == 0:
        for team, factor in DEFAULT_PARK_FACTORS.items():
            cursor.execute("INSERT OR REPLACE INTO Park_Factors (home_team, run_factor) VALUES (?, ?);", (team, factor))

def apply_bayesian_hit_shrinkage(raw_prob_over_0_5: float, ab_sample: int = 120) -> float:
    """
    Applies empirical Bayes log-odds shrinkage to compress extreme over/under
    hit probabilities toward the baseline 60.5% starter hit rate.
    """
    p_clipped = float(np.clip(raw_prob_over_0_5, 0.05, 0.95))
    logit_raw = np.log(p_clipped / (1.0 - p_clipped))
    prior_p = 0.605
    logit_prior = np.log(prior_p / (1.0 - prior_p))
    w = float(np.clip(ab_sample / (ab_sample + 80), 0.70, 0.85))
    shrunk_logit = (w * logit_raw) + ((1.0 - w) * logit_prior)
    shrunk_p = 1.0 / (1.0 + np.exp(-shrunk_logit))
    return float(np.clip(shrunk_p, 0.20, 0.82))

def log5_matchup_odds(p_batter: float, p_pitcher: float, p_league: float) -> float:
    p_b = float(np.clip(p_batter, 0.05, 0.95))
    p_p = float(np.clip(p_pitcher, 0.05, 0.95))
    p_l = float(np.clip(p_league, 0.05, 0.95))
    odds_b = p_b / (1.0 - p_b)
    odds_p = p_p / (1.0 - p_p)
    odds_l = p_l / (1.0 - p_l)
    odds_matchup = (odds_b * odds_p) / odds_l
    return float(np.clip(odds_matchup / (1.0 + odds_matchup), 0.01, 0.99))

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
    return sp_weight, pen_weight, round(ip_projected, 1)

def sync_historical_schedule_if_needed(conn, cursor, min_required=2000, days_back=180):
    cursor.execute('''
        SELECT COUNT(*) 
        FROM Post_Match_Analysis p 
        INNER JOIN Daily_Lineups d ON p.game_pk = d.game_pk 
        WHERE p.home_score IS NOT NULL AND d.home_team IS NOT NULL
    ''')
    matched_count = cursor.fetchone()[0]
    print(f"[DATABASE CHECK] Existing matched game records in DB: {matched_count} games.")

    if matched_count >= min_required:
        return

    print(f"[INGESTION] Ingesting empirical games from MLB Stats API ({matched_count} < {min_required})...")
    today = datetime.now()
    chunk_days = 30

    for chunk_start in range(0, days_back, chunk_days):
        end_dt = (today - timedelta(days=chunk_start)).strftime('%Y-%m-%d')
        start_dt = (today - timedelta(days=chunk_start + chunk_days)).strftime('%Y-%m-%d')

        url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={start_dt}&endDate={end_dt}&gameType=R&hydrate=linescore,probablePitcher"
        try:
            res = requests.get(url, timeout=15).json()
            for date_item in res.get('dates', []):
                for g in date_item.get('games', []):
                    pk = g.get('gamePk')
                    if not pk or g.get('status', {}).get('abstractGameState') != 'Final' or 'linescore' not in g:
                        continue
                    
                    teams = g.get('teams', {})
                    home = teams.get('home', {}).get('team', {}).get('name')
                    away = teams.get('away', {}).get('team', {}).get('name')
                    h_score = teams.get('home', {}).get('score')
                    a_score = teams.get('away', {}).get('score')
                    
                    if not home or not away or h_score is None or a_score is None or h_score == a_score:
                        continue

                    actual_winner = home if h_score > a_score else away
                    innings = g.get('linescore', {}).get('innings', [])
                    h_f5 = sum(inn.get('home', {}).get('runs') or 0 for inn in innings[:5])
                    a_f5 = sum(inn.get('away', {}).get('runs') or 0 for inn in innings[:5])
                    now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    cursor.execute('''
                    INSERT OR REPLACE INTO Post_Match_Analysis 
                    (game_pk, actual_winner, home_score, away_score, home_f5_score, away_f5_score, model_correct, processed_at)
                    VALUES (?, ?, ?, ?, ?, ?, -1, ?)
                    ''', (pk, actual_winner, h_score, a_score, h_f5, a_f5, now_ts))

                    home_p = teams.get('home', {}).get('probablePitcher', {}).get('fullName', 'Unknown')
                    away_p = teams.get('away', {}).get('probablePitcher', {}).get('fullName', 'Unknown')
                    rho = STADIUM_RHO_BASELINES.get(home, 1.225)

                    cursor.execute('''
                    INSERT OR REPLACE INTO Daily_Lineups 
                    (game_pk, game_date, away_team, home_team, away_pitcher, home_pitcher, lineup_status, air_density, uv_modifier, status)
                    VALUES (?, ?, ?, ?, ?, ?, 'Official', ?, 5.0, 'Final')
                    ''', (pk, g.get('gameDate', '')[:10], away, home, away_p, home_p, rho))
            conn.commit()
        except Exception as e:
            print(f"Schedule chunk ingestion error: {e}")

def sync_sample_historical_boxscores(conn, cursor, game_pks, max_games=150):
    """Caches actual starter batter boxscores for precision hit MAE/Brier backtesting."""
    cursor.execute("SELECT DISTINCT game_pk FROM Historical_Batter_Boxscores;")
    cached = set(r[0] for r in cursor.fetchall())
    to_fetch = [pk for pk in game_pks if pk not in cached][-max_games:]

    if not to_fetch:
        return

    print(f"[BOXSCORE SYNC] Ingesting empirical batter boxscores across {len(to_fetch)} slate matchups...")
    for pk in to_fetch:
        url = f"https://statsapi.mlb.com/api/v1/game/{pk}/boxscore"
        try:
            res = requests.get(url, timeout=6).json()
            teams = res.get('teams', {})
            for side in ('away', 'home'):
                t_name = teams.get(side, {}).get('team', {}).get('name', side)
                batters = teams.get(side, {}).get('batters', [])
                players = teams.get(side, {}).get('players', {})
                order_idx = 1
                for b_id in batters:
                    p_data = players.get(f"ID{b_id}", {})
                    if p_data.get('position', {}).get('abbreviation') == 'P' and len(batters) > 9:
                        continue
                    name = p_data.get('person', {}).get('fullName')
                    stats = p_data.get('stats', {}).get('batting', {})
                    hits = stats.get('hits', 0)
                    ab = stats.get('atBats', 0)
                    pa = stats.get('plateAppearances', ab)
                    if name and order_idx <= 9:
                        cursor.execute('''
                        INSERT OR REPLACE INTO Historical_Batter_Boxscores 
                        (game_pk, player_name, team_name, batting_order, hits, ab, pa)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        ''', (pk, name, t_name, order_idx, hits, ab, pa))
                        order_idx += 1
        except Exception:
            continue
    conn.commit()

def build_mlb_stacking_classifier():
    # Enforce n_jobs=1 on base models to prevent nested thread contention with top-level StackingClassifier
    rf_base = RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=10, random_state=42, n_jobs=1)
    xgb_base = XGBClassifier(n_estimators=80, learning_rate=0.03, max_depth=3, subsample=0.8, eval_metric='logloss', random_state=42, n_jobs=1)
    level_1_meta = LogisticRegression(penalty='l2', C=0.5, solver='lbfgs', max_iter=1000)
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    return StackingClassifier(
        estimators=[('rf', rf_base), ('xgb', xgb_base)],
        final_estimator=level_1_meta,
        cv=cv_strategy,
        stack_method='predict_proba',
        passthrough=True,
        n_jobs=-1
    )

def run_chronological_walk_forward_backtest(conn, cursor, max_eval=2000, iterations=50000):
    print("=" * 65)
    print(f"[{datetime.now()}] Initializing Pure Walk-Forward Historical Calibration Backtest (N={iterations} Iterations)...")
    print("=" * 65)

    query = '''
    SELECT 
        p.game_pk,
        p.home_score,
        p.away_score,
        COALESCE(p.home_f5_score, 0),
        COALESCE(p.away_f5_score, 0),
        COALESCE(d.home_team, 'Home'),
        COALESCE(d.away_team, 'Away'),
        COALESCE(d.home_pitcher, 'Unknown'),
        COALESCE(d.away_pitcher, 'Unknown'),
        COALESCE(d.air_density, 1.225),
        COALESCE(d.uv_modifier, 5.0),
        COALESCE(u.run_modifier, 1.00),
        COALESCE(ps_home.throws, 'R'),
        COALESCE(ps_away.throws, 'R'),
        COALESCE(ps_home.xfip, ps_home.est_era, 4.20),
        COALESCE(ps_away.xfip, ps_away.est_era, 4.20),
        COALESCE(t_home.ops_vs_rhp, 0.720),
        COALESCE(t_home.ops_vs_lhp, 0.720),
        COALESCE(t_away.ops_vs_rhp, 0.720),
        COALESCE(t_away.ops_vs_lhp, 0.720),
        COALESCE(t_home.bsr_per_game, 4.50),
        COALESCE(t_away.bsr_per_game, 4.50),
        COALESCE(d.game_date, '2025-04-01')
    FROM Post_Match_Analysis p
    INNER JOIN Daily_Lineups d ON p.game_pk = d.game_pk
    LEFT JOIN Daily_Umpires u ON p.game_pk = u.game_pk
    LEFT JOIN Pitcher_Stats ps_home ON d.home_pitcher LIKE '%' || ps_home.last_name
    LEFT JOIN Pitcher_Stats ps_away ON d.away_pitcher LIKE '%' || ps_away.last_name
    LEFT JOIN Team_Offense t_home ON d.home_team = t_home.team_name
    LEFT JOIN Team_Offense t_away ON d.away_team = t_away.team_name
    WHERE p.home_score IS NOT NULL 
      AND p.away_score IS NOT NULL 
      AND p.home_score != p.away_score
    ORDER BY d.game_date ASC, p.game_pk ASC
    LIMIT ?
    '''
    cursor.execute(query, (max_eval,))
    records = cursor.fetchall()

    if len(records) < 50:
        print(f"[BYPASS] Insufficient dataset for walk-forward evaluation (N = {len(records)} < 50).")
        return

    all_pks = [r[0] for r in records]
    sync_sample_historical_boxscores(conn, cursor, all_pks, max_games=150)

    cursor.execute("SELECT game_pk, player_name, team_name, batting_order, hits, ab FROM Historical_Batter_Boxscores;")
    box_lookup = {}
    for r in cursor.fetchall():
        box_lookup.setdefault(r[0], []).append((r[1], r[2], r[3], r[4], r[5]))

    print(f"Replaying chronological slate progression across {len(records)} empirical matchups at {iterations} Monte Carlo iterations...")

    sim_team_off, sim_team_pitch, sim_team_count = {}, {}, {}
    sim_pitcher_f5, sim_pitcher_k, sim_pitcher_count = {}, {}, {}
    team_recent_workload = {}

    eval_history = []
    batter_eval_history = []
    calibrator_pool_X, calibrator_pool_y = [], []
    calibrator = None

    dispersion = 1.35
    now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for idx, row in enumerate(records):
        (pk, home_score, away_score, h_f5, a_f5, home, away,
         home_p, away_p, rho, uv_raw, ump_run, h_throws, a_throws,
         h_sp_metric, a_sp_metric, h_ops_rhp, h_ops_lhp, a_ops_rhp, a_ops_lhp,
         h_bsr, a_bsr, g_date) = row

        try:
            curr_dt = datetime.strptime(g_date, "%Y-%m-%d")
        except Exception:
            curr_dt = datetime.now()

        # 1. Rolling Bullpen Fatigue Tracking
        def compute_sim_pen_workload(team_name):
            recent_games = team_recent_workload.get(team_name, [])
            valid = [r for r in recent_games if 1 <= (curr_dt - r[0]).days <= 3]
            if not valid:
                return 1.00, 8.0
            tot_ip = sum(r[1] for r in valid)
            played_yest = any((curr_dt - r[0]).days == 1 for r in valid)
            strain = (tot_ip - 8.0) * 0.025
            b2b_tax = 0.03 if (played_yest and len(valid) >= 2) else 0.00
            three_day_tax = 0.05 if len(valid) >= 3 else 0.00
            mult = round(float(np.clip(1.00 + strain + b2b_tax + three_day_tax, 0.85, 1.25)), 4)
            return mult, round(tot_ip, 1)

        h_pen_fatigue, h_ip_3d = compute_sim_pen_workload(home)
        a_pen_fatigue, a_ip_3d = compute_sim_pen_workload(away)

        # 2. Bayesian Modifier Shrinkage (N / 10.0)
        w_h_team = min(1.0, sim_team_count.get(home, 0) / 10.0)
        w_a_team = min(1.0, sim_team_count.get(away, 0) / 10.0)
        h_off_mod = w_h_team * sim_team_off.get(home, 1.0) + (1.0 - w_h_team) * 1.0
        h_pitch_mod = w_h_team * sim_team_pitch.get(home, 1.0) + (1.0 - w_h_team) * 1.0
        a_off_mod = w_a_team * sim_team_off.get(away, 1.0) + (1.0 - w_a_team) * 1.0
        a_pitch_mod = w_a_team * sim_team_pitch.get(away, 1.0) + (1.0 - w_a_team) * 1.0

        w_h_p = min(1.0, sim_pitcher_count.get(home_p, 0) / 10.0)
        w_a_p = min(1.0, sim_pitcher_count.get(away_p, 0) / 10.0)
        h_p_run_mod = w_h_p * sim_pitcher_f5.get(home_p, 1.0) + (1.0 - w_h_p) * 1.0
        a_p_run_mod = w_a_p * sim_pitcher_f5.get(away_p, 1.0) + (1.0 - w_a_p) * 1.0

        # 3. Dynamic Starter Length Weighting
        w_h_sp, w_h_pen, h_ip_proj = project_starter_innings(h_sp_metric * h_p_run_mod)
        w_a_sp, w_a_pen, a_ip_proj = project_starter_innings(a_sp_metric * a_p_run_mod)

        # 4. Environmental Scaling
        base_pf = DEFAULT_PARK_FACTORS.get(home, 1.00)
        air_drag_mult = 1.000 + ((1.225 - rho) * 1.5)
        uv_glare_mult = 1.000 + (np.clip(uv_raw, 1.0, 11.0) - 5.0) * 0.005
        full_env_scalar = base_pf * air_drag_mult * uv_glare_mult * ump_run

        # 5. Full-Game Platoon & Expected Runs
        a_platoon_ops = a_ops_lhp if h_throws == 'L' else a_ops_rhp
        h_platoon_ops = h_ops_lhp if a_throws == 'L' else h_ops_rhp

        a_sp_matchup_runs = a_bsr * (a_platoon_ops / 0.720)
        h_sp_matchup_runs = h_bsr * (h_platoon_ops / 0.720)

        exp_away_runs = max(
            0.2, 
            ((a_sp_matchup_runs * a_off_mod * w_h_sp * (h_sp_metric / 4.20) * h_p_run_mod) + 
             (a_bsr * a_off_mod * w_h_pen * h_pen_fatigue * h_pitch_mod)) * full_env_scalar
        )
        exp_home_runs = max(
            0.2, 
            ((h_sp_matchup_runs * h_off_mod * w_a_sp * (a_sp_metric / 4.20) * a_p_run_mod) + 
             (h_bsr * h_off_mod * w_a_pen * a_pen_fatigue * a_pitch_mod)) * full_env_scalar
        )

        # 6. High-Precision Monte Carlo Simulation (50,000 Iterations)
        rng = np.random.default_rng(seed=int(pk))
        va = max(exp_away_runs + 0.01, exp_away_runs * dispersion)
        vh = max(exp_home_runs + 0.01, exp_home_runs * dispersion)
        pa = max(0.01, min(0.99, exp_away_runs / va))
        ph = max(0.01, min(0.99, exp_home_runs / vh))
        na = max(0.1, (exp_away_runs ** 2) / (va - exp_away_runs))
        nh = max(0.1, (exp_home_runs ** 2) / (vh - exp_home_runs))

        away_sim = np.clip(rng.negative_binomial(na, pa, iterations), 0, 22)
        home_sim = np.clip(rng.negative_binomial(nh, ph, iterations), 0, 22)

        p_home_reg = float(np.mean(home_sim > away_sim))
        p_tie_reg = float(np.mean(home_sim == away_sim))
        raw_home_prob = p_home_reg + (0.53 * p_tie_reg)

        # 7. Stacking Classifier Ensemble
        if idx > 0 and idx % 200 == 0 and len(calibrator_pool_y) >= 150:
            try:
                X_fit = np.array(calibrator_pool_X)
                y_fit = np.array(calibrator_pool_y)
                if len(np.unique(y_fit)) > 1:
                    calibrator = build_mlb_stacking_classifier()
                    calibrator.fit(X_fit, y_fit)
            except Exception:
                calibrator = None

        if calibrator:
            try:
                feat = np.array([[exp_home_runs, exp_away_runs, raw_home_prob]])
                final_home_prob = float(calibrator.predict_proba(feat)[0][1])
                final_home_prob = max(0.05, min(0.95, final_home_prob))
            except Exception:
                final_home_prob = raw_home_prob
        else:
            final_home_prob = raw_home_prob

        final_away_prob = round(1.0 - final_home_prob, 4)
        final_home_prob = round(final_home_prob, 4)
        edge = round(abs(final_home_prob - final_away_prob), 4)

        # 8. Decoupled F5 Expectancy
        f5_regressed_pf = 1.000 + (base_pf - 1.000) * PARK_REGRESSION_FACTOR
        f5_env_scalar = f5_regressed_pf * air_drag_mult * uv_glare_mult * ump_run

        a_xera_f5 = np.clip(a_sp_metric, 1.5, 9.0) * a_p_run_mod * TTOP_SUPPRESSION_FACTOR
        h_xera_f5 = np.clip(h_sp_metric, 1.5, 9.0) * h_p_run_mod * TTOP_SUPPRESSION_FACTOR

        a_off_f5 = (a_platoon_ops / 0.720) * a_off_mod * TOP_ORDER_WEIGHT
        h_off_f5 = (h_platoon_ops / 0.720) * h_off_mod * TOP_ORDER_WEIGHT

        lam_f5_a = max(0.10, (h_xera_f5 * a_off_f5 * f5_env_scalar) * F5_VOLUME_SCALAR)
        lam_f5_h = max(0.10, (a_xera_f5 * h_off_f5 * f5_env_scalar) * F5_VOLUME_SCALAR)

        f5_disp = 1.22
        f5_va, f5_vh = lam_f5_a * f5_disp, lam_f5_h * f5_disp
        f5_pa, f5_ph = max(0.01, min(0.99, lam_f5_a / f5_va)), max(0.01, min(0.99, lam_f5_h / f5_vh))
        f5_na, f5_nh = max(0.1, (lam_f5_a ** 2) / (f5_va - lam_f5_a)), max(0.1, (lam_f5_h ** 2) / (f5_vh - lam_f5_h))

        f5_sim_a = np.clip(rng.negative_binomial(f5_na, f5_pa, iterations), 0, 15)
        f5_sim_h = np.clip(rng.negative_binomial(f5_nh, f5_ph, iterations), 0, 15)
        f5_sim_tot = f5_sim_a + f5_sim_h

        f5_median_continuous = float(np.median(f5_sim_tot))
        f5_mean_total = round(lam_f5_a + lam_f5_h, 2)

        f5_away_prob = float(np.mean(f5_sim_a > f5_sim_h))
        f5_home_prob = float(np.mean(f5_sim_h > f5_sim_a))
        f5_tie_prob = float(np.mean(f5_sim_a == f5_sim_h))

        # 9. Individual Batter Hit Evaluation with Empirical Bayes Shrinkage
        box_batters = box_lookup.get(pk, [])
        if box_batters:
            env_hit_scalar = (1.000 + (base_pf - 1.000) * 0.70) * (1.000 + ((1.225 - rho) * 0.8))
            for b_name, b_team, b_order, b_act_hits, b_act_ab in box_batters:
                is_h = (b_team == home)
                tm_runs = exp_home_runs if is_h else exp_away_runs
                opp_throws = a_throws if is_h else h_throws
                opp_era = a_sp_metric if is_h else h_sp_metric
                w_sp = w_a_sp if is_h else w_h_sp

                proj_pa, proj_ab = project_endogenous_plate_appearances(b_order, tm_runs, is_h, final_home_prob if is_h else final_away_prob)
                
                # Log5 Matchup Synthesis
                matchup_k = log5_matchup_odds(0.220, 0.220, LEAGUE_AVG_K_RATE)
                p_in_play = max(0.40, 1.0 - matchup_k - 0.085)
                matchup_ba_sp = log5_matchup_odds(0.250, float(np.clip(opp_era / 17.5, 0.18, 0.32)), LEAGUE_AVG_BA) * env_hit_scalar
                p_hit_pa_sp = p_in_play * (matchup_ba_sp / max(0.01, 1.0 - LEAGUE_AVG_K_RATE - LEAGUE_AVG_BB_RATE))
                p_hit_pa_pen = (1.0 - 0.220 - 0.085) * (0.250 * env_hit_scalar / max(0.01, 1.0 - LEAGUE_AVG_K_RATE - LEAGUE_AVG_BB_RATE))

                p_hit_pa = float(np.clip(w_sp * p_hit_pa_sp + (1.0 - w_sp) * p_hit_pa_pen, 0.10, 0.45))
                p_hit_ab = float(np.clip(p_hit_pa / 0.895, 0.12, 0.48))

                b_sim_hits = rng.binomial(int(np.round(proj_ab)), p_hit_ab, 5000)
                pred_hits_exp = float(proj_ab * p_hit_ab)
                raw_over_0_5 = float(np.mean(b_sim_hits >= 1))
                
                # Calibrated through Empirical Bayes Log-Odds Shrinkage
                pred_over_0_5 = apply_bayesian_hit_shrinkage(raw_over_0_5, ab_sample=120)

                batter_eval_history.append({
                    'hit_err': abs(b_act_hits - pred_hits_exp),
                    'hit_brier': (pred_over_0_5 - (1.0 if b_act_hits >= 1 else 0.0)) ** 2
                })

        cursor.execute('''
        INSERT OR REPLACE INTO Model_Forecasts 
        (game_pk, home_team, away_team, home_prob, away_prob, predicted_edge, predicted_home_runs, predicted_away_runs, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (pk, home, away, final_home_prob, final_away_prob, edge, round(exp_home_runs, 2), round(exp_away_runs, 2), now_ts))

        cursor.execute('''
        INSERT OR REPLACE INTO F5_Forecasts 
        (game_pk, away_team, home_team, away_starter, home_starter, f5_away_prob, f5_home_prob, f5_tie_prob, f5_exp_away_runs, f5_exp_home_runs, f5_total_runs, f5_median_total)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (pk, away, home, away_p, home_p, round(f5_away_prob, 4), round(f5_home_prob, 4), round(f5_tie_prob, 4), round(lam_f5_a, 2), round(lam_f5_h, 2), f5_mean_total, round(f5_median_continuous, 2)))

        actual_home_win = 1.0 if home_score > away_score else 0.0
        model_home_pick = 1.0 if final_home_