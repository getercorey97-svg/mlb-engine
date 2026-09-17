import sqlite3
import requests
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, brier_score_loss
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings('ignore')

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

CORRELATION_SIGNIFICANCE_THRESHOLD = 0.20

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
        last_updated TEXT
    );
    CREATE TABLE IF NOT EXISTS Backtest_Ledger (
        run_id INTEGER PRIMARY KEY AUTOINCREMENT,
        games_evaluated INTEGER,
        brier_score REAL,
        win_accuracy REAL,
        avg_run_error REAL,
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
    ''')

    migrations = [
        ("Pitcher_Stats", "throws", "TEXT DEFAULT 'R'"),
        ("Pitcher_Stats", "xfip", "REAL DEFAULT 4.20"),
        ("Team_Offense", "ops_vs_rhp", "REAL DEFAULT 0.720"),
        ("Team_Offense", "ops_vs_lhp", "REAL DEFAULT 0.720"),
        ("Pitcher_Modifiers", "appearance_count", "INTEGER DEFAULT 0"),
        ("Dynamic_Modifiers", "appearance_count", "INTEGER DEFAULT 0"),
        ("Daily_Umpires", "umpire_locked", "INTEGER DEFAULT 0")
    ]
    for table, col, col_def in migrations:
        cursor.execute(f"PRAGMA table_info({table});")
        cols = [c[1] for c in cursor.fetchall()]
        if col not in cols:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_def};")

def sync_historical_schedule_if_needed(conn, cursor, min_required=200):
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

    print(f"[INGESTION] Valid matched dataset below target ({matched_count} < {min_required}). Ingesting empirical games from MLB Stats API...")
    today = datetime.now()
    chunk_days = 30
    days_back = 180

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
                    
                    # Skip suspended or tied games
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
            print(f"Schedule chunk ingestion error ({start_dt} to {end_dt}): {e}")

def build_mlb_stacking_classifier():
    rf_base = RandomForestClassifier(n_estimators=250, max_depth=5, min_samples_leaf=6, random_state=42, n_jobs=-1)
    xgb_base = XGBClassifier(n_estimators=180, learning_rate=0.03, max_depth=4, subsample=0.8, colsample_bytree=0.8, eval_metric='logloss', random_state=42, n_jobs=-1)
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

def run_ml_backtest(conn, cursor, max_eval=1600):
    print("=" * 65)
    print(f"[{datetime.now()}] Initializing Clean Baseline Replay Backtest...")
    print("=" * 65)

    query = '''
    SELECT 
        p.game_pk,
        p.home_score,
        p.away_score,
        COALESCE(p.home_f5_score, 0),
        COALESCE(p.away_f5_score, 0),
        COALESCE(d.home_team, m.home_team),
        COALESCE(d.away_team, m.away_team),
        COALESCE(d.home_pitcher, 'Unknown'),
        COALESCE(d.away_pitcher, 'Unknown'),
        COALESCE(d.air_density, 1.225),
        COALESCE(d.uv_modifier, 5.0),
        COALESCE(u.run_modifier, 1.00),
        COALESCE(ps_home.throws, 'R'),
        COALESCE(ps_away.throws, 'R'),
        COALESCE(t_home.ops_vs_rhp, 0.720),
        COALESCE(t_home.ops_vs_lhp, 0.720),
        COALESCE(t_away.ops_vs_rhp, 0.720),
        COALESCE(t_away.ops_vs_lhp, 0.720)
    FROM Post_Match_Analysis p
    LEFT JOIN Daily_Lineups d ON p.game_pk = d.game_pk
    LEFT JOIN Model_Forecasts m ON p.game_pk = m.game_pk
    LEFT JOIN Daily_Umpires u ON p.game_pk = u.game_pk
    LEFT JOIN Pitcher_Stats ps_home ON d.home_pitcher LIKE '%' || ps_home.last_name
    LEFT JOIN Pitcher_Stats ps_away ON d.away_pitcher LIKE '%' || ps_away.last_name
    LEFT JOIN Team_Offense t_home ON d.home_team = t_home.team_name
    LEFT JOIN Team_Offense t_away ON d.away_team = t_away.team_name
    WHERE p.home_score IS NOT NULL 
      AND p.away_score IS NOT NULL 
      AND p.home_score != p.away_score
      AND (d.home_team IS NOT NULL OR m.home_team IS NOT NULL)
    ORDER BY p.game_pk ASC
    LIMIT ?
    '''
    cursor.execute(query, (max_eval,))
    records = cursor.fetchall()

    if len(records) < 100:
        print(f"[BYPASS] Insufficient dataset for ML Hold-Out validation (N={len(records)} < 100).")
        return

    print(f"Replaying chronological history across {len(records)} games from baseline 1.00...")

    sim_team_off, sim_team_pitch, sim_team_count = {}, {}, {}
    sim_pitcher_mod, sim_pitcher_count = {}, {}
    sim_bullpen_fatigue = {}

    X_data, y_data, run_errors = [], [], []
    now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for row in records:
        (pk, home_score, away_score, h_f5, a_f5, home, away, 
         home_p, away_p, rho, uv, ump_run, h_sp_throws, a_sp_throws,
         h_ops_rhp, h_ops_lhp, a_ops_rhp, a_ops_lhp) = row
         
        h_late = max(0, home_score - h_f5)
        a_late = max(0, away_score - a_f5)
        actual_home_win = 1.0 if home_score > away_score else 0.0

        # Bayesian Shrinkage towards neutral 1.00
        w_h_team = min(1.0, sim_team_count.get(home, 0) / 15.0)
        w_a_team = min(1.0, sim_team_count.get(away, 0) / 15.0)
        h_off_mod = w_h_team * sim_team_off.get(home, 1.0) + (1.0 - w_h_team) * 1.0
        h_pitch_mod = w_h_team * sim_team_pitch.get(home, 1.0) + (1.0 - w_h_team) * 1.0
        a_off_mod = w_a_team * sim_team_off.get(away, 1.0) + (1.0 - w_a_team) * 1.0
        a_pitch_mod = w_a_team * sim_team_pitch.get(away, 1.0) + (1.0 - w_a_team) * 1.0

        w_h_p = min(1.0, sim_pitcher_count.get(home_p, 0) / 10.0)
        w_a_p = min(1.0, sim_pitcher_count.get(away_p, 0) / 10.0)
        h_p_mod = w_h_p * sim_pitcher_mod.get(home_p, 1.0) + (1.0 - w_h_p) * 1.0
        a_p_mod = w_a_p * sim_pitcher_mod.get(away_p, 1.0) + (1.0 - w_a_p) * 1.0

        park_mult = DEFAULT_PARK_FACTORS.get(home, 1.00)
        air_drag_mult = 1.000 + ((1.225 - rho) * 1.5)
        uv_glare_mult = 1.000 + (np.clip(uv, 1.0, 11.0) - 5.0) * 0.005

        # Platoon Differential
        a_platoon_ops = a_ops_lhp if h_sp_throws == 'L' else a_ops_rhp
        h_platoon_ops = h_ops_lhp if a_sp_throws == 'L' else h_ops_rhp
        net_platoon_diff = ((h_platoon_ops / 0.720) - (a_platoon_ops / 0.720))

        env_scalar = park_mult * air_drag_mult * uv_glare_mult * ump_run

        base_h = 4.45 * (h_platoon_ops / 0.720) * h_off_mod * (0.55 * a_p_mod + 0.45 * sim_bullpen_fatigue.get(away, 1.0) * a_pitch_mod) * env_scalar
        base_a = 4.25 * (a_platoon_ops / 0.720) * a_off_mod * (0.55 * h_p_mod + 0.45 * sim_bullpen_fatigue.get(home, 1.0) * h_pitch_mod) * env_scalar

        denom = (base_h ** 1.83) + (base_a ** 1.83)
        raw_home_prob = round((base_h ** 1.83) / denom, 4) if denom != 0 else 0.50

        # Multi-factor Feature Vector
        feature_vector = [
            base_h, base_a, base_h - base_a,
            base_h / max(0.5, (base_h + base_a)),
            raw_home_prob, park_mult, air_drag_mult, uv_glare_mult,
            ump_run, net_platoon_diff, h_off_mod, a_off_mod,
            h_pitch_mod, a_pitch_mod, h_p_mod, a_p_mod
        ]

        X_data.append(feature_vector)
        y_data.append(actual_home_win)
        run_errors.append(abs((home_score + away_score) - (base_h + base_a)))

        cursor.execute('''
        INSERT OR REPLACE INTO Model_Forecasts 
        (game_pk, home_team, away_team, home_prob, away_prob, predicted_edge, predicted_home_runs, predicted_away_runs, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (pk, home, away, raw_home_prob, 1.0 - raw_home_prob, round(raw_home_prob - 0.5, 4), round(base_h, 2), round(base_a, 2), now_ts))

        # Aligned EWMA Updates
        pred_home_f5, pred_away_f5 = base_h * 0.55, base_a * 0.55
        pred_home_late, pred_away_late = base_h * 0.45, base_a * 0.45

        # 1. Starting Pitcher F5 EWMA
        for p_name, pred_f5, act_f5 in [(home_p, pred_away_f5, a_f5), (away_p, pred_home_f5, h_f5)]:
            err = act_f5 - pred_f5
            alpha = min(0.12, 0.03 + (abs(err) * 0.01))
            old_mod = sim_pitcher_mod.get(p_name, 1.0)
            sim_pitcher_mod[p_name] = max(0.70, min(1.30, alpha * (old_mod + err * 0.05) + (1.0 - alpha) * old_mod))
            sim_pitcher_count[p_name] = sim_pitcher_count.get(p_name, 0) + 1

        # 2. Team Offense & Pitching Late-Inning
        for t_name, pred_late, act_late, is_off in [(home, pred_home_late, h_late, True), (away, pred_home_late, h_late, False), (away, pred_away_late, a_late, True), (home, pred_away_late, a_late, False)]:
            err = act_late - pred_late
            alpha = min(0.10, 0.02 + (abs(err) * 0.008))
            if is_off:
                old_mod = sim_team_off.get(t_name, 1.0)
                sim_team_off[t_name] = max(0.70, min(1.30, alpha * (old_mod + err * 0.04) + (1.0 - alpha) * old_mod))
            else:
                old_mod = sim_team_pitch.get(t_name, 1.0)
                sim_team_pitch[t_name] = max(0.70, min(1.30, alpha * (old_mod + err * 0.04) + (1.0 - alpha) * old_mod))
            sim_team_count[t_name] = sim_team_count.get(t_name, 0) + 1

        # 3. Bullpen Fatigue Workload
        for t_name, pred_late, act_late in [(away, pred_home_late, h_late), (home, pred_away_late, a_late)]:
            err = act_late - pred_late
            alpha = min(0.10, 0.02 + (abs(err) * 0.008))
            old_fatigue = sim_bullpen_fatigue.get(t_name, 1.0)
            sim_bullpen_fatigue[t_name] = max(0.80, min(1.25, alpha * (old_fatigue + err * 0.04) + (1.0 - alpha) * old_fatigue))

    print(f"Committing {len(sim_pitcher_mod)} pitcher weights and {len(sim_team_off)} team weights to operational memory for live pipeline consumption...")
    for p_name, mod in sim_pitcher_mod.items():
        if p_name != 'Unknown':
            cursor.execute('''
            INSERT OR REPLACE INTO Pitcher_Modifiers (pitcher_name, k_modifier, f5_run_modifier, appearance_count, last_updated)
            VALUES (?, 1.0, ?, ?, ?)
            ''', (p_name, round(mod, 4), sim_pitcher_count.get(p_name, 0), now_ts))

    all_teams = set(list(sim_team_off.keys()) + list(sim_team_pitch.keys()))
    for t_name in all_teams:
        off_mod = sim_team_off.get(t_name, 1.0)
        pitch_mod = sim_team_pitch.get(t_name, 1.0)
        count = sim_team_count.get(t_name, 0)
        cursor.execute('''
        INSERT OR REPLACE INTO Dynamic_Modifiers (team_name, offensive_modifier, pitching_modifier, appearance_count, last_updated)
        VALUES (?, ?, ?, ?, ?)
        ''', (t_name, round(off_mod, 4), round(pitch_mod, 4), count, now_ts))

    for t_name, fatigue in sim_bullpen_fatigue.items():
        cursor.execute('''
        INSERT OR REPLACE INTO Bullpen_Fatigue (team_name, fatigue_multiplier, last_updated)
        VALUES (?, ?, ?)
        ''', (t_name, round(fatigue, 4), now_ts))

    conn.commit()

    X = np.array(X_data)
    y = np.array(y_data)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print("Fitting Calibrated SOTA Level-1 Stacking Classifier...")
    stack = build_mlb_stacking_classifier()
    stack.fit(X_train, y_train)

    y_pred_proba = stack.predict_proba(X_test)[:, 1]
    y_pred_bin = stack.predict(X_test)

    final_brier = round(float(brier_score_loss(y_test, y_pred_proba)), 4)
    final_acc = round(float(accuracy_score(y_test, y_pred_bin)), 4)
    final_run_err = round(float(np.mean(run_errors)), 2)

    cursor.execute('''
    INSERT INTO Backtest_Ledger (games_evaluated, brier_score, win_accuracy, avg_run_error, executed_at)
    VALUES (?, ?, ?, ?, ?)
    ''', (len(y_test), final_brier, final_acc, final_run_err, now_ts))
    conn.commit()

    print("\n" + "=" * 65)
    print("⚡ MACHINE LEARNING VALIDATION COMPLETED (80/20 SPLIT)")
    print(f"• Total Validation Set Evaluated : {len(y_test)} Games (Unseen)")
    print(f"• Stacked Ensemble Accuracy    : {final_acc:.2%}")
    print(f"• Calibrated Brier Score       : {final_brier:.4f} (Target < 0.2500)")
    print(f"• Average Total Run Error      : {final_run_err:.2f} Runs/Game")
    print("=" * 65)

def run_correlation_sweep(conn, cursor):
    print("\n" + "=" * 65)
    print(f"[{datetime.now()}] Sweeping Extended Feature Correlation Matrix & Calibrating Betas...")
    print("=" * 65)

    query = '''
    SELECT 
        p.game_pk,
        m.predicted_home_runs,
        m.predicted_away_runs,
        p.home_score as actual_home_runs,
        p.away_score as actual_away_runs,
        (p.home_score - m.predicted_home_runs) as home_error_delta,
        (p.away_score - m.predicted_away_runs) as away_error_delta,
        COALESCE(d.air_density, 1.225) as air_density,
        COALESCE(d.uv_modifier, 5.0) as uv_modifier,
        COALESCE(u.run_modifier, 1.00) as umpire_modifier,
        COALESCE(ps_home.throws, 'R') as home_sp_throws,
        COALESCE(ps_away.throws, 'R') as away_sp_throws,
        COALESCE(t_home.ops_vs_rhp, 0.720) as home_ops_rhp,
        COALESCE(t_home.ops_vs_lhp, 0.720) as home_ops_lhp,
        COALESCE(t_away.ops_vs_rhp, 0.720) as away_ops_rhp,
        COALESCE(t_away.ops_vs_lhp, 0.720) as away_ops_lhp
    FROM Post_Match_Analysis p
    INNER JOIN Model_Forecasts m ON p.game_pk = m.game_pk
    LEFT JOIN Daily_Lineups d ON p.game_pk = d.game_pk
    LEFT JOIN Daily_Umpires u ON p.game_pk = u.game_pk
    LEFT JOIN Pitcher_Stats ps_home ON d.home_pitcher LIKE '%' || ps_home.last_name
    LEFT JOIN Pitcher_Stats ps_away ON d.away_pitcher LIKE '%' || ps_away.last_name
    LEFT JOIN Team_Offense t_home ON d.home_team = t_home.team_name
    LEFT JOIN Team_Offense t_away ON d.away_team = t_away.team_name
    WHERE p.home_score IS NOT NULL AND m.predicted_home_runs IS NOT NULL
    '''
    df = pd.read_sql_query(query, conn)

    if len(df) < 25:
        print(f"[BYPASS] Insufficient matched pairs for correlation sweep (N = {len(df)} < 25).")
        return

    df['total_abs_error'] = (df['home_error_delta'].abs() + df['away_error_delta'].abs())
    df['actual_total_runs'] = df['actual_home_runs'] + df['actual_away_runs']
    
    # Feature transformations
    df['uv_glare'] = (df['uv_modifier'].clip(1.0, 11.0) - 5.0) * 0.005
    df['umpire_k_zone'] = 1.000 - ((df['umpire_modifier'] - 1.000) * 1.6)
    
    away_platoon = np.where(df['home_sp_throws'] == 'L', df['away_ops_lhp'], df['away_ops_rhp'])
    home_platoon = np.where(df['away_sp_throws'] == 'L', df['home_ops_lhp'], df['home_ops_rhp'])
    df['net_platoon_edge'] = (home_platoon + away_platoon) - 1.440

    features = {
        'air_density': df['air_density'],
        'uv_glare': df['uv_glare'],
        'umpire_run_mod': df['umpire_modifier'],
        'umpire_k_zone': df['umpire_k_zone'],
        'net_platoon_edge': df['net_platoon_edge']
    }
    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    for feat_name, series in features.items():
        if series.std() == 0:
            continue

        r_runs = float(series.corr(df['actual_total_runs']))
        r_error = float(series.corr(df['total_abs_error']))
        beta = float(np.clip(-r_error * 0.15, -0.05, 0.05))
        is_anomaly = 1 if (abs(r_runs) >= CORRELATION_SIGNIFICANCE_THRESHOLD or abs(r_error) >= CORRELATION_SIGNIFICANCE_THRESHOLD) else 0

        cursor.execute('''
        INSERT OR REPLACE INTO Feature_Correlations
        (feature_name, corr_with_total_runs, corr_with_model_error, sample_size, anomaly_flagged, last_updated)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (feat_name, round(r_runs, 4), round(r_error, 4), len(df), is_anomaly, now_str))

        status_str = "HIGH_SENSITIVITY" if is_anomaly else "STABLE"
        cursor.execute('''
        INSERT OR REPLACE INTO Feature_Weights
        (feature_name, r_runs, r_error, beta_weight, status, last_calibrated)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (feat_name, round(r_runs, 4), round(r_error, 4), round(beta, 4), status_str, now_str))

        flag_str = "🚨 [HIGH ANOMALY]" if is_anomaly else "   [STABLE]"
        print(f"{flag_str} {feat_name:<20} | r(Runs): {r_runs:+.3f} | r(Error): {r_error:+.3f} | beta: {beta:+.4f}")

    conn.commit()
    print("[SUCCESS] Feature correlations and operational weights committed.")

def main():
    conn = sqlite3.connect('mlb_engine.db', timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=10000;")
    cursor = conn.cursor()

    ensure_unified_schemas(cursor)
    conn.commit()

    sync_historical_schedule_if_needed(conn, cursor, min_required=200)
    run_ml_backtest(conn, cursor, max_eval=1600)
    run_correlation_sweep(conn, cursor)

    cursor.execute("PRAGMA wal_checkpoint(TRUNCATE);")
    conn.close()

if __name__ == "__main__":
    main()
