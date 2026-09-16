import sqlite3
import requests
import numpy as np
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

def ensure_backtest_tables(cursor):
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
    CREATE TABLE IF NOT EXISTS Backtest_Ledger (
        run_id INTEGER PRIMARY KEY AUTOINCREMENT,
        games_evaluated INTEGER,
        brier_score REAL,
        win_accuracy REAL,
        avg_run_error REAL,
        executed_at TEXT
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
    ''')

def build_mlb_stacking_classifier():
    """Constructs the Level-1 Stacked Generalization ensemble for hold-out validation."""
    rf_base = RandomForestClassifier(n_estimators=200, max_depth=6, min_samples_leaf=4, random_state=42, n_jobs=-1)
    xgb_base = XGBClassifier(n_estimators=150, learning_rate=0.05, max_depth=5, eval_metric='logloss', random_state=42, n_jobs=-1)
    
    level_1_meta = LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=1000)
    cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    stacked_model = StackingClassifier(
        estimators=[('rf', rf_base), ('xgb', xgb_base)],
        final_estimator=level_1_meta,
        cv=cv_strategy,
        stack_method='predict_proba',
        passthrough=True,
        n_jobs=-1
    )
    return stacked_model

def run_backtest_engine(target_games=1600):
    print("=" * 65)
    print(f"[{datetime.now()}] Initializing SOTA Machine Learning Backtest ({target_games} Games)...")
    print("=" * 65)

    conn = sqlite3.connect('mlb_engine.db', timeout=30)
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA busy_timeout=10000;")
    ensure_backtest_tables(cursor)
    conn.commit()

    cursor.execute("SELECT game_pk FROM Post_Match_Analysis")
    seen_pks = {row[0] for row in cursor.fetchall()}
    print(f"Existing historical records in database: {len(seen_pks)} games.")

    today = datetime.now()
    all_games = []
    chunk_days = 30
    days_back = 150 

    print("Fetching bulk schedule chunks from MLB Stats API...")
    for chunk_start in range(0, days_back, chunk_days):
        end_dt = (today - timedelta(days=chunk_start)).strftime('%Y-%m-%d')
        start_dt = (today - timedelta(days=chunk_start + chunk_days)).strftime('%Y-%m-%d')

        bulk_url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={start_dt}&endDate={end_dt}&gameType=R&hydrate=linescore,probablePitcher"
        try:
            res = requests.get(bulk_url, timeout=15).json()
            for date_item in res.get('dates', []):
                for g in date_item.get('games', []):
                    pk = g.get('gamePk')
                    if pk and pk not in seen_pks and g.get('status', {}).get('abstractGameState') == 'Final' and 'linescore' in g:
                        seen_pks.add(pk)
                        all_games.append(g)
        except Exception as e:
            print(f"Chunk fetch error ({start_dt} to {end_dt}): {e}")

        if len(all_games) >= target_games:
            break

    print(f"Ingested {len(all_games)} completed MLB games for ML evaluation.")

    if not all_games:
        print("[BYPASS] No new games retrieved for backtesting.")
        conn.close()
        return

    # Sort games chronologically to simulate real-time EWMA learning
    all_games.sort(key=lambda x: x.get('gamePk', 0))

    # In-memory simulation of EWMA, Multi-Target, and Bayesian Shrinkage weights
    sim_team_off = {}
    sim_team_pitch = {}
    sim_team_count = {}
    sim_pitcher_mod = {}
    sim_pitcher_count = {}
    sim_bullpen_fatigue = {}

    X_data = []
    y_data = []
    run_errors = []
    now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    processed_count = 0
    for g in all_games[:target_games]:
        pk = g['gamePk']
        teams = g.get('teams', {})
        home = teams.get('home', {}).get('team', {}).get('name')
        away = teams.get('away', {}).get('team', {}).get('name')

        if not home or not away:
            continue
        
        home_score = teams.get('home', {}).get('score')
        away_score = teams.get('away', {}).get('score')

        if home_score is None or away_score is None:
            continue

        actual_winner = home if home_score > away_score else away
        actual_home_win = 1.0 if home_score > away_score else 0.0

        # Extract F5 and Late Inning scores
        innings = g.get('linescore', {}).get('innings', [])
        h_f5 = sum(inn.get('home', {}).get('runs') or 0 for inn in innings[:5])
        a_f5 = sum(inn.get('away', {}).get('runs') or 0 for inn in innings[:5])
        h_late = sum(inn.get('home', {}).get('runs') or 0 for inn in innings[5:9])
        a_late = sum(inn.get('away', {}).get('runs') or 0 for inn in innings[5:9])

        home_p = teams.get('home', {}).get('probablePitcher', {}).get('fullName', 'Unknown')
        away_p = teams.get('away', {}).get('probablePitcher', {}).get('fullName', 'Unknown')

        # Apply Bayesian Shrinkage to simulated weights
        w_h_team = min(1.0, sim_team_count.get(home, 0) / 10.0)
        w_a_team = min(1.0, sim_team_count.get(away, 0) / 10.0)
        
        h_off_mod = w_h_team * sim_team_off.get(home, 1.0) + (1.0 - w_h_team) * 1.0
        h_pitch_mod = w_h_team * sim_team_pitch.get(home, 1.0) + (1.0 - w_h_team) * 1.0
        a_off_mod = w_a_team * sim_team_off.get(away, 1.0) + (1.0 - w_a_team) * 1.0
        a_pitch_mod = w_a_team * sim_team_pitch.get(away, 1.0) + (1.0 - w_a_team) * 1.0

        w_h_p = min(1.0, sim_pitcher_count.get(home_p, 0) / 10.0)
        w_a_p = min(1.0, sim_pitcher_count.get(away_p, 0) / 10.0)
        h_p_mod = w_h_p * sim_pitcher_mod.get(home_p, 1.0) + (1.0 - w_h_p) * 1.0
        a_p_mod = w_a_p * sim_pitcher_mod.get(away_p, 1.0) + (1.0 - w_a_p) * 1.0

        rho = STADIUM_RHO_BASELINES.get(home, 1.225)
        park_mult = DEFAULT_PARK_FACTORS.get(home, 1.00)
        air_drag_mult = 1.000 + ((1.225 - rho) * 1.5)

        # Base runs with dynamic modifiers
        base_h = 4.45 * park_mult * air_drag_mult * h_off_mod * (0.55 * a_p_mod + 0.45 * sim_bullpen_fatigue.get(away, 1.0) * a_pitch_mod)
        base_a = 4.25 * park_mult * air_drag_mult * a_off_mod * (0.55 * h_p_mod + 0.45 * sim_bullpen_fatigue.get(home, 1.0) * h_pitch_mod)

        denom = (base_h ** 1.83) + (base_a ** 1.83)
        raw_home_prob = round((base_h ** 1.83) / denom, 4) if denom != 0 else 0.5
        
        # Features for ML Pipeline
        X_data.append([base_h, base_a, raw_home_prob])
        y_data.append(actual_home_win)
        run_errors.append(abs((home_score + away_score) - (base_h + base_a)))

        # Chronological EWMA & Multi-Target Updates
        pred_home_f5, pred_away_f5 = base_h * 0.55, base_a * 0.55
        pred_home_late, pred_away_late = base_h * 0.45, base_a * 0.45

        # 1. Pitcher F5 Updates
        for p_name, pred_f5, act_f5 in [(home_p, pred_away_f5, a_f5), (away_p, pred_home_f5, h_f5)]:
            err = act_f5 - pred_f5
            alpha = min(0.30, 0.10 + (abs(err) * 0.02))
            old_mod = sim_pitcher_mod.get(p_name, 1.0)
            sim_pitcher_mod[p_name] = max(0.60, min(1.40, alpha * (old_mod + err * 0.15) + (1.0 - alpha) * old_mod))
            sim_pitcher_count[p_name] = sim_pitcher_count.get(p_name, 0) + 1

        # 2. Team Late Inning Updates
        for t_name, pred_late, act_late, is_off in [(home, pred_home_late, h_late, True), (away, pred_home_late, h_late, False), (away, pred_away_late, a_late, True), (home, pred_away_late, a_late, False)]:
            err = act_late - pred_late
            alpha = min(0.30, 0.10 + (abs(err) * 0.02))
            if is_off:
                old_mod = sim_team_off.get(t_name, 1.0)
                sim_team_off[t_name] = max(0.60, min(1.40, alpha * (old_mod + err * 0.15) + (1.0 - alpha) * old_mod))
            else:
                old_mod = sim_team_pitch.get(t_name, 1.0)
                sim_team_pitch[t_name] = max(0.60, min(1.40, alpha * (old_mod + err * 0.15) + (1.0 - alpha) * old_mod))
            sim_team_count[t_name] = sim_team_count.get(t_name, 0) + 1

        # 3. Bullpen Fatigue Updates
        for t_name, pred_late, act_late in [(away, pred_home_late, h_late), (home, pred_away_late, a_late)]:
            err = act_late - pred_late
            alpha = min(0.30, 0.10 + (abs(err) * 0.02))
            old_fatigue = sim_bullpen_fatigue.get(t_name, 1.0)
            sim_bullpen_fatigue[t_name] = max(0.70, min(1.30, alpha * (old_fatigue + err * 0.15) + (1.0 - alpha) * old_fatigue))

        cursor.execute('''
        INSERT OR REPLACE INTO Post_Match_Analysis 
        (game_pk, actual_winner, home_score, away_score, home_f5_score, away_f5_score, model_correct, processed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (pk, actual_winner, home_score, away_score, h_f5, a_f5, -1, now_ts))

        processed_count += 1

    conn.commit()

    if processed_count < 100:
        print(f"[BYPASS] Not enough data for a valid Machine Learning hold-out validation (N={processed_count}). Needed 100+.")
        conn.close()
        return

    print("Splitting dataset into 80% Training / 20% Hold-Out Testing...")
    X = np.array(X_data)
    y = np.array(y_data)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training SOTA Stacking Classifier (RF + XGB -> LogReg) on historical data...")
    stack = build_mlb_stacking_classifier()
    stack.fit(X_train, y_train)

    print("Executing inferences on unseen validation set...")
    y_pred_proba = stack.predict_proba(X_test)[:, 1]
    y_pred_bin = stack.predict(X_test)
    
    final_brier = round(brier_score_loss(y_test, y_pred_proba), 4)
    final_acc = round(accuracy_score(y_test, y_pred_bin), 4)
    final_run_err = round(float(np.mean(run_errors)), 2)

    cursor.execute('''
    INSERT INTO Backtest_Ledger (games_evaluated, brier_score, win_accuracy, avg_run_error, executed_at)
    VALUES (?, ?, ?, ?, ?)
    ''', (len(y_test), final_brier, final_acc, final_run_err, now_ts))

    conn.commit()
    cursor.execute("PRAGMA wal_checkpoint(TRUNCATE);")
    conn.close()

    print("\n" + "=" * 65)
    print("⚡ MACHINE LEARNING VALIDATION COMPLETED (80/20 SPLIT)")
    print(f"• Total Validation Set Evaluated : {len(y_test)} Games (Unseen)")
    print(f"• Stacked Ensemble Accuracy    : {final_acc:.2%}")
    print(f"• Calibrated Brier Score       : {final_brier:.4f} (Lower is better, < 0.25 is profitable)")
    print(f"• Average Total Run Error      : {final_run_err:.2f} Runs/Game")
    print("=" * 65)

if __name__ == "__main__":
    run_backtest_engine(target_games=1600)
