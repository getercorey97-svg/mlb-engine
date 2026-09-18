import sqlite3
import numpy as np
from datetime import datetime, timedelta
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import warnings

warnings.filterwarnings('ignore')

try:
    from auto_calibration import audit_rolling_telemetry, ensure_telemetry_schemas
except ImportError:
    def ensure_telemetry_schemas(cursor):
        pass
    def audit_rolling_telemetry(conn, cursor, window_size=50):
        return 0.0, False, False

def ensure_engine_schemas(cursor):
    """Guarantees all production reference, telemetry, and modifier tables exist."""
    cursor.executescript('''
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
    CREATE TABLE IF NOT EXISTS Team_Offense (
        team_name TEXT PRIMARY KEY,
        ops REAL DEFAULT 0.720,
        ops_vs_rhp REAL DEFAULT 0.720,
        ops_vs_lhp REAL DEFAULT 0.720,
        bsr_per_game REAL DEFAULT 4.50,
        updated_at TEXT
    );
    CREATE TABLE IF NOT EXISTS Pitcher_Stats (
        last_name TEXT PRIMARY KEY,
        est_era REAL DEFAULT 4.20,
        xfip REAL DEFAULT 4.20,
        throws TEXT DEFAULT 'R',
        updated_at TEXT
    );
    CREATE TABLE IF NOT EXISTS Park_Factors (
        home_team TEXT PRIMARY KEY,
        run_factor REAL DEFAULT 1.00
    );
    CREATE TABLE IF NOT EXISTS Bullpen_Fatigue (
        team_name TEXT PRIMARY KEY,
        fatigue_multiplier REAL DEFAULT 1.00,
        rolling_ip_3d REAL DEFAULT 8.0,
        last_updated TEXT
    );
    CREATE TABLE IF NOT EXISTS Biological_Modifiers (
        team_name TEXT PRIMARY KEY,
        jet_lag_runs_penalty REAL DEFAULT 0.00
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
        uv_modifier REAL DEFAULT 5.00,
        status TEXT
    );
    CREATE TABLE IF NOT EXISTS Daily_Umpires (
        game_pk INTEGER PRIMARY KEY,
        home_plate_umpire TEXT,
        run_modifier REAL DEFAULT 1.00,
        umpire_locked INTEGER DEFAULT 0,
        updated_at TEXT
    );
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
    CREATE TABLE IF NOT EXISTS Feature_Weights (
        feature_name TEXT PRIMARY KEY,
        r_runs REAL,
        r_error REAL,
        beta_weight REAL,
        status TEXT,
        last_calibrated TEXT
    );
    CREATE TABLE IF NOT EXISTS Backtest_Ledger (
        run_id INTEGER PRIMARY KEY AUTOINCREMENT,
        games_evaluated INTEGER,
        brier_score REAL,
        win_accuracy REAL,
        f5_win_accuracy REAL DEFAULT 0.0,
        avg_run_error REAL,
        f5_avg_run_error REAL DEFAULT 0.0,
        executed_at TEXT
    );
    CREATE TABLE IF NOT EXISTS System_Telemetry (
        telemetry_id INTEGER PRIMARY KEY AUTOINCREMENT,
        evaluated_games INTEGER,
        rolling_brier REAL,
        rolling_win_acc REAL,
        rolling_f5_acc REAL,
        rolling_full_run_mae REAL,
        rolling_f5_median_mae REAL,
        rolling_f5_signed_bias REAL,
        rolling_full_signed_bias REAL,
        drift_flag INTEGER DEFAULT 0,
        retrain_recommended INTEGER DEFAULT 0,
        logged_at TEXT
    );
    CREATE TABLE IF NOT EXISTS Calibration_Offsets (
        market_type TEXT PRIMARY KEY,
        active_bias_offset REAL DEFAULT 0.00,
        consecutive_drifts INTEGER DEFAULT 0,
        last_adjusted TEXT
    );
    ''')

    migrations = [
        ("Daily_Umpires", "umpire_locked", "INTEGER DEFAULT 0"),
        ("Pitcher_Stats", "xfip", "REAL DEFAULT 4.20"),
        ("Pitcher_Stats", "throws", "TEXT DEFAULT 'R'"),
        ("Team_Offense", "ops_vs_rhp", "REAL DEFAULT 0.720"),
        ("Team_Offense", "ops_vs_lhp", "REAL DEFAULT 0.720"),
        ("Team_Offense", "bsr_per_game", "REAL DEFAULT 4.50"),
        ("Bullpen_Fatigue", "rolling_ip_3d", "REAL DEFAULT 8.0"),
        ("Pitcher_Modifiers", "appearance_count", "INTEGER DEFAULT 0"),
        ("Dynamic_Modifiers", "appearance_count", "INTEGER DEFAULT 0"),
        ("Backtest_Ledger", "f5_win_accuracy", "REAL DEFAULT 0.0"),
        ("Backtest_Ledger", "f5_avg_run_error", "REAL DEFAULT 0.0")
    ]
    for table, col, col_def in migrations:
        cursor.execute(f"PRAGMA table_info({table});")
        cols = [c[1] for c in cursor.fetchall()]
        if col not in cols:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_def};")

def project_starter_innings(effective_metric: float) -> tuple:
    """Computes dynamic starting pitcher innings expectation and derived workload weights."""
    projected_outs = float(np.clip(27.0 - (effective_metric * 2.2), 9.0, 21.6))
    ip_projected = projected_outs / 3.0
    sp_weight = float(np.clip(ip_projected / 9.0, 0.33, 0.80))
    pen_weight = round(1.0 - sp_weight, 4)
    return sp_weight, pen_weight, round(ip_projected, 1)

def get_rolling_bullpen_workload(cursor, team_name, ref_date_str):
    """Calculates physical 3-day rolling reliever innings and congestion penalty."""
    try:
        cursor.execute('''
            SELECT d.game_date, p.home_score, p.away_score, p.home_f5_score, p.away_f5_score
            FROM Daily_Lineups d
            JOIN Post_Match_Analysis p ON d.game_pk = p.game_pk
            WHERE (d.home_team = ? OR d.away_team = ?)
              AND d.game_date BETWEEN date(?, '-3 days') AND date(?, '-1 day')
              AND p.actual_winner IS NOT NULL
        ''', (team_name, team_name, ref_date_str, ref_date_str))
        rows = cursor.fetchall()
        if not rows:
            return 1.00, 8.0

        ref_dt = datetime.strptime(ref_date_str, "%Y-%m-%d")
        yesterday_str = (ref_dt - timedelta(days=1)).strftime("%Y-%m-%d")

        total_reliever_ip = 0.0
        played_yesterday = False

        for g_date, h_score, a_score, h_f5, a_f5 in rows:
            if g_date == yesterday_str:
                played_yesterday = True
            late_runs = max(0, (h_score - (h_f5 or 0))) + max(0, (a_score - (a_f5 or 0)))
            total_reliever_ip += 4.0 + max(0.0, (late_runs - 3) * 0.25)

        ip_strain = (total_reliever_ip - 8.0) * 0.025
        back_to_back_tax = 0.03 if (played_yesterday and len(rows) >= 2) else 0.00
        three_day_tax = 0.05 if len(rows) >= 3 else 0.00

        multiplier = round(float(np.clip(1.00 + ip_strain + back_to_back_tax + three_day_tax, 0.85, 1.25)), 4)
        return multiplier, round(total_reliever_ip, 1)
    except Exception:
        return 1.00, 8.0

def probability_to_american(prob: float) -> str:
    prob = max(0.01, min(0.99, prob))
    if prob >= 0.5:
        odds = -(prob / (1.0 - prob)) * 100
    else:
        odds = ((1.0 - prob) / prob) * 100
    odds_int = int(round(odds))
    return f"{max(-10000, min(10000, odds_int)):+d}"

def update_readme(cursor):
    today_date = datetime.now().strftime("%Y-%m-%d")
    
    cursor.execute('''
        SELECT m.game_pk, m.away_team, m.home_team, m.away_prob, m.home_prob, 
               m.predicted_edge, m.predicted_away_runs, m.predicted_home_runs,
               l.away_pitcher, l.home_pitcher, 
               COALESCE(u.home_plate_umpire, 'Awaiting HP Umpire'), COALESCE(u.umpire_locked, 0)
        FROM Model_Forecasts m
        INNER JOIN Daily_Lineups l ON m.game_pk = l.game_pk
        LEFT JOIN Daily_Umpires u ON m.game_pk = u.game_pk
        WHERE l.status != 'Final' AND l.game_pk NOT IN (SELECT game_pk FROM Post_Match_Analysis)
    ''')
    active_games = cursor.fetchall()

    lines = [
        f"# MLB Game Predictions & Value Engine ({today_date})",
        "",
        "### 🎟️ Primary Value Bets (Full Game Moneyline)",
        "",
        "| Matchup | Best Pick | Fair Odds | Edge | Projected Score | Pitchers | Umpire State |",
        "| :--- | :--- | :---: | :---: | :---: | :--- | :--- |",
    ]

    if not active_games:
        lines.append("| No active games remaining today | - | - | - | - | - | - |")
    else:
        for row in active_games:
            (pk, away, home, p_away, p_home, edge, r_away, r_home, p_away_name, p_home_name, ump, ump_locked) = row
            best_pick = home if p_home >= p_away else away
            win_prob = max(p_home, p_away)
            fair_odds = probability_to_american(win_prob)
            ump_display = f"`{ump}`" if ump_locked == 1 else f"⏳ {ump}"
            edge_display = f"+{edge*100:.1f}%"
            lines.append(
                f"| {away} @ {home} | **{best_pick}** | {fair_odds} | "
                f"{edge_display} | {r_away:.1f} - {r_home:.1f} | {p_away_name} vs {p_home_name} | {ump_display} |"
            )

    lines.extend(["", "---", "*(Note: Engine uses upsert mechanisms on `game_pk`. Row duplication is disabled.)*", ""])
    with open("README.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

def build_mlb_stacking_classifier():
    rf_base = RandomForestClassifier(n_estimators=100, max_depth=3, min_samples_leaf=10, random_state=42, n_jobs=-1)
    xgb_base = XGBClassifier(n_estimators=80, learning_rate=0.03, max_depth=3, subsample=0.8, eval_metric='logloss', random_state=42, n_jobs=-1)
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

def run_ultimate_monte_carlo():
    print("=" * 65)
    print(f"[{datetime.now()}] Running Self-Optimizing Dual-Engine Monte Carlo (Dynamic Starter IP + NegBinomial)")
    print("=" * 65)

    conn = sqlite3.connect('mlb_engine.db', timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=10000;")
    cursor = conn.cursor()

    ensure_engine_schemas(cursor)
    ensure_telemetry_schemas(cursor)
    conn.commit()

    # Autonomous System Telemetry & Drift Check
    bias, is_drift, retrain_needed = audit_rolling_telemetry(conn, cursor, window_size=50)

    today_str = datetime.now().strftime("%Y-%m-%d")

    team_bsr = {r[0]: r[1] for r in cursor.execute("SELECT team_name, COALESCE(bsr_per_game, 4.50) FROM Team_Offense").fetchall()}
    team_ops = {r[0]: r[1] for r in cursor.execute("SELECT team_name, COALESCE(ops, 0.720) FROM Team_Offense").fetchall()}
    platoon_ops = {r[0]: (r[1], r[2]) for r in cursor.execute(
        "SELECT team_name, COALESCE(ops_vs_rhp, 0.720), COALESCE(ops_vs_lhp, 0.720) FROM Team_Offense"
    ).fetchall()}

    pitcher_data = {r[0]: (r[1], r[2]) for r in cursor.execute(
        "SELECT last_name, COALESCE(xfip, est_era, 4.20), COALESCE(throws, 'R') FROM Pitcher_Stats"
    ).fetchall()}
    
    park_mods = {r[0]: r[1] for r in cursor.execute("SELECT home_team, COALESCE(run_factor, 1.00) FROM Park_Factors").fetchall()}
    circadian_drag = {r[0]: r[1] for r in cursor.execute("SELECT team_name, COALESCE(jet_lag_runs_penalty, 0.00) FROM Biological_Modifiers").fetchall()}
    
    dynamic_mods = {r[0]: (r[1], r[2], r[3]) for r in cursor.execute(
        "SELECT team_name, COALESCE(offensive_modifier, 1.0), COALESCE(pitching_modifier, 1.0), COALESCE(appearance_count, 0) FROM Dynamic_Modifiers"
    ).fetchall()}
    pitcher_mods = {r[0]: (r[1], r[2], r[3]) for r in cursor.execute(
        "SELECT pitcher_name, COALESCE(k_modifier, 1.0), COALESCE(f5_run_modifier, 1.0), COALESCE(appearance_count, 0) FROM Pitcher_Modifiers"
    ).fetchall()}

    try:
        umpire_mods = {r[0]: (r[1], r[2], r[3]) for r in cursor.execute(
            "SELECT game_pk, COALESCE(run_modifier, 1.00), COALESCE(umpire_locked, 0), COALESCE(home_plate_umpire, 'TBD') FROM Daily_Umpires"
        ).fetchall()}
    except Exception:
        umpire_mods = {}

    cursor.execute('''
        SELECT game_pk, away_team, home_team, away_pitcher, home_pitcher, 
               COALESCE(air_density, 1.225), COALESCE(uv_modifier, 5.00), game_date
        FROM Daily_Lineups 
        WHERE status != "Final" AND game_pk NOT IN (SELECT game_pk FROM Post_Match_Analysis)
    ''')
    games = cursor.fetchall()

    if not games:
        print("No active games found. Generating current README.")
        update_readme(cursor)
        conn.close()
        return

    # Ingest Empirical Completed Linescores
    cursor.execute('''
        SELECT m.predicted_home_runs, m.predicted_away_runs, m.home_prob, 
               (CASE WHEN p.home_score > p.away_score THEN 1.0 ELSE 0.0 END)
        FROM Post_Match_Analysis p
        INNER JOIN Model_Forecasts m ON p.game_pk = m.game_pk
        WHERE m.home_prob IS NOT NULL AND p.home_score IS NOT NULL
        ORDER BY p.game_pk ASC LIMIT 3000
    ''')
    hist_data = cursor.fetchall()

    calibrator = None
    if len(hist_data) >= 100:
        try:
            X_train = np.array([[r[0], r[1], r[2]] for r in hist_data])
            y_train = np.array([r[3] for r in hist_data])
            if len(np.unique(y_train)) > 1:
                calibrator = build_mlb_stacking_classifier()
                
                # Apply Recency-Weighted Sample Weights if Drift Detected
                if retrain_needed:
                    print("[CALIBRATOR] Autonomous Recency-Weighted Re-Fit Active (Drift Triggered)...")
                    recency_weights = np.exp(np.linspace(-0.5, 0.0, len(y_train)))
                    try:
                        calibrator.fit(X_train, y_train, sample_weight=recency_weights)
                    except TypeError:
                        calibrator.fit(X_train, y_train)
                else:
                    calibrator.fit(X_train, y_train)
                    
                print(f"[CALIBRATOR] Regularized Stacking Ensemble active (trained on {len(hist_data)} empirical linescores)")
        except Exception as e:
            print(f"[CALIBRATOR] Error: {e}. Defaulting to Monte Carlo probabilities.")
            calibrator = None

    it = 50000
    dispersion = 1.35

    for pk, away, home, away_p, home_p, rho, uv_raw, g_date in games:
        rng = np.random.default_rng(seed=int(pk))
        ref_date = g_date if g_date else today_str

        a_sp_last = away_p.split(' ')[-1] if away_p and away_p != "TBD" else ""
        h_sp_last = home_p.split(' ')[-1] if home_p and home_p != "TBD" else ""

        a_sp_metric, a_sp_throws = pitcher_data.get(a_sp_last, (4.20, 'R'))
        h_sp_metric, h_sp_throws = pitcher_data.get(h_sp_last, (4.20, 'R'))

        away_rhp, away_lhp = platoon_ops.get(away, (0.720, 0.720))
        home_rhp, home_lhp = platoon_ops.get(home, (0.720, 0.720))

        a_platoon_ops = away_lhp if h_sp_throws == 'L' else away_rhp
        h_platoon_ops = home_lhp if a_sp_throws == 'L' else home_rhp

        a_base_runs = team_bsr.get(away, (team_ops.get(away, 0.720) / 0.720) * 4.50)
        h_base_runs = team_bsr.get(home, (team_ops.get(home, 0.720) / 0.720) * 4.50)

        a_sp_matchup_runs = a_base_runs * (a_platoon_ops / 0.720)
        h_sp_matchup_runs = h_base_runs * (h_platoon_ops / 0.720)

        a_off_mod_raw, a_pitch_mod_raw, a_team_n = dynamic_mods.get(away, (1.0, 1.0, 0))
        h_off_mod_raw, h_pitch_mod_raw, h_team_n = dynamic_mods.get(home, (1.0, 1.0, 0))
        
        w_a_team = min(1.0, a_team_n / 10.0)
        w_h_team = min(1.0, h_team_n / 10.0)
        
        a_off_mod = w_a_team * a_off_mod_raw + (1.0 - w_a_team) * 1.0
        a_pitch_mod = w_a_team * a_pitch_mod_raw + (1.0 - w_a_team) * 1.0
        h_off_mod = w_h_team * h_off_mod_raw + (1.0 - w_h_team) * 1.0
        h_pitch_mod = w_h_team * h_pitch_mod_raw + (1.0 - w_h_team) * 1.0

        a_p_k_mod, a_p_run_mod_raw, a_p_n = pitcher_mods.get(away_p, (1.0, 1.0, 0))
        if a_p_run_mod_raw == 1.0 and a_sp_last:
            for name, mods in pitcher_mods.items():
                if name.endswith(a_sp_last):
                    a_p_run_mod_raw = mods[1]
                    a_p_n = mods[2]
                    break
        w_a_p = min(1.0, a_p_n / 10.0)
        a_p_run_mod = w_a_p * a_p_run_mod_raw + (1.0 - w_a_p) * 1.0

        h_p_k_mod, h_p_run_mod_raw, h_p_n = pitcher_mods.get(home_p, (1.0, 1.0, 0))
        if h_p_run_mod_raw == 1.0 and h_sp_last:
            for name, mods in pitcher_mods.items():
                if name.endswith(h_sp_last):
                    h_p_run_mod_raw = mods[1]
                    h_p_n = mods[2]
                    break
        w_h_p = min(1.0, h_p_n / 10.0)
        h_p_run_mod = w_h_p * h_p_run_mod_raw + (1.0 - w_h_p) * 1.0

        w_h_sp, w_h_pen, h_ip_proj = project_starter_innings(h_sp_metric * h_p_run_mod)
        w_a_sp, w_a_pen, a_ip_proj = project_starter_innings(a_sp_metric * a_p_run_mod)

        # 3-Day Physical Bullpen Workload
        a_pen_fatigue, a_ip_3d = get_rolling_bullpen_workload(cursor, away, ref_date)
        h_pen_fatigue, h_ip_3d = get_rolling_bullpen_workload(cursor, home, ref_date)

        cursor.execute("INSERT OR REPLACE INTO Bullpen_Fatigue (team_name, fatigue_multiplier, rolling_ip_3d, last_updated) VALUES (?, ?, ?, ?)", (away, a_pen_fatigue, a_ip_3d, today_str))
        cursor.execute("INSERT OR REPLACE INTO Bullpen_Fatigue (team_name, fatigue_multiplier, rolling_ip_3d, last_updated) VALUES (?, ?, ?, ?)", (home, h_pen_fatigue, h_ip_3d, today_str))

        park_mult = park_mods.get(home, 1.00)
        air_drag_mult = 1.000 + ((1.225 - rho) * 1.5)
        uv_glare_mult = 1.000 + (np.clip(uv_raw, 1.0, 11.0) - 5.0) * 0.005

        ump_mod, ump_locked, ump_name = umpire_mods.get(pk, (1.00, 0, 'TBD'))
        ump_badge = f"🔒 {ump_name}" if ump_locked == 1 and ump_name != "Unknown / TBD" else "⏳ TBD"
        a_circadian_penalty = circadian_drag.get(away, 0.00)

        # Dynamic Starter Inning-Weighted Run Expectancies
        exp_away_runs = max(
            0.2, 
            ((a_sp_matchup_runs * a_off_mod * w_h_sp * (h_sp_metric / 4.20) * h_p_run_mod) + 
             (a_base_runs * a_off_mod * w_h_pen * h_pen_fatigue * h_pitch_mod)) 
            * park_mult * air_drag_mult * uv_glare_mult * ump_mod - a_circadian_penalty
        )
        exp_home_runs = max(
            0.2, 
            ((h_sp_matchup_runs * h_off_mod * w_a_sp * (a_sp_metric / 4.20) * a_p_run_mod) + 
             (h_base_runs * h_off_mod * w_a_pen * a_pen_fatigue * a_pitch_mod)) 
            * park_mult * air_drag_mult * uv_glare_mult * ump_mod
        )

        va, vh = max(exp_away_runs + 0.01, exp_away_runs * dispersion), max(exp_home_runs + 0.01, exp_home_runs * dispersion)
        pa, ph = max(0.01, min(0.99, exp_away_runs / va)), max(0.01, min(0.99, exp_home_runs / vh))
        na, nh = max(0.1, (exp_away_runs ** 2) / (va - exp_away_runs)), max(0.1, (exp_home_runs ** 2) / (vh - exp_home_runs))

        away_sim = np.clip(rng.negative_binomial(na, pa, it), 0, 22)
        home_sim = np.clip(rng.negative_binomial(nh, ph, it), 0, 22)

        p_home_reg = float(np.mean(home_sim > away_sim))
        p_tie_reg = float(np.mean(home_sim == away_sim))
        raw_home_prob = p_home_reg + (0.53 * p_tie_reg)

        if calibrator:
            try:
                feature_vector = np.array([[exp_home_runs, exp_away_runs, raw_home_prob]])
                final_home_prob = float(calibrator.predict_proba(feature_vector)[0][1])
                final_home_prob = max(0.05, min(0.95, final_home_prob))
            except Exception:
                final_home_prob = raw_home_prob
        else:
            final_home_prob = raw_home_prob

        final_away_prob = round(1.0 - final_home_prob, 4)
        final_home_prob = round(final_home_prob, 4)
        edge = round(abs(final_home_prob - final_away_prob), 4)
        now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        cursor.execute('''
        INSERT OR REPLACE INTO Model_Forecasts 
        (game_pk, home_team, away_team, home_prob, away_prob, predicted_edge, predicted_home_runs, predicted_away_runs, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (pk, home, away, final_home_prob, final_away_prob, edge, round(exp_home_runs, 2), round(exp_away_runs, 2), now_ts))

        print(f"Game {pk}: {away} ({exp_away_runs:.2f} r, SP {a_ip_proj} IP) @ {home} ({exp_home_runs:.2f} r, SP {h_ip_proj} IP) | H: {final_home_prob:.1%} | A: {final_away_prob:.1%} | Edge: {edge:.1%} | Pen(A/H): {a_pen_fatigue:.2f}/{h_pen_fatigue:.2f} | Umpire: {ump_badge}")

    update_readme(cursor)
    conn.commit()
    cursor.execute("PRAGMA wal_checkpoint(TRUNCATE);")
    conn.close()
    print("[SUCCESS] SOTA Monte Carlo simulation completed and README updated.")

if __name__ == "__main__":
    run_ultimate_monte_carlo()
