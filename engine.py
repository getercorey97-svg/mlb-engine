import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
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

def project_starter_innings(effective_metric: float) -> tuple:
    """Computes dynamic starter workload versus bullpen distribution."""
    projected_outs = float(np.clip(27.0 - (effective_metric * 2.2), 9.0, 21.6))
    ip_projected = projected_outs / 3.0
    sp_weight = float(np.clip(ip_projected / 9.0, 0.33, 0.80))
    pen_weight = round(1.0 - sp_weight, 4)
    return sp_weight, pen_weight, round(ip_projected, 1)

def build_mlb_stacking_classifier():
    """
    Constructs the Level-1 Stacking Classifier.
    Base estimators are strictly set to n_jobs=1 to prevent thread contention
    and worker warning floods when dispatched across StackingClassifier(n_jobs=-1).
    """
    rf_base = RandomForestClassifier(
        n_estimators=100, 
        max_depth=3, 
        min_samples_leaf=10, 
        random_state=42, 
        n_jobs=1
    )
    xgb_base = XGBClassifier(
        n_estimators=80, 
        learning_rate=0.03, 
        max_depth=3, 
        subsample=0.8, 
        eval_metric='logloss', 
        random_state=42, 
        n_jobs=1
    )
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

def fit_meta_calibrator(conn):
    """Fits the stacking meta-calibrator on historical completed matches."""
    query = '''
    SELECT 
        m.predicted_home_runs,
        m.predicted_away_runs,
        m.home_prob,
        CASE WHEN p.home_score > p.away_score THEN 1.0 ELSE 0.0 END as actual_home_win
    FROM Post_Match_Analysis p
    INNER JOIN Model_Forecasts m ON p.game_pk = m.game_pk
    WHERE p.home_score IS NOT NULL AND p.away_score IS NOT NULL AND p.home_score != p.away_score
    ORDER BY p.game_pk DESC
    LIMIT 600;
    '''
    try:
        df_fit = pd.read_sql_query(query, conn)
        if len(df_fit) >= 60 and len(df_fit['actual_home_win'].unique()) > 1:
            X = df_fit[['predicted_home_runs', 'predicted_away_runs', 'home_prob']].values
            y = df_fit['actual_home_win'].values
            clf = build_mlb_stacking_classifier()
            clf.fit(X, y)
            return clf
    except Exception as e:
        print(f"[META CALIBRATOR] Bypass dynamic fitting: {e}")
    return None

def run_ultimate_monte_carlo():
    print("=" * 65)
    print(f"[{datetime.now()}] Initializing Full-Game Engine (50,000 Iterations + Stacking Ensemble)")
    print("=" * 65)

    conn = sqlite3.connect('mlb_engine.db', timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=10000;")
    cursor = conn.cursor()

    # Ingest unplayed active games
    cursor.execute('''
        SELECT d.game_pk, d.away_team, d.home_team, d.away_pitcher, d.home_pitcher,
               COALESCE(d.air_density, 1.225), COALESCE(d.uv_modifier, 5.0),
               COALESCE(u.run_modifier, 1.00)
        FROM Daily_Lineups d
        LEFT JOIN Daily_Umpires u ON d.game_pk = u.game_pk
        WHERE d.status != 'Final' AND d.game_pk NOT IN (SELECT game_pk FROM Post_Match_Analysis)
    ''')
    active_games = cursor.fetchall()

    if not active_games:
        print("[BYPASS] No unplayed games scheduled in Daily_Lineups.")
        conn.close()
        return

    # Cache operational parameters
    park_factors = {r[0]: r[1] for r in cursor.execute("SELECT home_team, run_factor FROM Park_Factors").fetchall()}
    pitcher_stats = {r[0]: (r[1], r[2], r[3]) for r in cursor.execute(
        "SELECT last_name, COALESCE(xfip, est_era, 4.20), throws, est_era FROM Pitcher_Stats"
    ).fetchall()}
    team_offense = {r[0]: (r[1], r[2], r[3], r[4]) for r in cursor.execute(
        "SELECT team_name, ops, ops_vs_rhp, ops_vs_lhp, bsr_per_game FROM Team_Offense"
    ).fetchall()}
    dynamic_mods = {r[0]: (r[1], r[2]) for r in cursor.execute(
        "SELECT team_name, offensive_modifier, pitching_modifier FROM Dynamic_Modifiers"
    ).fetchall()}
    pitcher_mods = {r[0]: r[1] for r in cursor.execute(
        "SELECT pitcher_name, f5_run_modifier FROM Pitcher_Modifiers"
    ).fetchall()}
    pen_fatigue = {r[0]: r[1] for r in cursor.execute(
        "SELECT team_name, fatigue_multiplier FROM Bullpen_Fatigue"
    ).fetchall()}

    meta_calibrator = fit_meta_calibrator(conn)
    if meta_calibrator:
        print("[META CALIBRATOR] Stacking Classifier operational across active slate.")

    iterations = 50000
    dispersion = 1.35
    now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for game in active_games:
        pk, away, home, away_sp, home_sp, rho, uv_raw, ump_run = game
        rng = np.random.default_rng(seed=int(pk))

        base_pf = park_factors.get(home, DEFAULT_PARK_FACTORS.get(home, 1.00))
        air_drag_mult = 1.000 + ((1.225 - rho) * 1.5)
        uv_glare_mult = 1.000 + (np.clip(uv_raw, 1.0, 11.0) - 5.0) * 0.005
        full_env_scalar = base_pf * air_drag_mult * uv_glare_mult * ump_run

        a_sp_ln = away_sp.split()[-1] if away_sp and away_sp != 'TBD' else ''
        h_sp_ln = home_sp.split()[-1] if home_sp and home_sp != 'TBD' else ''

        a_sp_metric, a_throws, _ = pitcher_stats.get(a_sp_ln, (4.20, 'R', 4.20))
        h_sp_metric, h_throws, _ = pitcher_stats.get(h_sp_ln, (4.20, 'R', 4.20))

        h_p_run_mod = pitcher_mods.get(home_sp, 1.0)
        a_p_run_mod = pitcher_mods.get(away_sp, 1.0)

        w_h_sp, w_h_pen, _ = project_starter_innings(h_sp_metric * h_p_run_mod)
        w_a_sp, w_a_pen, _ = project_starter_innings(a_sp_metric * a_p_run_mod)

        h_off_mod, h_pitch_mod = dynamic_mods.get(home, (1.0, 1.0))
        a_off_mod, a_pitch_mod = dynamic_mods.get(away, (1.0, 1.0))

        h_pen_fatigue = pen_fatigue.get(home, 1.00)
        a_pen_fatigue = pen_fatigue.get(away, 1.00)

        _, a_rhp, a_lhp, a_bsr = team_offense.get(away, (0.720, 0.720, 0.720, 4.50))
        _, h_rhp, h_lhp, h_bsr = team_offense.get(home, (0.720, 0.720, 0.720, 4.50))

        a_platoon_ops = a_lhp if h_throws == 'L' else a_rhp
        h_platoon_ops = h_lhp if a_throws == 'L' else h_rhp

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

        if meta_calibrator:
            try:
                feat = np.array([[exp_home_runs, exp_away_runs, raw_home_prob]])
                final_home_prob = float(meta_calibrator.predict_proba(feat)[0][1])
                final_home_prob = max(0.05, min(0.95, final_home_prob))
            except Exception:
                final_home_prob = raw_home_prob
        else:
            final_home_prob = raw_home_prob

        final_away_prob = round(1.0 - final_home_prob, 4)
        final_home_prob = round(final_home_prob, 4)
        edge = round(abs(final_home_prob - final_away_prob), 4)

        cursor.execute('''
        INSERT OR REPLACE INTO Model_Forecasts 
        (game_pk, home_team, away_team, home_prob, away_prob, predicted_edge, predicted_home_runs, predicted_away_runs, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (pk, home, away, final_home_prob, final_away_prob, edge, round(exp_home_runs, 2), round(exp_away_runs, 2), now_ts))

    conn.commit()
    cursor.execute("PRAGMA wal_checkpoint(TRUNCATE);")
    conn.close()
    print(f"[SUCCESS] Evaluated {len(active_games)} matchups at 50,000 iterations. Forecasts persisted.")

if __name__ == "__main__":
    run_ultimate_monte_carlo()
