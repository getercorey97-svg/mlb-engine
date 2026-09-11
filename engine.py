import sqlite3
import numpy as np
from datetime import datetime
from sklearn.isotonic import IsotonicRegression

def ensure_engine_schemas(cursor):
    """Guarantees all reference tables exist with auto-migration for Daily_Umpires."""
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
        bsr_per_game REAL DEFAULT 4.50,
        updated_at TEXT
    );
    CREATE TABLE IF NOT EXISTS Pitcher_Stats (
        last_name TEXT PRIMARY KEY,
        est_era REAL DEFAULT 4.20,
        updated_at TEXT
    );
    CREATE TABLE IF NOT EXISTS Park_Factors (
        home_team TEXT PRIMARY KEY,
        run_factor REAL DEFAULT 1.00
    );
    CREATE TABLE IF NOT EXISTS Bullpen_Fatigue (
        team_name TEXT PRIMARY KEY,
        fatigue_multiplier REAL DEFAULT 1.00
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
        uv_modifier REAL DEFAULT 1.00,
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
    ''')

    cursor.execute("PRAGMA table_info(Daily_Umpires);")
    cols = [c[1] for c in cursor.fetchall()]
    if 'umpire_locked' not in cols:
        cursor.execute("ALTER TABLE Daily_Umpires ADD COLUMN umpire_locked INTEGER DEFAULT 0;")

def run_ultimate_monte_carlo():
    print("=" * 65)
    print(f"[{datetime.now()}] Running Deterministic Dual-Engine Monte Carlo (BsR 1.8 + NegBinomial)")
    print("=" * 65)

    conn = sqlite3.connect('mlb_engine.db', timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=10000;")
    cursor = conn.cursor()

    ensure_engine_schemas(cursor)
    conn.commit()

    # Preload metrics into memory for lock-free fast execution
    team_bsr = {r[0]: r[1] for r in cursor.execute("SELECT team_name, COALESCE(bsr_per_game, 4.50) FROM Team_Offense").fetchall()}
    team_ops = {r[0]: r[1] for r in cursor.execute("SELECT team_name, COALESCE(ops, 0.720) FROM Team_Offense").fetchall()}
    pitcher_era = {r[0]: r[1] for r in cursor.execute("SELECT last_name, COALESCE(est_era, 4.20) FROM Pitcher_Stats").fetchall()}
    park_mods = {r[0]: r[1] for r in cursor.execute("SELECT home_team, COALESCE(run_factor, 1.00) FROM Park_Factors").fetchall()}
    bullpen_fatigue = {r[0]: r[1] for r in cursor.execute("SELECT team_name, COALESCE(fatigue_multiplier, 1.00) FROM Bullpen_Fatigue").fetchall()}
    circadian_drag = {r[0]: r[1] for r in cursor.execute("SELECT team_name, COALESCE(jet_lag_runs_penalty, 0.00) FROM Biological_Modifiers").fetchall()}
    
    # Safe umpire loading with fallback
    try:
        umpire_mods = {r[0]: (r[1], r[2]) for r in cursor.execute("SELECT game_pk, COALESCE(run_modifier, 1.00), COALESCE(umpire_locked, 0) FROM Daily_Umpires").fetchall()}
    except Exception:
        umpire_mods = {}

    cursor.execute('''
        SELECT game_pk, away_team, home_team, away_pitcher, home_pitcher, 
               COALESCE(air_density, 1.225), COALESCE(uv_modifier, 1.00)
        FROM Daily_Lineups 
        WHERE status != "Final"
    ''')
    games = cursor.fetchall()

    if not games:
        print("No active games found in Daily_Lineups to simulate.")
        conn.close()
        return

    # Train Isotonic Calibrator
    cursor.execute('''
        SELECT m.home_prob, (CASE WHEN p.home_score > p.away_score THEN 1.0 ELSE 0.0 END)
        FROM Post_Match_Analysis p
        INNER JOIN Model_Forecasts m ON p.game_pk = m.game_pk
        WHERE m.home_prob IS NOT NULL AND p.home_score IS NOT NULL
        ORDER BY p.game_pk DESC LIMIT 400
    ''')
    hist_data = cursor.fetchall()

    calibrator = None
    if len(hist_data) >= 50:
        try:
            raw_p = np.array([r[0] for r in hist_data])
            actual_w = np.array([r[1] for r in hist_data])
            calibrator = IsotonicRegression(out_of_bounds='clip')
            calibrator.fit(raw_p, actual_w)
            print(f"[CALIBRATOR] Isotonic Regression active (trained on {len(hist_data)} empirical linescores)")
        except Exception as e:
            print(f"[CALIBRATOR] Warning during training: {e}. Using raw probability.")
            calibrator = None

    it = 50000
    dispersion = 1.35

    for pk, away, home, away_p, home_p, rho, uv in games:
        # Deterministic seed anchor: seed=game_pk
        rng = np.random.default_rng(seed=int(pk))

        a_sp_last = away_p.split(' ')[-1] if away_p and away_p != "TBD" else ""
        h_sp_last = home_p.split(' ')[-1] if home_p and home_p != "TBD" else ""

        a_sp_era = pitcher_era.get(a_sp_last, 4.20)
        h_sp_era = pitcher_era.get(h_sp_last, 4.20)

        a_base_runs = team_bsr.get(away, (team_ops.get(away, 0.720) / 0.720) * 4.50)
        h_base_runs = team_bsr.get(home, (team_ops.get(home, 0.720) / 0.720) * 4.50)

        park_mult = park_mods.get(home, 1.00)
        air_drag_mult = 1.000 + ((1.225 - rho) * 1.5)
        uv_mult = uv or 1.00
        
        ump_mod, ump_locked = umpire_mods.get(pk, (1.00, 0))
        ump_badge = "🔒 LOCKED" if ump_locked == 1 else "⏳ TBD"

        a_pen_fatigue = bullpen_fatigue.get(away, 1.00)
        h_pen_fatigue = bullpen_fatigue.get(home, 1.00)
        a_circadian_penalty = circadian_drag.get(away, 0.00)

        exp_away_runs = max(0.2, (
            (a_base_runs * 0.55 * (h_sp_era / 4.20)) +
            (a_base_runs * 0.45 * h_pen_fatigue)
        ) * park_mult * air_drag_mult * uv_mult * ump_mod - a_circadian_penalty)

        exp_home_runs = max(0.2, (
            (h_base_runs * 0.55 * (a_sp_era / 4.20)) +
            (h_base_runs * 0.45 * a_pen_fatigue)
        ) * park_mult * air_drag_mult * uv_mult * ump_mod)

        va = max(exp_away_runs + 0.01, exp_away_runs * dispersion)
        vh = max(exp_home_runs + 0.01, exp_home_runs * dispersion)

        pa = max(0.01, min(0.99, exp_away_runs / va))
        ph = max(0.01, min(0.99, exp_home_runs / vh))

        na = max(0.1, (exp_away_runs ** 2) / (va - exp_away_runs))
        nh = max(0.1, (exp_home_runs ** 2) / (vh - exp_home_runs))

        away_sim = np.clip(rng.negative_binomial(na, pa, it), 0, 22)
        home_sim = np.clip(rng.negative_binomial(nh, ph, it), 0, 22)

        raw_home_prob = float(np.mean(home_sim > away_sim))

        if calibrator:
            try:
                final_home_prob = float(calibrator.predict([raw_home_prob])[0])
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

        print(f"Game {pk}: {away} ({exp_away_runs:.2f} r) @ {home} ({exp_home_runs:.2f} r) | H: {final_home_prob:.1%} | A: {final_away_prob:.1%} | Edge: {edge:.1%} | Umpire: {ump_badge}")

    conn.commit()
    conn.close()
    print("[SUCCESS] Deterministic Monte Carlo simulation completed.")

if __name__ == "__main__":
    run_ultimate_monte_carlo()
