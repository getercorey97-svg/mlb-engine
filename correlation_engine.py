import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime

CORRELATION_SIGNIFICANCE_THRESHOLD = 0.25

def ensure_correlation_schemas(cursor):
    """Guarantees all join tables exist before sweeping matrix."""
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
        air_density REAL,
        uv_modifier REAL,
        status TEXT
    );
    CREATE TABLE IF NOT EXISTS Daily_Umpires (
        game_pk INTEGER PRIMARY KEY,
        home_plate_umpire TEXT,
        run_modifier REAL
    );
    CREATE TABLE IF NOT EXISTS Esoteric_Signals (
        game_pk INTEGER PRIMARY KEY,
        geomagnetic_kp REAL DEFAULT 2.0,
        solar_xray_flux REAL DEFAULT 1.0,
        home_media_pressure INTEGER DEFAULT 0,
        home_media_tone REAL DEFAULT 0.0,
        roster_birthday_active INTEGER DEFAULT 0,
        captured_at TEXT
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

def run_correlation_engine():
    print("=" * 65)
    print(f"[{datetime.now()}] Initializing High-Speed Matrix Sweeper...")
    print("=" * 65)

    conn = sqlite3.connect('mlb_engine.db', timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=10000;")
    cursor = conn.cursor()

    ensure_correlation_schemas(cursor)
    conn.commit()

    query = '''
    SELECT 
        p.game_pk,
        m.home_prob,
        m.away_prob,
        m.predicted_home_runs,
        m.predicted_away_runs,
        p.home_score as actual_home_runs,
        p.away_score as actual_away_runs,
        p.model_correct,
        (p.home_score - m.predicted_home_runs) as home_error_delta,
        (p.away_score - m.predicted_away_runs) as away_error_delta,
        COALESCE(d.air_density, 1.225) as air_density,
        COALESCE(d.uv_modifier, 1.0) as uv_modifier,
        COALESCE(u.run_modifier, 1.0) as umpire_modifier,
        COALESCE(e.geomagnetic_kp, 2.0) as geomagnetic_kp,
        COALESCE(e.home_media_pressure, 0) as media_pressure,
        COALESCE(e.home_media_tone, 0.0) as media_tone
    FROM Post_Match_Analysis p
    INNER JOIN Model_Forecasts m ON p.game_pk = m.game_pk
    LEFT JOIN Daily_Lineups d ON p.game_pk = d.game_pk
    LEFT JOIN Daily_Umpires u ON p.game_pk = u.game_pk
    LEFT JOIN Esoteric_Signals e ON p.game_pk = e.game_pk
    WHERE p.home_score IS NOT NULL AND m.home_prob IS NOT NULL
    '''

    try:
        df = pd.read_sql_query(query, conn)
    except Exception as e:
        print(f"[CORRELATION ERROR] Failed to query join dataset: {e}")
        conn.close()
        return

    if len(df) < 25:
        print(f"[BYPASS] Insufficient sample size for correlation analysis (N = {len(df)} < 25).")
        conn.close()
        return

    print(f"Dataset compiled. Sweeping correlation matrix across {len(df)} empirical linescores...")

    df['total_abs_error'] = (df['home_error_delta'].abs() + df['away_error_delta'].abs())
    df['actual_total_runs'] = df['actual_home_runs'] + df['actual_away_runs']

    feature_cols = [
        'air_density', 'uv_modifier', 'umpire_modifier', 
        'geomagnetic_kp', 'media_pressure', 'media_tone'
    ]

    now_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    for feat in feature_cols:
        if df[feat].std() == 0:
            continue

        r_runs = float(df[feat].corr(df['actual_total_runs']))
        r_error = float(df[feat].corr(df['total_abs_error']))

        is_anomaly = 1 if (abs(r_runs) >= CORRELATION_SIGNIFICANCE_THRESHOLD or abs(r_error) >= CORRELATION_SIGNIFICANCE_THRESHOLD) else 0

        cursor.execute('''
        INSERT OR REPLACE INTO Feature_Correlations
        (feature_name, corr_with_total_runs, corr_with_model_error, sample_size, anomaly_flagged, last_updated)
        VALUES (?, ?, ?, ?, ?, ?)
        ''', (feat, round(r_runs, 4), round(r_error, 4), len(df), is_anomaly, now_str))

        flag_str = "🚨 [HIGH ANOMALY]" if is_anomaly else "   [STABLE]"
        print(f"{flag_str} {feat:<20} | r(Runs): {r_runs:+.3f} | r(Error): {r_error:+.3f}")

    conn.commit()
    conn.close()
    print("[SUCCESS] Correlation sweep complete without network latency.")

if __name__ == "__main__":
    run_correlation_engine()
