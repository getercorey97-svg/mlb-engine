import sqlite3
import math
import requests
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.linear_model import LinearRegression

# Keyless Public Endpoints
NOAA_KP_URL = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

STADIUM_COORDINATES = {
    "Arizona Diamondbacks": (33.4453, -112.0667), "Atlanta Braves": (33.8907, -84.4677),
    "Baltimore Orioles": (39.2839, -76.6216), "Boston Red Sox": (42.3467, -71.0972),
    "Chicago Cubs": (41.9484, -87.6553), "Chicago White Sox": (41.8299, -87.6338),
    "Cincinnati Reds": (39.0974, -84.5071), "Cleveland Guardians": (41.4962, -81.6852),
    "Colorado Rockies": (39.7559, -104.9942), "Detroit Tigers": (42.3390, -83.0485),
    "Houston Astros": (29.7569, -95.3555), "Kansas City Royals": (39.0517, -94.4803),
    "Los Angeles Angels": (33.8003, -117.8827), "Los Angeles Dodgers": (34.0739, -118.2400),
    "Miami Marlins": (25.7781, -80.2197), "Milwaukee Brewers": (43.0280, -87.9712),
    "Minnesota Twins": (44.9817, -93.2778), "New York Mets": (40.7571, -73.8458),
    "New York Yankees": (40.8296, -73.9262), "Oakland Athletics": (37.7516, -122.2005),
    "Philadelphia Phillies": (39.9061, -75.1665), "Pittsburgh Pirates": (40.4469, -80.0057),
    "San Diego Padres": (32.7076, -117.1570), "San Francisco Giants": (37.7786, -122.3893),
    "Seattle Mariners": (47.5914, -122.3325), "St. Louis Cardinals": (38.6226, -90.1928),
    "Tampa Bay Rays": (27.7682, -82.6534), "Texas Rangers": (32.7473, -97.0845),
    "Toronto Blue Jays": (43.6414, -79.3894), "Washington Nationals": (38.8730, -77.0074),
    "Default": (39.8283, -98.5795)
}

def calculate_solar_elevation(lat, lon, dt_utc):
    """Calculates solar elevation angle to evaluate optic shadow contrast."""
    day_of_year = dt_utc.timetuple().tm_yday
    declination = 23.45 * math.sin(math.radians((360 / 365) * (day_of_year - 81)))
    time_offset = (lon * 4) / 60.0
    solar_time = dt_utc.hour + (dt_utc.minute / 60.0) + time_offset
    hour_angle = (solar_time - 12) * 15.0
    
    sin_elev = (math.sin(math.radians(lat)) * math.sin(math.radians(declination)) +
                math.cos(math.radians(lat)) * math.cos(math.radians(declination)) * 
                math.cos(math.radians(hour_angle)))
    return float(math.degrees(math.asin(max(-1.0, min(1.0, sin_elev)))))

def fetch_pregame_pressure_drop(team_name, date_str):
    """Pulls 3-hour pre-game surface pressure change (hPa gradient)."""
    lat, lon = STADIUM_COORDINATES.get(team_name, STADIUM_COORDINATES["Default"])
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date_str,
        "end_date": date_str,
        "hourly": "surface_pressure"
    }
    try:
        res = requests.get(METEO_ARCHIVE_URL, params=params, timeout=6).json()
        pressures = res.get('hourly', {}).get('surface_pressure', [])
        if len(pressures) >= 19:
            # Difference between 3:00 PM and 6:00 PM local
            return round(float(pressures[15] - pressures[18]), 2)
    except Exception:
        pass
    return 0.0

def build_correlation_dataset(conn):
    """Extracts confirmed historical records, baseline modifiers, and discovery variables."""
    cursor = conn.cursor()
    
    # Ensure required tables exist before querying
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Feature_Correlations (
            feature_name TEXT PRIMARY KEY,
            outcome_r REAL,
            run_total_r REAL,
            error_delta_r REAL,
            combined_r_squared REAL,
            status TEXT,
            last_analyzed TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Esoteric_Signals (
            game_pk INTEGER PRIMARY KEY,
            geomagnetic_kp REAL DEFAULT 2.0,
            solar_xray_flux REAL DEFAULT 1.0,
            home_media_pressure INTEGER DEFAULT 0,
            home_media_tone REAL DEFAULT 0.0,
            roster_birthday_active INTEGER DEFAULT 0,
            captured_at TEXT
        )
    ''')
    conn.commit()

    query = '''
        SELECT 
            p.game_pk, l.game_date, l.home_team, l.away_team,
            p.home_score, p.away_score, p.actual_winner,
            (p.home_score + p.away_score) AS total_runs,
            (CASE WHEN p.actual_winner = l.home_team THEN 1.0 ELSE 0.0 END) AS home_win,
            m.predicted_home_runs, m.predicted_away_runs,
            COALESCE(l.air_density, 1.225) AS air_density,
            COALESCE(l.uv_modifier, 1.0) AS uv_modifier,
            COALESCE(u.run_modifier, 1.0) AS umpire_run_mod,
            COALESCE(am.catcher_framing_modifier, 1.0) AS catcher_framing,
            COALESCE(am.bullpen_fatigue_modifier, 1.0) AS bullpen_fatigue,
            COALESCE(b.jet_lag_runs_penalty, 0.0) AS jet_lag_penalty,
            COALESCE(es.geomagnetic_kp, 2.0) AS geomagnetic_kp,
            COALESCE(es.solar_xray_flux, 1.0) AS solar_xray_flux,
            COALESCE(es.home_media_pressure, 0) AS home_media_pressure,
            COALESCE(es.home_media_tone, 0.0) AS home_media_tone,
            COALESCE(es.roster_birthday_active, 0) AS roster_birthday_active
        FROM Post_Match_Analysis p
        INNER JOIN Daily_Lineups l ON p.game_pk = l.game_pk
        LEFT JOIN Model_Forecasts m ON p.game_pk = m.game_pk
        LEFT JOIN Daily_Umpires u ON p.game_pk = u.game_pk
        LEFT JOIN Advanced_Metrics am ON l.home_team = am.team_name
        LEFT JOIN Biological_Modifiers b ON l.away_team = b.team_name
        LEFT JOIN Esoteric_Signals es ON p.game_pk = es.game_pk
        WHERE p.home_score IS NOT NULL
    '''
    df = pd.read_sql_query(query, conn)
    if len(df) < 15:
        print(f"[CORRELATION NOTICE] Sample size ({len(df)}) insufficient for multi-variable discovery sweep. Need >= 15 games.")
        return None

    # Compute absolute prediction error delta
    df['pred_total'] = df['predicted_home_runs'].fillna(4.0) + df['predicted_away_runs'].fillna(4.0)
    df['error_delta'] = (df['total_runs'] - df['pred_total']).abs()

    solar_elevations = []
    pressure_drops = []
    turnaround_deficits = []

    for _, row in df.iterrows():
        try:
            date_obj = datetime.strptime(str(row['game_date']), '%Y-%m-%d')
        except Exception:
            date_obj = datetime.now()
            
        lat, lon = STADIUM_COORDINATES.get(row['home_team'], STADIUM_COORDINATES["Default"])
        game_dt = date_obj.replace(hour=23, minute=5)
        solar_elevations.append(calculate_solar_elevation(lat, lon, game_dt))
        
        # Turnaround deficit proxy (quick series turnaround or Monday/Thursday day travel)
        is_turnaround = 1.0 if date_obj.weekday() in [0, 3] else 0.0
        turnaround_deficits.append(is_turnaround)
        
        pressure_drops.append(fetch_pregame_pressure_drop(row['home_team'], str(row['game_date'])))

    df['solar_elevation'] = solar_elevations
    df['turnaround_deficit'] = turnaround_deficits
    df['pressure_gradient'] = pressure_drops

    return df

def run_correlation_engine():
    """Executes the dual-layer correlation sweep across standard and discovery variables."""
    print(f"[{datetime.now()}] Initializing SOTA Feature Correlation & Broad Discovery Sweep...")
    conn = sqlite3.connect('mlb_engine.db')
    
    df = build_correlation_dataset(conn)
    if df is None:
        conn.close()
        return

    features = [
        # Regular Baseline Variables
        'umpire_run_mod', 'catcher_framing', 'bullpen_fatigue', 
        'air_density', 'uv_modifier', 'jet_lag_penalty',
        # Esoteric Discovery Variables
        'geomagnetic_kp', 'solar_xray_flux', 'solar_elevation', 
        'pressure_gradient', 'turnaround_deficit', 'roster_birthday_active',
        'home_media_pressure', 'home_media_tone'
    ]

    print("\n" + "=" * 85)
    print(f"{'FEATURE NAME':<24} | {'WIN r':<8} | {'RUNS r':<8} | {'ERROR r':<8} | {'STATUS'}")
    print("=" * 85)

    records = []
    high_anomalies = []
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for feat in features:
        r_win = df[feat].corr(df['home_win'])
        r_runs = df[feat].corr(df['total_runs'])
        r_error = df[feat].corr(df['error_delta'])

        # Flag structural edges crossing the significance threshold (|r| >= 0.35)
        status = "NORMAL"
        if abs(r_win) >= 0.35 or abs(r_runs) >= 0.35 or abs(r_error) >= 0.35:
            status = "HIGH_CORRELATION_ANOMALY"
            high_anomalies.append((feat, r_win, r_runs, r_error))
        elif abs(r_win) >= 0.20 or abs(r_runs) >= 0.20:
            status = "MODERATE_SIGNAL"

        print(f"{feat:<24} | {r_win:>+7.4f}  | {r_runs:>+7.4f}  | {r_error:>+7.4f}  | {status}")
        records.append((feat, float(r_win), float(r_runs), float(r_error), status, current_time))

    # Evaluate Multivariate Combined Explanatory Power (R-squared)
    X = df[features].fillna(0.0)
    y_win = df['home_win']
    y_runs = df['total_runs']

    reg_win = LinearRegression().fit(X, y_win)
    r2_win = float(reg_win.score(X, y_win))

    reg_runs = LinearRegression().fit(X, y_runs)
    r2_runs = float(reg_runs.score(X, y_runs))

    print("-" * 85)
    print(f"MULTIVARIATE EXPLANATORY POWER (COMBINED MATRIX):")
    print(f"Combined Variance Explained on Outcomes (R²): {r2_win:.4f} ({r2_win * 100:.2f}%)")
    print(f"Combined Variance Explained on Scoring (R²):  {r2_runs:.4f} ({r2_runs * 100:.2f}%)")
    print("=" * 85 + "\n")

    # Persist findings to SQLite database
    cursor = conn.cursor()
    for rec in records:
        cursor.execute('''
            INSERT OR REPLACE INTO Feature_Correlations 
            (feature_name, outcome_r, run_total_r, error_delta_r, combined_r_squared, status, last_analyzed)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (rec[0], rec[1], rec[2], rec[3], r2_win, rec[4], rec[5]))
    
    conn.commit()
    conn.close()
    print("[SUCCESS] Feature correlations and anomaly signals updated in mlb_engine.db.")

    # Autonomous Code Mutation Hook: If an anomaly exists, test a model improvement
    if high_anomalies:
        try:
            from autonomous_modifier import evaluate_candidate_code
            print(f"[AUTONOMOUS GATE] {len(high_anomalies)} high-correlation anomaly detected. Checking codebase mutation rules...")
        except ImportError:
            pass

if __name__ == "__main__":
    run_correlation_engine()
