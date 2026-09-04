import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime

def calculate_pearson_r(x, y):
    """Calculates Pearson correlation coefficient safely."""
    x = np.array(x, dtype=float)
    y = np.array(y, dtype=float)
    
    if len(x) < 5:
        return 0.0, "Insufficient sample size (N < 5)"
    
    std_x = np.std(x)
    std_y = np.std(y)
    
    if std_x == 0 or std_y == 0:
        return 0.0, "Zero variance in feature distribution"
        
    corr_matrix = np.corrcoef(x, y)
    r = float(corr_matrix[0, 1])
    
    if np.isnan(r):
        return 0.0, "NaN Output"
        
    return r, "Valid"

def run_correlation_engine():
    print("Initializing Micro & Macro Feature Importance Engine (Including F5 Modules)...")
    
    conn = sqlite3.connect('mlb_engine.db', timeout=30)
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")
    
    # Create persistent correlation metrics table
    cursor.executescript('''
    CREATE TABLE IF NOT EXISTS Feature_Correlations (
        feature_name TEXT PRIMARY KEY,
        sample_size INTEGER,
        pearson_r REAL,
        status TEXT,
        anomaly_detected INTEGER,
        last_analyzed TEXT
    );
    ''')

    # Pulls core predictions, F5 forecasts, thermodynamics, and umpire variables
    query = '''
    SELECT 
        p.game_pk,
        p.home_score, p.away_score,
        p.home_f5_score, p.away_f5_score,
        m.home_prob, m.away_prob,
        m.predicted_edge, m.predicted_home_runs, m.predicted_away_runs,
        f.f5_exp_home_runs, f.f5_exp_away_runs, f.f5_home_prob, f.f5_total_runs,
        COALESCE(dl.air_density, 1.225) AS air_density,
        COALESCE(dl.uv_modifier, 1.0) AS uv_modifier,
        COALESCE(u.run_modifier, 1.0) AS umpire_modifier
    FROM Post_Match_Analysis p
    INNER JOIN Model_Forecasts m ON p.game_pk = m.game_pk
    LEFT JOIN F5_Forecasts f ON p.game_pk = f.game_pk
    LEFT JOIN Daily_Lineups dl ON p.game_pk = dl.game_pk
    LEFT JOIN Daily_Umpires u ON p.game_pk = u.game_pk
    WHERE m.predicted_home_runs IS NOT NULL AND p.home_f5_score IS NOT NULL
    '''
    
    try:
        df = pd.read_sql_query(query, conn)
    except Exception as e:
        print(f"Error querying dataset for correlation engine: {e}")
        conn.close()
        return

    if len(df) < 5:
        print(f"Post-mortem sample size ({len(df)} games) is too small. Minimum required: 5.")
        conn.close()
        return

    # Sanitize data to prevent NaN array errors
    df = df.fillna(0)

    # 1. Derive Full-Game Target Errors
    df['total_actual_runs'] = df['home_score'] + df['away_score']
    df['total_pred_runs'] = df['predicted_home_runs'] + df['predicted_away_runs']
    df['abs_run_error'] = np.abs(df['total_actual_runs'] - df['total_pred_runs'])
    df['actual_run_diff'] = np.abs(df['home_score'] - df['away_score'])
    df['pred_run_diff'] = np.abs(df['predicted_home_runs'] - df['predicted_away_runs'])

    # 2. Derive First 5 Innings (F5) Target Errors
    df['actual_f5_runs'] = df['home_f5_score'] + df['away_f5_score']
    df['abs_f5_run_error'] = np.abs(df['actual_f5_runs'] - df['f5_total_runs'])
    df['actual_f5_margin'] = df['home_f5_score'] - df['away_f5_score']
    df['pred_f5_margin'] = df['f5_exp_home_runs'] - df['f5_exp_away_runs']
    df['abs_f5_margin_error'] = np.abs(df['actual_f5_margin'] - df['pred_f5_margin'])

    # Map features to the specific error vectors they influence
    features_to_test = [
        # Full Game Engine Diagnostics
        ("Model_Predicted_Edge", df['predicted_edge'].values, df['abs_run_error'].values),
        ("Projected_Total_Runs", df['total_pred_runs'].values, df['abs_run_error'].values),
        ("Expected_Run_Differential", df['pred_run_diff'].values, df['abs_run_error'].values),
        ("Actual_Blowout_Margin", df['actual_run_diff'].values, df['abs_run_error'].values),
        
        # F5 Engine & Environmental Diagnostics (Removes Bullpen Variance)
        ("F5_Projected_Total_Runs", df['f5_total_runs'].values, df['abs_f5_run_error'].values),
        ("F5_Home_Probability", df['f5_home_prob'].values, df['abs_f5_margin_error'].values),
        ("Air_Density_Thermodynamics", df['air_density'].values, df['abs_f5_run_error'].values),
        ("UV_Visual_Contrast", df['uv_modifier'].values, df['abs_f5_run_error'].values),
        ("Umpire_Bias_Modifier", df['umpire_modifier'].values, df['abs_f5_run_error'].values)
    ]

    current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("-" * 75)
    print(f"{'Feature Analyzed':<28} | {'N':<5} | {'Pearson r':<10} | {'Flag'}")
    print("-" * 75)

    for feature_name, feature_values, target_error in features_to_test:
        r_val, status = calculate_pearson_r(feature_values, target_error)
        
        is_anomaly = 1 if abs(r_val) >= 0.35 and status == "Valid" else 0
        flag_text = "ANOMALY DETECTED" if is_anomaly else "Nominal"

        cursor.execute('''
            INSERT OR REPLACE INTO Feature_Correlations 
            (feature_name, sample_size, pearson_r, status, anomaly_detected, last_analyzed)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (feature_name, len(df), r_val, status, is_anomaly, current_timestamp))
        
        print(f"{feature_name:<28} | {len(df):<5} | {r_val:<+10.4f} | {flag_text}")

    conn.commit()
    conn.close()
    print("-" * 75)
    print("Correlation Engine execution complete. Discrepancy matrix updated with SOTA F5 variables.")

if __name__ == "__main__":
    run_correlation_engine()
