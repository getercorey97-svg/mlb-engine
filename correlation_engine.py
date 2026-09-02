import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime

def calculate_pearson_r(x, y):
    """Calculates Pearson correlation coefficient safely."""
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    if len(x) < 5: return 0.0, "Insufficient sample size (N < 5)"
    std_x, std_y = np.std(x), np.std(y)
    if std_x == 0 or std_y == 0: return 0.0, "Zero variance in feature distribution"
    corr_matrix = np.corrcoef(x, y)
    return float(corr_matrix[0, 1]), "Valid"

def run_correlation_engine():
    print("Initializing Micro & Macro Feature Importance Engine...")
    conn = sqlite3.connect('mlb_engine.db')
    cursor = conn.cursor()
    
    cursor.executescript('''
    CREATE TABLE IF NOT EXISTS Feature_Correlations (
        feature_name TEXT PRIMARY KEY, sample_size INTEGER, pearson_r REAL, status TEXT, anomaly_detected INTEGER, last_analyzed TEXT
    );
    ''')

    for col in ["uv_modifier REAL DEFAULT 1.0", "air_density REAL DEFAULT 1.225"]:
        try: cursor.execute(f"ALTER TABLE Daily_Lineups ADD COLUMN {col}")
        except sqlite3.OperationalError: pass
    
    query = '''
    SELECT p.game_pk, p.home_score, p.away_score, m.home_prob, m.away_prob, m.predicted_edge, m.predicted_home_runs, m.predicted_away_runs, 
           COALESCE(dl.air_density, 1.225) AS air_density, COALESCE(dl.uv_modifier, 1.0) AS uv_modifier, COALESCE(u.run_modifier, 1.0) AS umpire_modifier
    FROM Post_Match_Analysis p
    INNER JOIN Model_Forecasts m ON p.game_pk = m.game_pk
    LEFT JOIN Daily_Lineups dl ON p.game_pk = dl.game_pk
    LEFT JOIN Daily_Umpires u ON p.game_pk = u.game_pk
    WHERE m.predicted_home_runs IS NOT NULL AND m.predicted_away_runs IS NOT NULL
    '''
    
    try: df = pd.read_sql_query(query, conn)
    except Exception as e: return print(f"Error querying dataset for correlation engine: {e}")

    if len(df) < 5: return print(f"Post-mortem sample size ({len(df)} games) is too small. Minimum required: 5.")

    df['total_actual_runs'] = df['home_score'] + df['away_score']
    df['total_pred_runs'] = df['predicted_home_runs'] + df['predicted_away_runs']
    df['abs_run_error'] = np.abs(df['total_actual_runs'] - df['total_pred_runs'])
    df['actual_run_diff'] = np.abs(df['home_score'] - df['away_score'])
    df['pred_run_diff'] = np.abs(df['predicted_home_runs'] - df['predicted_away_runs'])

    features_to_test = {
        "Model_Predicted_Edge": df['predicted_edge'].values, "Projected_Total_Runs": df['total_pred_runs'].values,
        "Expected_Run_Differential": df['pred_run_diff'].values, "Actual_Blowout_Margin": df['actual_run_diff'].values,
        "Air_Density_Thermodynamics": df['air_density'].values, "UV_Visual_Contrast": df['uv_modifier'].values, "Umpire_Bias_Modifier": df['umpire_modifier'].values
    }

    current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("-" * 65)
    print(f"{'Feature Analyzed':<28} | {'N':<5} | {'Pearson r':<10} | {'Flag'}")
    print("-" * 65)

    for feature_name, feature_values in features_to_test.items():
        r_val, status = calculate_pearson_r(feature_values, df['abs_run_error'].values)
        is_anomaly = 1 if abs(r_val) >= 0.35 and status == "Valid" else 0
        flag_text = "ANOMALY DETECTED" if is_anomaly else "Nominal"

        cursor.execute('''INSERT OR REPLACE INTO Feature_Correlations (feature_name, sample_size, pearson_r, status, anomaly_detected, last_analyzed) VALUES (?, ?, ?, ?, ?, ?)''', (feature_name, len(df), r_val, status, is_anomaly, current_timestamp))
        print(f"{feature_name:<28} | {len(df):<5} | {r_val:<+10.4f} | {flag_text}")

    conn.commit()
    conn.close()
    print("-" * 65)

if __name__ == "__main__":
    run_correlation_engine()
