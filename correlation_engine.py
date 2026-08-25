import sqlite3
import numpy as np
import pandas as pd
from datetime import datetime

def calculate_pearson_r(x, y):
    """Calculates Pearson correlation coefficient safely without external scipy dependency."""
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
    return r, "Valid"

def run_correlation_engine():
    print("Initializing Feature Importance & Correlation Discovery Engine...")
    
    conn = sqlite3.connect('mlb_engine.db')
    cursor = conn.cursor()
    
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
    
    # Direct query without hard dependencies on external lookup tables
    query = '''
    SELECT 
        p.game_pk,
        p.actual_winner,
        p.home_score,
        p.away_score,
        p.model_correct,
        m.home_team,
        m.away_team,
        m.home_prob,
        m.away_prob,
        m.predicted_edge,
        m.predicted_home_runs,
        m.predicted_away_runs
    FROM Post_Match_Analysis p
    INNER JOIN Model_Forecasts m ON p.game_pk = m.game_pk
    WHERE m.predicted_home_runs IS NOT NULL AND m.predicted_away_runs IS NOT NULL
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

    # Derive factual target errors
    df['total_actual_runs'] = df['home_score'] + df['away_score']
    df['total_pred_runs'] = df['predicted_home_runs'] + df['predicted_away_runs']
    df['abs_run_error'] = np.abs(df['total_actual_runs'] - df['total_pred_runs'])
    df['actual_run_diff'] = np.abs(df['home_score'] - df['away_score'])
    df['pred_run_diff'] = np.abs(df['predicted_home_runs'] - df['predicted_away_runs'])

    # Features evaluated against run discrepancy
    features_to_test = {
        "Model_Predicted_Edge": df['predicted_edge'].values,
        "Projected_Total_Runs": df['total_pred_runs'].values,
        "Home_Win_Probability": df['home_prob'].values,
        "Away_Win_Probability": df['away_prob'].values,
        "Expected_Run_Differential": df['pred_run_diff'].values,
        "Actual_Blowout_Margin": df['actual_run_diff'].values
    }

    current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("-" * 65)
    print(f"{'Feature Analyzed':<28} | {'N':<5} | {'Pearson r':<10} | {'Flag'}")
    print("-" * 65)

    for feature_name, feature_values in features_to_test.items():
        r_val, status = calculate_pearson_r(feature_values, df['abs_run_error'].values)
        
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
    print("-" * 65)
    print("Correlation Engine execution complete. Discrepancy matrix updated.")

if __name__ == "__main__":
    run_correlation_engine()
