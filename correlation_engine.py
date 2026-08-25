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
    
    # Query completed post-match outcomes joined with pre-match forecasts and park factors
    query = '''
    SELECT 
        p.game_pk,
        p.actual_winner,
        p.home_score,
        p.away_score,
        p.model_correct,
        m.home_prob,
        m.away_prob,
        m.predicted_edge,
        m.predicted_home_runs,
        m.predicted_away_runs,
        COALESCE(pf.run_factor, 1.0) AS park_factor
    FROM Post_Match_Analysis p
    INNER JOIN Model_Forecasts m ON p.game_pk = m.game_pk
    LEFT JOIN Daily_Lineups dl ON p.game_pk = dl.game_pk
    LEFT JOIN Park_Factors pf ON dl.home_team = pf.home_team
    WHERE m.predicted_home_runs IS NOT NULL AND m.predicted_away_runs IS NOT NULL
    '''
    
    try:
        df = pd.read_sql_query(query, conn)
    except Exception as e:
        print(f"Error querying dataset for correlation engine: {e}")
        conn.close()
        return

    if len(df) < 5:
        print(f"Post-mortem sample size ({len(df)} games) is too small to calculate statistically valid correlations. Minimum required: 5.")
        conn.close()
        return

    # Derive factual target errors
    # 1. Total Absolute Run Error
    df['total_actual_runs'] = df['home_score'] + df['away_score']
    df['total_pred_runs'] = df['predicted_home_runs'] + df['predicted_away_runs']
    df['abs_run_error'] = np.abs(df['total_actual_runs'] - df['total_pred_runs'])
    
    # 2. Prediction Discrepancy (Binary 1 if incorrect, 0 if correct)
    df['model_miss'] = (df['model_correct'] == 0).astype(int)

    # Features to analyze against Absolute Run Error
    features_to_test = {
        "Park_Run_Factor": df['park_factor'].values,
        "Model_Predicted_Edge": df['predicted_edge'].values,
        "Projected_Total_Runs": df['total_pred_runs'].values,
        "Home_Win_Probability": df['home_prob'].values,
        "Expected_Run_Differential": np.abs(df['predicted_home_runs'].values - df['predicted_away_runs'].values)
    }

    current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print("-" * 60)
    print(f"{'Feature Analyzed':<30} | {'Sample':<6} | {'Pearson r':<10} | {'Flag'}")
    print("-" * 60)

    for feature_name, feature_values in features_to_test.items():
        r_val, status = calculate_pearson_r(feature_values, df['abs_run_error'].values)
        
        # Anomaly threshold: |r| >= 0.35 denotes a moderate-to-strong correlation with error
        is_anomaly = 1 if abs(r_val) >= 0.35 and status == "Valid" else 0
        flag_text = "ANOMALY DETECTED" if is_anomaly else "Nominal"

        cursor.execute('''
            INSERT OR REPLACE INTO Feature_Correlations 
            (feature_name, sample_size, pearson_r, status, anomaly_detected, last_analyzed)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (feature_name, len(df), r_val, status, is_anomaly, current_timestamp))
        
        print(f"{feature_name:<30} | {len(df):<6} | {r_val:<+10.4f} | {flag_text}")

    conn.commit()
    conn.close()
    print("-" * 60)
    print("Correlation Engine execution complete. Discrepancy matrix updated.")

if __name__ == "__main__":
    run_correlation_engine()
