    FROM Post_Match_Analysis p
    INNER JOIN Model_Forecasts m ON p.game_pk = m.game_pk
    LEFT JOIN Daily_Lineups dl ON p.game_pk = dl.game_pk
    LEFT JOIN Daily_Umpires u ON p.game_pk = u.game_pk
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

    # Features evaluated against run discrepancy (Includes Air Density, UV Contrast, & Umpire Bias)
    features_to_test = {
        "Model_Predicted_Edge": df['predicted_edge'].values,
        "Projected_Total_Runs": df['total_pred_runs'].values,
        "Expected_Run_Differential": df['pred_run_diff'].values,
        "Actual_Blowout_Margin": df['actual_run_diff'].values,
        "Air_Density_Thermodynamics": df['air_density'].values,
        "UV_Visual_Contrast": df['uv_modifier'].values,
        "Umpire_Bias_Modifier": df['umpire_modifier'].values
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
    print("Correlation Engine execution complete. Discrepancy matrix updated with SOTA variables.")

if __name__ == "__main__":
    run_correlation_engine()
