import sqlite3

def apply_timestamp_migrations():
    conn = sqlite3.connect("mlb_engine.db")
    c = conn.cursor()

    tables_to_columns = {
        "Daily_Lineups": [
            ("game_datetime_utc", "TEXT"),
            ("game_time_et", "TEXT"),
            ("gatekeeper_trigger_utc", "TEXT"),
            ("ingested_at", "TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
        ],
        "Daily_Batters": [
            ("game_datetime_utc", "TEXT"),
            ("game_time_et", "TEXT")
        ],
        "Model_Forecasts": [
            ("game_date", "TEXT"),
            ("game_datetime_utc", "TEXT"),
            ("game_time_et", "TEXT"),
            ("forecast_timestamp_utc", "TEXT")
        ],
        "Historical_Forecasts": [
            ("game_date", "TEXT"),
            ("game_datetime_utc", "TEXT"),
            ("game_time_et", "TEXT"),
            ("forecast_timestamp_utc", "TEXT")
        ],
        "Batter_Hit_Forecasts": [
            ("game_date", "TEXT"),
            ("game_datetime_utc", "TEXT"),
            ("game_time_et", "TEXT"),
            ("forecast_timestamp_utc", "TEXT")
        ],
        "Pitcher_K_Forecasts": [
            ("game_date", "TEXT"),
            ("game_datetime_utc", "TEXT"),
            ("game_time_et", "TEXT"),
            ("forecast_timestamp_utc", "TEXT")
        ],
        "Batter_Post_Mortem_Logs": [
            ("game_date", "TEXT"),
            ("game_datetime_utc", "TEXT"),
            ("game_time_et", "TEXT"),
            ("outcome_timestamp_utc", "TEXT")
        ],
        "Pitcher_Post_Mortem_Logs": [
            ("game_date", "TEXT"),
            ("game_datetime_utc", "TEXT"),
            ("game_time_et", "TEXT"),
            ("outcome_timestamp_utc", "TEXT")
        ]
    }

    for table, columns in tables_to_columns.items():
        # Ensure table exists
        c.execute(f"CREATE TABLE IF NOT EXISTS {table} (id INTEGER PRIMARY KEY AUTOINCREMENT)")
        existing_cols = [r[1] for r in c.execute(f"PRAGMA table_info({table})").fetchall()]
        
        for col_name, col_type in columns:
            if col_name not in existing_cols:
                try:
                    c.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}")
                    print(f"[MIGRATION] Added {col_name} ({col_type}) to {table}.")
                except Exception as e:
                    print(f"[MIGRATION ERROR] {table}.{col_name}: {e}")

    conn.commit()
    conn.close()
    print("[MIGRATION COMPLETE] All operational tables updated with timestamp schemas.")

if __name__ == "__main__":
    apply_timestamp_migrations()
