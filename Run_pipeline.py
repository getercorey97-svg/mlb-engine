import os
import sqlite3
from datetime import datetime
from alv_database import execute_unified_alv
from engine import run_ultimate_monte_carlo
from export_and_odds import export_forecasts_and_check_odds
from post_match_analysis import run_post_match_analysis

def ensure_forecasts_table():
    """Guarantees the Model_Forecasts table exists before simulation runs."""
    conn = sqlite3.connect('mlb_engine.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Model_Forecasts (
            game_pk INTEGER PRIMARY KEY,
            home_team TEXT,
            away_team TEXT,
            home_prob REAL,
            away_prob REAL,
            timestamp TEXT
        )
    ''')
    conn.commit()
    conn.close()

def main():
    print(f"[{datetime.now()}] Starting GitHub Actions MLB Engine Pipeline...")
    
    # 1. Run post-mortem check for prior games
    print("Executing Phase 1: Post-Match Analysis...")
    try:
        run_post_match_analysis()
    except Exception as e:
        print(f"Post-match analysis note: {e}")

    # 2. Sync ALV schedule and explicitly ensure schema tables exist
    print("Executing Phase 2: Daily Slate, Simulation, and Export...")
    execute_unified_alv()
    ensure_forecasts_table()
    
    # 3. Run Monte Carlo simulations and export CSV
    run_ultimate_monte_carlo()
    export_forecasts_and_check_odds()
    
    print("Pipeline execution complete.")

if __name__ == "__main__":
    main()
