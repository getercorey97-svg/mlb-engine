import sqlite3
from datetime import datetime
from post_match_analysis import run_post_match_analysis
from correlation_engine import run_correlation_engine
from ingest_stats import ingest_mlb_data
from alv_database import execute_unified_alv
from umpire_variance import execute_umpire_variance_pipeline
from statcast_metrics import execute_statcast_pipeline
from engine import run_ultimate_monte_carlo
from export_and_odds import export_forecasts_and_check_odds

def run_full_pipeline():
    print(f"[{datetime.now()}] Starting GitHub Actions MLB Engine Pipeline...")
    
    # Phase 1: Micro-Evolution (Learn from yesterday)
    try:
        run_post_match_analysis()
        run_correlation_engine()
    except Exception as e:
        print(f"Post-match evolution error: {e}")

    # Phase 2: Ingest fresh baseline statistics 
    ingest_mlb_data()
    
    # Phase 3: Build today's physics, weather, and lineup parameters
    execute_unified_alv()
    execute_umpire_variance_pipeline()
    execute_statcast_pipeline()
    
    # Phase 4: Execute 50,000 Iteration Forecast & Export
    run_ultimate_monte_carlo()
    
    try:
        export_forecasts_and_check_odds()
    except Exception as e:
        print(f"Export Note: {e}")
    
    print("MLB Engine Pipeline execution completed successfully.")

if __name__ == "__main__":
    run_full_pipeline()
