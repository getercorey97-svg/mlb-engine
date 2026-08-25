import os
from datetime import datetime
from alv_database import execute_unified_alv
from engine import run_ultimate_monte_carlo
from export_and_odds import export_forecasts_and_check_odds
from post_match_analysis import run_post_match_analysis

def main():
    print(f"[{datetime.now()}] Starting GitHub Actions MLB Engine Pipeline...")
    
    # 1. Run post-mortem check for prior games
    print("Executing Phase 1: Post-Match Analysis...")
    try:
        run_post_match_analysis()
    except Exception as e:
        print(f"Post-match analysis note: {e}")

    # 2. Fetch today's slate, run simulations, and export CSV
    print("Executing Phase 2: Daily Slate, Simulation, and Export...")
    execute_unified_alv()
    run_ultimate_monte_carlo()
    export_forecasts_and_check_odds()
    
    print("Pipeline execution complete.")

if __name__ == "__main__":
    main()
