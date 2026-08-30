import sqlite3
from datetime import datetime
from evolve_engine import execute_factual_post_mortem
from ingest_stats import ingest_mlb_data
from alv_database import execute_unified_alv
from engine import run_ultimate_monte_carlo

def run_full_pipeline():
    print(f"[{datetime.now()}] Starting GitHub Actions MLB Engine Pipeline...")
    
    # Phase 1: Factual Post-Mortem & Database Evolution
    execute_factual_post_mortem()
    
    # Phase 2: Ingest/Update baseline statistics
    ingest_mlb_data()
    
    # Phase 3: Execute ALV Mandate (Build today's Daily_Lineups)
    execute_unified_alv()
    
    # Phase 4: Execute Ultimate Monte Carlo Simulation on Live Slate
    run_ultimate_monte_carlo()
    
    print("MLB Engine Pipeline execution completed successfully.")

if __name__ == "__main__":
    run_full_pipeline()
