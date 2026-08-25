import sqlite3
from datetime import datetime
from ingest_stats import ingest_mlb_data
from engine import run_ultimate_monte_carlo

def run_full_pipeline():
    print(f"[{datetime.now()}] Starting GitHub Actions MLB Engine Pipeline...")
    
    # Step 1: Ingest fresh data to prevent fallback constants and default clustering
    ingest_mlb_data()
    
    # Step 2: Execute Ultimate Monte Carlo Simulation across daily matchups
    run_ultimate_monte_carlo()
    
    print("MLB Engine Pipeline execution completed successfully.")

if __name__ == "__main__":
    run_full_pipeline()
