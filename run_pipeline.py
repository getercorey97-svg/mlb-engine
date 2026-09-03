import os
import sqlite3
from datetime import datetime
from post_match_analysis import run_post_match_analysis
from correlation_engine import run_correlation_engine
from ingest_stats import ingest_mlb_data
from biological_modifiers import execute_biological_pipeline
from alv_database import execute_unified_alv
from umpire_variance import execute_umpire_variance_pipeline
from statcast_metrics import execute_statcast_pipeline
from engine import run_ultimate_monte_carlo
from export_and_odds import export_forecasts_and_check_odds

def ensure_forecasts_table():
    conn = sqlite3.connect('mlb_engine.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Model_Forecasts (
            game_pk INTEGER PRIMARY KEY,
            home_team TEXT,
            away_team TEXT,
            home_prob REAL,
            away_prob REAL,
            predicted_edge REAL,
            predicted_home_runs REAL,
            predicted_away_runs REAL,
            timestamp TEXT
        )
    ''')
    for col in ["predicted_edge REAL", "predicted_home_runs REAL", "predicted_away_runs REAL"]:
        try:
            cursor.execute(f"ALTER TABLE Model_Forecasts ADD COLUMN {col}")
        except sqlite3.OperationalError:
            pass 
    conn.commit()
    conn.close()

def ensure_dynamic_modifiers_table():
    conn = sqlite3.connect('mlb_engine.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Dynamic_Modifiers (
            team_name TEXT PRIMARY KEY,
            offensive_modifier REAL DEFAULT 1.0,
            pitching_modifier REAL DEFAULT 1.0,
            last_updated TEXT
        )
    ''')
    cursor.execute("SELECT COUNT(*) FROM Dynamic_Modifiers")
    if cursor.fetchone()[0] == 0:
        teams = [
            "Arizona Diamondbacks", "Atlanta Braves", "Baltimore Orioles", "Boston Red Sox",
            "Chicago Cubs", "Chicago White Sox", "Cincinnati Reds", "Cleveland Guardians",
            "Colorado Rockies", "Detroit Tigers", "Houston Astros", "Kansas City Royals",
            "Los Angeles Angels", "Los Angeles Dodgers", "Miami Marlins", "Milwaukee Brewers",
            "Minnesota Twins", "New York Mets", "New York Yankees", "Oakland Athletics",
            "Philadelphia Phillies", "Pittsburgh Pirates", "San Diego Padres", "San Francisco Giants",
            "Seattle Mariners", "St. Louis Cardinals", "Tampa Bay Rays", "Texas Rangers",
            "Toronto Blue Jays", "Washington Nationals"
        ]
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for team in teams:
            cursor.execute('''
                INSERT INTO Dynamic_Modifiers (team_name, offensive_modifier, pitching_modifier, last_updated)
                VALUES (?, 1.0, 1.0, ?)
            ''', (team, current_time))
    conn.commit()
    conn.close()

def main():
    print(f"[{datetime.now()}] Starting GitHub Actions MLB Engine Pipeline...")
    
    # Step 0: Ensure database tables & ALV environmental columns exist first
    print("Initializing ALV Database & Schema Lock...")
    execute_unified_alv()
    ensure_forecasts_table()
    ensure_dynamic_modifiers_table()

    # Phase 1: Post-Match Analysis & Correlation Engine
    print("Executing Phase 1: Post-Match Analysis & Correlation Engine...")
    try:
        run_post_match_analysis()
        run_correlation_engine()
    except Exception as e:
        print(f"Post-match evolution error: {e}")

    # Phase 2: Ingest Base MLB Data
    print("Executing Phase 2: Ingesting Base MLB Data...")
    ingest_mlb_data()
    
    # Phase 3: Applying Advanced Environmental Context
    print("Executing Phase 3: Applying Advanced Environmental Context...")
    execute_biological_pipeline()
    execute_umpire_variance_pipeline()
    execute_statcast_pipeline()
    
    # Phase 4: Simulation and Export
    print("Executing Phase 4: Simulation and Export...")
    run_ultimate_monte_carlo()
    
    try:
        export_forecasts_and_check_odds()
    except Exception as e:
        print(f"Export Note: {e}")
    
    print("MLB Engine Pipeline execution completed successfully.")

if __name__ == "__main__":
    main()
