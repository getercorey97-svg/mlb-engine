import os
import sqlite3
from datetime import datetime

# Phase 1: Post-Match & Evolution
from post_match_analysis import run_post_match_analysis
from correlation_engine import run_correlation_engine

# Phase 2: Ingestion & Base Sabermetrics
from ingest_stats import ingest_mlb_data
from park_factors import fetch_park_factors
from bullpen_fatigue import calculate_bullpen_fatigue

# Phase 3: Environmental Context & Absolute Live Verification
from alv_database import execute_unified_alv
from biological_modifiers import execute_biological_pipeline
from umpire_variance import execute_umpire_variance_pipeline
from statcast_metrics import execute_statcast_pipeline
from open_source_discovery import execute_discovery_ingestion

# Phase 4: Dual Engines & Betting Card Export
from engine import run_ultimate_monte_carlo
from engine_f5_props import run_f5_and_props_engine
from export_and_odds import export_forecasts_and_check_odds

def initialize_database_schemas():
    """Ensures all database tables exist before any queries or joins occur."""
    conn = sqlite3.connect('mlb_engine.db', timeout=30)
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA busy_timeout=10000;")
    
    cursor.executescript('''
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
        );
        CREATE TABLE IF NOT EXISTS F5_Forecasts (
            game_pk INTEGER PRIMARY KEY,
            away_team TEXT,
            home_team TEXT,
            away_starter TEXT,
            home_starter TEXT,
            f5_away_prob REAL,
            f5_home_prob REAL,
            f5_tie_prob REAL,
            f5_exp_away_runs REAL,
            f5_exp_home_runs REAL,
            f5_total_runs REAL
        );
        CREATE TABLE IF NOT EXISTS Dynamic_Modifiers (
            team_name TEXT PRIMARY KEY,
            offensive_modifier REAL DEFAULT 1.0,
            pitching_modifier REAL DEFAULT 1.0,
            last_updated TEXT
        );
        CREATE TABLE IF NOT EXISTS Daily_Lineups (
            game_pk INTEGER PRIMARY KEY,
            game_date TEXT,
            away_team TEXT,
            home_team TEXT,
            away_pitcher TEXT,
            home_pitcher TEXT,
            lineup_status TEXT,
            air_density REAL,
            uv_modifier REAL,
            status TEXT
        );
        CREATE TABLE IF NOT EXISTS Daily_Umpires (
            game_pk INTEGER PRIMARY KEY,
            home_plate_umpire TEXT,
            run_modifier REAL
        );
        CREATE TABLE IF NOT EXISTS Esoteric_Signals (
            game_pk INTEGER PRIMARY KEY,
            geomagnetic_kp REAL DEFAULT 2.0,
            solar_xray_flux REAL DEFAULT 1.0,
            home_media_pressure INTEGER DEFAULT 0,
            home_media_tone REAL DEFAULT 0.0,
            roster_birthday_active INTEGER DEFAULT 0,
            captured_at TEXT
        );
        CREATE TABLE IF NOT EXISTS Park_Factors (
            home_team TEXT PRIMARY KEY,
            run_factor REAL
        );
        CREATE TABLE IF NOT EXISTS Bullpen_Fatigue (
            team_name TEXT PRIMARY KEY,
            fatigue_multiplier REAL
        );
        CREATE TABLE IF NOT EXISTS Post_Match_Analysis (
            game_pk INTEGER PRIMARY KEY,
            actual_winner TEXT,
            home_score INTEGER,
            away_score INTEGER,
            home_f5_score INTEGER,
            away_f5_score INTEGER,
            model_correct INTEGER,
            processed_at TEXT
        );
    ''')
    
    # Pre-seed Dynamic Modifiers baseline if empty
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
                INSERT OR REPLACE INTO Dynamic_Modifiers (team_name, offensive_modifier, pitching_modifier, last_updated)
                VALUES (?, 1.0, 1.0, ?)
            ''', (team, current_time))

    conn.commit()
    conn.close()
    print("[INIT] Central SQLite ledger integrity locked with WAL mode.")

def main():
    print("=" * 65)
    print(f"[{datetime.now()}] Starting GitHub Actions MLB Prediction Pipeline...")
    print("=" * 65)
    
    # 0. Initialize Schema Locks
    initialize_database_schemas()

    # 1. Phase 1: Post-Match Analysis & Correlation Matrix
    print("\n--- PHASE 1: Post-Match Analysis & Matrix Sweeper ---")
    try:
        run_post_match_analysis()
    except Exception as e:
        print(f"[Phase 1 Warning] Post-match analysis note: {e}")

    try:
        run_correlation_engine()
    except Exception as e:
        print(f"[Phase 1 Warning] Correlation engine note: {e}")

    # 2. Phase 2: Ingest Base MLB Data, Park Factors & Bullpens
    print("\n--- PHASE 2: Ingestion, Park Factors & Bullpen Loads ---")
    try:
        ingest_mlb_data()
    except Exception as e:
        print(f"[Phase 2 Error] Stats ingestion error: {e}")

    try:
        fetch_park_factors()
    except Exception as e:
        print(f"[Phase 2 Warning] Park factors note: {e}")

    try:
        calculate_bullpen_fatigue()
    except Exception as e:
        print(f"[Phase 2 Warning] Bullpen fatigue note: {e}")

    # 3. Phase 3: Absolute Live Verification (ALV) & Environmental Context
    print("\n--- PHASE 3: ALV Thermodynamics & Biological Metrics ---")
    try:
        execute_unified_alv()
    except Exception as e:
        print(f"[Phase 3 Error] ALV Database error: {e}")

    try:
        execute_biological_pipeline()
    except Exception as e:
        print(f"[Phase 3 Warning] Biological pipeline note: {e}")

    try:
        execute_umpire_variance_pipeline()
    except Exception as e:
        print(f"[Phase 3 Warning] Umpire variance note: {e}")

    try:
        execute_statcast_pipeline()
    except Exception as e:
        print(f"[Phase 3 Warning] Statcast pipeline note: {e}")

    try:
        execute_discovery_ingestion()
    except Exception as e:
        print(f"[Phase 3 Warning] Discovery signals note: {e}")

    # 4. Phase 4: Dual-Engine Monte Carlo Simulations & Betting Cards
    print("\n--- PHASE 4: Dual-Engine Simulation & Export ---")
    try:
        run_ultimate_monte_carlo()
    except Exception as e:
        print(f"[Phase 4 Error] Monte Carlo engine error: {e}")

    try:
        run_f5_and_props_engine()
    except Exception as e:
        print(f"[Phase 4 Error] F5 engine error: {e}")

    try:
        export_forecasts_and_check_odds()
    except Exception as e:
        print(f"[Phase 4 Error] Export and odds error: {e}")

    print("\n" + "=" * 65)
    print(f"[{datetime.now()}] MLB Prediction Pipeline execution completed.")
    print("=" * 65)

if __name__ == "__main__":
    main()
