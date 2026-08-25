import os
import sqlite3
from datetime import datetime
from alv_database import execute_unified_alv
from engine import run_ultimate_monte_carlo
from export_and_odds import export_forecasts_and_check_odds
from post_match_analysis import run_post_match_analysis

def ensure_forecasts_table():
    """Guarantees the Model_Forecasts table and all required columns exist."""
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
            timestamp TEXT
        )
    ''')
    try:
        cursor.execute('ALTER TABLE Model_Forecasts ADD COLUMN predicted_edge REAL;')
    except sqlite3.OperationalError:
        pass 
    
    conn.commit()
    conn.close()

def ensure_dynamic_modifiers_table():
    """Establishes the tracking matrix for the Automated Feedback Loop."""
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
    
    print("Executing Phase 1: Post-Match Analysis...")
    try:
        run_post_match_analysis()
    except Exception as e:
        print(f"Post-match analysis note: {e}")

    print("Executing Phase 2: Daily Slate, Simulation, and Export...")
    execute_unified_alv()
    ensure_forecasts_table()
    ensure_dynamic_modifiers_table()
    
    run_ultimate_monte_carlo()
    export_forecasts_and_check_odds()
    
    print("Pipeline execution complete.")

if __name__ == "__main__":
    main()
