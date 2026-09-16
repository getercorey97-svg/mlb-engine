import sqlite3
from datetime import datetime

DEFAULT_PARK_FACTORS = {
    "Colorado Rockies": 1.38, "Boston Red Sox": 1.09, "Cincinnati Reds": 1.08,
    "Kansas City Royals": 1.05, "Texas Rangers": 1.04, "Arizona Diamondbacks": 1.04,
    "Philadelphia Phillies": 1.03, "Washington Nationals": 1.02, "Atlanta Braves": 1.01,
    "Baltimore Orioles": 1.01, "Chicago Cubs": 1.01, "Los Angeles Angels": 1.00,
    "Milwaukee Brewers": 1.00, "Minnesota Twins": 1.00, "Toronto Blue Jays": 1.00,
    "Chicago White Sox": 0.99, "Houston Astros": 0.99, "Pittsburgh Pirates": 0.98,
    "St. Louis Cardinals": 0.98, "Detroit Tigers": 0.97, "New York Yankees": 0.97,
    "Cleveland Guardians": 0.96, "Miami Marlins": 0.95, "Oakland Athletics": 0.95,
    "San Francisco Giants": 0.95, "Tampa Bay Rays": 0.94, "New York Mets": 0.94,
    "Los Angeles Dodgers": 0.93, "San Diego Padres": 0.92, "Seattle Mariners": 0.91
}

def initialize_database_schemas():
    """Guarantees every single table and column exists with zero schema mismatch."""
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
        CREATE TABLE IF NOT EXISTS Pitcher_Modifiers (
            pitcher_name TEXT PRIMARY KEY,
            k_modifier REAL DEFAULT 1.0,
            f5_run_modifier REAL DEFAULT 1.0,
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
            air_density REAL DEFAULT 1.225,
            uv_modifier REAL DEFAULT 1.00,
            status TEXT
        );
        CREATE TABLE IF NOT EXISTS Daily_Umpires (
            game_pk INTEGER PRIMARY KEY,
            home_plate_umpire TEXT,
            run_modifier REAL DEFAULT 1.00,
            umpire_locked INTEGER DEFAULT 0,
            updated_at TEXT
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
            run_factor REAL DEFAULT 1.00
        );
        CREATE TABLE IF NOT EXISTS Bullpen_Fatigue (
            team_name TEXT PRIMARY KEY,
            fatigue_multiplier REAL DEFAULT 1.00
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