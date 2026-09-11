import sqlite3
from datetime import datetime

PARK_FACTORS_TABLE = {
    "Colorado Rockies": 1.38,
    "Boston Red Sox": 1.09,
    "Cincinnati Reds": 1.08,
    "Kansas City Royals": 1.05,
    "Texas Rangers": 1.04,
    "Arizona Diamondbacks": 1.04,
    "Philadelphia Phillies": 1.03,
    "Washington Nationals": 1.02,
    "Atlanta Braves": 1.01,
    "Baltimore Orioles": 1.01,
    "Chicago Cubs": 1.01,
    "Los Angeles Angels": 1.00,
    "Milwaukee Brewers": 1.00,
    "Minnesota Twins": 1.00,
    "Toronto Blue Jays": 1.00,
    "Chicago White Sox": 0.99,
    "Houston Astros": 0.99,
    "Pittsburgh Pirates": 0.98,
    "St. Louis Cardinals": 0.98,
    "Detroit Tigers": 0.97,
    "New York Yankees": 0.97,
    "Cleveland Guardians": 0.96,
    "Miami Marlins": 0.95,
    "Oakland Athletics": 0.95,
    "San Francisco Giants": 0.95,
    "Tampa Bay Rays": 0.94,
    "New York Mets": 0.94,
    "Los Angeles Dodgers": 0.93,
    "San Diego Padres": 0.92,
    "Seattle Mariners": 0.91
}

def fetch_park_factors():
    """Populates Park_Factors table for all 30 MLB ballparks."""
    print("Synchronizing Official MLB Ballpark Run Factors...")
    conn = sqlite3.connect('mlb_engine.db', timeout=30)
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA busy_timeout=10000;")

    cursor.executescript('''
    CREATE TABLE IF NOT EXISTS Park_Factors (
        home_team TEXT PRIMARY KEY,
        run_factor REAL
    );
    ''')

    for team, factor in PARK_FACTORS_TABLE.items():
        cursor.execute('''
        INSERT OR REPLACE INTO Park_Factors (home_team, run_factor)
        VALUES (?, ?)
        ''', (team, factor))

    conn.commit()
    conn.close()
    print(f"Park Factors locked for all {len(PARK_FACTORS_TABLE)} MLB venues.")

# Aliases to prevent any possible import mismatch
update_park_factors = fetch_park_factors
populate_park_factors = fetch_park_factors

if __name__ == "__main__":
    fetch_park_factors()
