import sqlite3
import requests
import numpy as np
import warnings
import concurrent.futures
from datetime import datetime, timedelta
from sklearn.isotonic import IsotonicRegression
from engine import run_ultimate_monte_carlo
from engine_f5_props import run_f5_and_props_engine

# Suppress sklearn warnings
warnings.filterwarnings("ignore", category=UserWarning)

STADIUMS = {
    "Arizona Diamondbacks": (33.4453, -112.0667), "Atlanta Braves": (33.8907, -84.4677),
    "Baltimore Orioles": (39.2839, -76.6216), "Boston Red Sox": (42.3467, -71.0972),
    "Chicago Cubs": (41.9484, -87.6553), "Chicago White Sox": (41.8299, -87.6338),
    "Cincinnati Reds": (39.0974, -84.5071), "Cleveland Guardians": (41.4962, -81.6852),
    "Colorado Rockies": (39.7559, -104.9942), "Detroit Tigers": (42.3390, -83.0485),
    "Houston Astros": (29.7569, -95.3555), "Kansas City Royals": (39.0517, -94.4803),
    "Los Angeles Angels": (33.8003, -117.8827), "Los Angeles Dodgers": (34.0739, -118.2400),
    "Miami Marlins": (25.7781, -80.2197), "Milwaukee Brewers": (43.0280, -87.9712),
    "Minnesota Twins": (44.9817, -93.2778), "New York Mets": (40.7571, -73.8458),
    "New York Yankees": (40.8296, -73.9262), "Oakland Athletics": (37.7516, -122.2005),
    "Philadelphia Phillies": (39.9061, -75.1665), "Pittsburgh Pirates": (40.4469, -80.0057),
    "San Diego Padres": (32.7076, -117.1570), "San Francisco Giants": (37.7786, -122.3893),
    "Seattle Mariners": (47.5914, -122.3325), "St. Louis Cardinals": (38.6226, -90.1928),
    "Tampa Bay Rays": (27.7682, -82.6534), "Texas Rangers": (32.7473, -97.0845),
    "Toronto Blue Jays": (43.6414, -79.3894), "Washington Nationals": (38.8730, -77.0074),
    "Default": (39.8283, -98.5795)
}

def get_historical_atmosphere(team_name, date_str):
    coords = STADIUMS.get(team_name, STADIUMS["Default"])
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={coords[0]}&longitude={coords[1]}&start_date={date_str}&end_date={date_str}&hourly=surface_pressure,temperature_2m,cloud_cover"
    try:
        res = requests.get(url, timeout=5).json()
        temp_c, pres, cloud = res['hourly']['temperature_2m'][12], res['hourly']['surface_pressure'][12], res['hourly']['cloud_cover'][12]
        rho = round((pres * 100) / (287.05 * (temp_c + 273.15)), 4)
        return rho, (1.03 if cloud > 70 else 1.00)
    except: return 1.225, 1.00

def run_backtest_engine():
    print(f"Initializing Incremental Backtest...")
    conn = sqlite3.connect('mlb_engine.db', timeout=30)
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")

    # 1. Load existing games to skip them
    try:
        cursor.execute("SELECT game_pk FROM Post_Match_Analysis")
        processed_pks = {row[0] for row in cursor.fetchall()}
    except: processed_pks = set()

    current_date = datetime.now() - timedelta(days=220)
    end_date = datetime.now()
    
    while current_date <= end_date:
        ds = current_date.strftime('%Y-%m-%d')
        current_date += timedelta(days=1)
        day_url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={ds}&gameType=R&hydrate=probablePitcher,linescore,officials"
        
        try:
            day_res = requests.get(day_url, timeout=10).json()
            for date_data in day_res.get('dates', []):
                for game in date_data.get('games', []):
                    pk = game['gamePk']
                    if pk in processed_pks or game['status']['abstractGameState'] != 'Final':
                        continue
                    
                    home_team, away_team = game['teams']['home']['team']['name'], game['teams']['away']['team']['name']
                    rho, uv = get_historical_atmosphere(home_team, ds)
                    
                    cursor.execute('''INSERT OR REPLACE INTO Daily_Lineups (game_pk, game_date, away_team, home_team, air_density, uv_modifier, status)
                                      VALUES (?, ?, ?, ?, ?, ?, ?)''', (pk, ds, away_team, home_team, rho, uv, 'Final'))
            conn.commit()
        except: continue
        
    print("Incremental Sync Complete. Triggering Engine Simulation...")
    run_ultimate_monte_carlo()
    run_f5_and_props_engine()
    conn.close()

if __name__ == "__main__":
    run_backtest_engine()
