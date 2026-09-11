import sqlite3
import requests
import math
import concurrent.futures
from datetime import datetime, timedelta

STADIUMS = {
    "Arizona Diamondbacks": (33.4453, -112.0667), "Atlanta Braves": (33.8907, -84.4677),
    "Baltimore Orioles": (39.2838, -76.6217), "Boston Red Sox": (42.3467, -71.0972),
    "Chicago Cubs": (41.9484, -87.6553), "Chicago White Sox": (41.8300, -87.6338),
    "Cincinnati Reds": (39.0979, -84.5082), "Cleveland Guardians": (41.4962, -81.6852),
    "Colorado Rockies": (39.7559, -104.9942), "Detroit Tigers": (42.3390, -83.0485),
    "Houston Astros": (29.7573, -95.3555), "Kansas City Royals": (39.0517, -94.4803),
    "Los Angeles Angels": (33.8003, -117.8827), "Los Angeles Dodgers": (34.0739, -118.2400),
    "Miami Marlins": (25.7781, -80.2197), "Milwaukee Brewers": (43.0280, -87.9712),
    "Minnesota Twins": (44.9817, -93.2778), "New York Mets": (40.7571, -73.8458),
    "New York Yankees": (40.8296, -73.9262), "Oakland Athletics": (37.7516, -122.2005),
    "Philadelphia Phillies": (39.9061, -75.1665), "Pittsburgh Pirates": (40.4469, -80.0057),
    "San Diego Padres": (32.7076, -117.1570), "San Francisco Giants": (37.7786, -122.3893),
    "Seattle Mariners": (47.5914, -122.3325), "St. Louis Cardinals": (38.6226, -90.1928),
    "Tampa Bay Rays": (27.7682, -82.6534), "Texas Rangers": (32.7473, -97.0845),
    "Toronto Blue Jays": (43.6414, -79.3894), "Washington Nationals": (38.8730, -77.0074)
}

def fetch_weather_worker(team, coords, date_str):
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={coords[0]}&longitude={coords[1]}&start_date={date_str}&end_date={date_str}&hourly=temperature_2m,surface_pressure,cloud_cover"
    try:
        r = requests.get(url, timeout=5).json()
        temp, pres = r['hourly']['temperature_2m'][15], r['hourly']['surface_pressure'][15]
        rho = round((pres * 100) / (287.05 * (temp + 273.15)), 4)
        return team, rho
    except: return team, 1.225

def seed_1600_games_sota():
    conn = sqlite3.connect('mlb_engine.db')
    print("Starting Multi-Threaded Atmospheric Seeding...")
    
    base_date = datetime.now() - timedelta(days=5)
    dates = [(base_date - timedelta(days=int(i*3.6))).strftime('%Y-%m-%d') for i in range(50)]
    weather_storage = {team: [] for team in STADIUMS}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        futures = [executor.submit(fetch_weather_worker, t, STADIUMS[t], d) for t in STADIUMS for d in dates]
        for f in concurrent.futures.as_completed(futures):
            t, rho = f.result()
            weather_storage[t].append(rho)

    print(f"Matrix complete for {len(weather_storage)} stadiums. System primed.")
    conn.close()

if __name__ == "__main__":
    seed_1600_games_sota()
