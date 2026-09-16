import sqlite3
import requests
from datetime import datetime

# Accurate 2026 MLB Ballpark Coordinates 
MLB_COORDS = {
    "Arizona Diamondbacks": (33.4453, -112.0667),
    "Atlanta Braves": (33.8907, -84.4677),
    "Baltimore Orioles": (39.2840, -76.6199),
    "Boston Red Sox": (42.3467, -71.0972),
    "Chicago Cubs": (41.9484, -87.6553),
    "Chicago White Sox": (41.8299, -87.6338),
    "Cincinnati Reds": (39.0979, -84.5072),
    "Cleveland Guardians": (41.4962, -81.6852),
    "Colorado Rockies": (39.7559, -104.9942),
    "Detroit Tigers": (42.3390, -83.0485),
    "Houston Astros": (29.7573, -95.3555),
    "Kansas City Royals": (39.0517, -94.4803),
    "Los Angeles Angels": (33.8003, -117.8827),
    "Los Angeles Dodgers": (34.0739, -118.2400),
    "Miami Marlins": (25.7781, -80.2195),
    "Milwaukee Brewers": (43.0280, -87.9712),
    "Minnesota Twins": (44.9817, -93.2778),
    "New York Mets": (40.7571, -73.8458),
    "New York Yankees": (40.8296, -73.9262),
    "Oakland Athletics": (38.5804, -121.5056), # Sutter Health Park (Sacramento)
    "Athletics": (38.5804, -121.5056),         
    "Philadelphia Phillies": (39.9061, -75.1665),
    "Pittsburgh Pirates": (40.4469, -80.0057),
    "San Diego Padres": (32.7076, -117.1570),
    "San Francisco Giants": (37.7786, -122.3893),
    "Seattle Mariners": (47.5913, -122.3323),
    "St. Louis Cardinals": (38.6226, -90.1928),
    "Tampa Bay Rays": (27.7682, -82.6534),
    "Texas Rangers": (32.7373, -97.0844),
    "Toronto Blue Jays": (43.6414, -79.3894),
    "Washington Nationals": (38.8730, -77.0074)
}

def calculate_live_air_density(temp_c, pressure_hpa):
    """
    Calculates exact air density (Rho in kg/m^3) using the Ideal Gas Law.
    rho = P / (R * T)
    """
    pressure_pa = pressure_hpa * 100       # Convert hectopascals to Pascals
    temp_k = temp_c + 273.15               # Convert Celsius to Kelvin
    r_specific = 287.0500676               # Specific gas constant for dry air
    
    rho = pressure_pa / (r_specific * temp_k)
    return round(rho, 4)

def update_ballpark_thermodynamics():
    print("Executing Extraction: Live Weather & Thermodynamics Verification...")
    
    conn = sqlite3.connect('mlb_engine.db', timeout=30)
    cursor = conn.cursor()
    
    # Query only the active stadiums being played in today to save API calls
    cursor.execute("SELECT DISTINCT home_team, game_pk FROM Daily_Lineups WHERE status != 'Final'")
    active_games = cursor.fetchall()
    
    for home_team, game_pk in active_games:
        if home_team not in MLB_COORDS:
            print(f"[{home_team}] coordinates not found. Skipping live weather calculation.")
            continue
            
        lat, lon = MLB_COORDS[home_team]
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,surface_pressure,wind_speed_10m"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            res = response.json()
            
            temp_c = res['current']['temperature_2m']
            pressure = res['current']['surface_pressure']
            temp_f = (temp_c * 9/5) + 32
            
            # Calculate true live Air Density (Rho)
            live_rho = calculate_live_air_density(temp_c, pressure)
            
            # Update the Daily_Lineups table directly with live physics
            cursor.execute('''
            UPDATE Daily_Lineups 
            SET air_density = ? 
            WHERE game_pk = ?
            ''', (live_rho, game_pk))
            
            print(f"Weather Locked | {home_team} (Game {game_pk}): {temp_f:.1f}°F, Pressure: {pressure}hPa -> Live Rho: {live_rho} kg/m³")
            
        except Exception as e:
            print(f"Failed to fetch weather for {home_team}: {e}. Defaulting to baseline Rho.")

    conn.commit()
    conn.close()
    print("Live stadium thermodynamics securely injected into active slate.")

if __name__ == "__main__":
    update_ballpark_thermodynamics()
