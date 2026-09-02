import sqlite3
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from datetime import datetime

# Centralized coordinates for Air Density (All 30 MLB Stadiums)
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
    "Toronto Blue Jays": (43.6414, -79.3894), "Washington Nationals": (38.8730, -77.0074),
    "Default": (39.8283, -98.5795)
}

def get_robust_session():
    """Builds a requests session that auto-retries on timeouts or server drops."""
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    return session

def get_dynamic_atmosphere(team_name, session):
    """Calculates stadium air density and UV contrast using Open-Meteo."""
    coords = STADIUMS.get(team_name, STADIUMS["Default"])
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": coords[0],
        "longitude": coords[1],
        "current_weather": True,
        "hourly": "surface_pressure,cloud_cover"
    }
    
    try:
        res = session.get(url, params=params, timeout=10).json()
        temp_c = res['current_weather']['temperature']
        pressure_hpa = res['hourly']['surface_pressure'][0]
        cloud_cover = res['hourly']['cloud_cover'][0]
        
        # Corrected Ideal Gas Law: ρ = P / (R * T)
        temp_k = temp_c + 273.15
        pressure_pa = pressure_hpa * 100
        density = round(pressure_pa / (287.05 * temp_k), 4)
        
        # UV Contrast Logic
        uv_modifier = 1.03 if cloud_cover > 70 else 1.00
        
        return density, uv_modifier
    except Exception as e:
        print(f"Atmospheric Pipeline fell back to standard due to API lag: {e}")
        return 1.225, 1.00

def execute_unified_alv():
    print("Rebuilding Database Schema to enforce Pitchers, Lineups, and Air Density...")
    conn = sqlite3.connect('mlb_engine.db')
    cursor = conn.cursor()
    
    cursor.executescript('''
    DROP TABLE IF EXISTS Daily_Lineups;
    CREATE TABLE Daily_Lineups (
        game_pk INTEGER PRIMARY KEY, game_date TEXT, away_team TEXT, home_team TEXT,
        away_pitcher TEXT, home_pitcher TEXT, lineup_status TEXT, air_density REAL, uv_modifier REAL, status TEXT
    );
    ''')
    
    today = datetime.now().strftime('%Y-%m-%d')
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today}&hydrate=probablePitcher,lineups"
    
    session = get_robust_session()
    print(f"Executing Unified ALV Mandate for {today}...")
    
    try:
        response = session.get(url, timeout=15).json()
    except Exception as e:
        print(f"API Error fetching schedule: {e}")
        conn.close()
        return

    for date_data in response.get('dates', []):
        for game in date_data.get('games', []):
            game_pk = game['gamePk']
            status = game['status']['abstractGameState']
            teams = game.get('teams', {})
            
            away = teams['away']['team']['name']
            home = teams['home']['team']['name']
            
            away_pitcher = teams.get('away', {}).get('probablePitcher', {}).get('fullName', 'TBD')
            home_pitcher = teams.get('home', {}).get('probablePitcher', {}).get('fullName', 'TBD')
            
            home_lineup = teams['home'].get('lineup', [])
            away_lineup = teams['away'].get('lineup', [])
            lineup_status = "Confirmed" if len(home_lineup) >= 9 and len(away_lineup) >= 9 else "Pending/TBD"
            
            air_density, uv_modifier = get_dynamic_atmosphere(home, session)
            
            cursor.execute('''
            INSERT INTO Daily_Lineups (game_pk, game_date, away_team, home_team, away_pitcher, home_pitcher, lineup_status, air_density, uv_modifier, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (game_pk, today, away, home, away_pitcher, home_pitcher, lineup_status, air_density, uv_modifier, status))
            
            print(f"Verified & Mapped: {away} @ {home} | ρ: {air_density} | UV: {uv_modifier}")

    conn.commit()
    print("-" * 50)
    print("Unified ALV sync complete. Schema perfectly locked.")
    conn.close()

if __name__ == "__main__":
    execute_unified_alv()
