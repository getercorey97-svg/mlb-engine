import sqlite3
import requests
import math
from datetime import datetime

STADIUMS = {
    "Atlanta Braves": (33.8907, -84.4677),
    "Colorado Rockies": (39.7559, -104.9942),
    "San Diego Padres": (32.7076, -117.1570),
    "Chicago Cubs": (41.9484, -87.6553),
    "New York Yankees": (40.8296, -73.9262),
    "Default": (39.8283, -98.5795)
}

def get_stadium_atmosphere(team_name):
    coords = STADIUMS.get(team_name, STADIUMS["Default"])
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": coords[0],
        "longitude": coords[1],
        "current": "temperature_2m,relative_humidity_2m,surface_pressure,cloud_cover,uv_index"
    }
    
    try:
        res = requests.get(url, params=params, timeout=6).json()
        current = res['current']
        temp_c = current['temperature_2m']
        rh = current['relative_humidity_2m']
        pressure_hpa = current['surface_pressure']
        cloud_cover = current['cloud_cover']
        uv_index = current['uv_index']
        
        term1 = (3.4837 * pressure_hpa) / (temp_c + 273.15)
        term2 = (0.0080434 * rh) / (temp_c + 273.15)
        term3 = math.exp(17.67 * (temp_c / (temp_c + 243.5)))
        rho = term1 - (term2 * term3)
        
        uv_modifier = 1.00
        if uv_index >= 5.0 and cloud_cover < 30:
            uv_modifier = 0.97 
        elif cloud_cover > 70:
            uv_modifier = 1.03 
            
        return round(rho, 4), uv_modifier
    except Exception:
        return 1.225, 1.00

def execute_unified_alv():
    print("Rebuilding Database Schema for Pitchers, Lineups, Thermodynamics, and UV Optics...")
    conn = sqlite3.connect('mlb_engine.db')
    cursor = conn.cursor()
    
    cursor.executescript('''
    DROP TABLE IF EXISTS Daily_Lineups;
    CREATE TABLE Daily_Lineups (
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
    ''')
    
    today = datetime.now().strftime('%Y-%m-%d')
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today}&hydrate=probablePitcher,lineups"
    
    try:
        response = requests.get(url, timeout=10).json()
    except Exception as e:
        print(f"API Error fetching schedule: {e}")
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
            
            air_density, uv_modifier = get_stadium_atmosphere(home)
            
            cursor.execute('''
            INSERT INTO Daily_Lineups (game_pk, game_date, away_team, home_team, away_pitcher, home_pitcher, lineup_status, air_density, uv_modifier, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (game_pk, today, away, home, away_pitcher, home_pitcher, lineup_status, air_density, uv_modifier, status))

    conn.commit()
    print("Unified ALV sync complete.")
    conn.close()

if __name__ == "__main__":
    execute_unified_alv()
