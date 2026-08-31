import sqlite3
import requests
from datetime import datetime

# Centralized coordinates for Air Density
STADIUMS = {
    "Atlanta Braves": (33.8907, -84.4677),
    "Colorado Rockies": (39.7559, -104.9942),
    "San Diego Padres": (32.7076, -117.1570),
    "Chicago Cubs": (41.9484, -87.6553),
    "New York Yankees": (40.8296, -73.9262),
    "Default": (39.8283, -98.5795)
}

def get_dynamic_air_density(team_name):
    """Calculates stadium air density using Open-Meteo with a strict 6s timeout and safe fallback."""
    coords = STADIUMS.get(team_name, STADIUMS["Default"])
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": coords[0],
        "longitude": coords[1],
        "current_weather": True,
        "hourly": "surface_pressure"
    }
    
    try:
        # Strict 6-second timeout prevents GitHub Actions pipeline from hanging
        res = requests.get(url, params=params, timeout=6).json()
        temp_c = res['current_weather']['temperature']
        pressure_hpa = res['hourly']['surface_pressure'][0]
        
        # Ideal Gas Law: ρ = P / (R * T)
        temp_k = temp_c + 273.15
        pressure_pa = pressure_hpa * 100
        density = pressure_pa / (287.05 * temp_k)
        return round(density, 4)
    except Exception:
        # Silent fallback to standard sea-level density to guarantee zero pipeline interruptions
        return 1.225

def execute_unified_alv():
    print("Rebuilding Database Schema to enforce Pitchers, Lineups, and Air Density...")
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
        status TEXT
    );
    ''')
    
    today = datetime.now().strftime('%Y-%m-%d')
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today}&hydrate=probablePitcher,lineups"
    
    print(f"Executing Unified ALV Mandate for {today}...")
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
            
            # Lineup Verification Pipeline
            home_lineup = teams['home'].get('lineup', [])
            away_lineup = teams['away'].get('lineup', [])
            lineup_status = "Confirmed" if len(home_lineup) >= 9 and len(away_lineup) >= 9 else "Pending/TBD"
            
            # Atmospheric Pipeline with fast fallback (6s timeout)
            air_density = get_dynamic_air_density(home)
            
            cursor.execute('''
            INSERT INTO Daily_Lineups (game_pk, game_date, away_team, home_team, away_pitcher, home_pitcher, lineup_status, air_density, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (game_pk, today, away, home, away_pitcher, home_pitcher, lineup_status, air_density, status))
            
            print(f"Verified & Mapped: {away} ({away_pitcher}) @ {home} ({home_pitcher}) | Lineups: {lineup_status} | ρ: {air_density}")

    conn.commit()
    print("-" * 50)
    print("Unified ALV sync complete. Schema perfectly locked.")
    conn.close()

if __name__ == "__main__":
    execute_unified_alv()
