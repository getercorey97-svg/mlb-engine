import sqlite3
import requests

def fetch_ballpark_weather():
    print("Executing Extraction: Weather & Thermodynamics...")
    
    # Coordinates for MLB ballparks (Sample mapping for core venues)
    ballpark_coords = {
        "Colorado Rockies": (39.7559, -104.9942),
        "New York Yankees": (40.8296, -73.9262),
        "Chicago Cubs": (41.9484, -87.6553),
        "Los Angeles Dodgers": (34.0739, -118.2400),
        "Boston Red Sox": (42.3467, -71.0972)
    }

    conn = sqlite3.connect('mlb_engine.db')
    cursor = conn.cursor()
    
    cursor.executescript('''
    CREATE TABLE IF NOT EXISTS Ballpark_Weather (
        home_team TEXT PRIMARY KEY,
        weather_multiplier REAL
    );
    ''')

    for team, (lat, lon) in ballpark_coords.items():
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,wind_speed_10m"
        try:
            res = requests.get(url).json()
            temp_c = res['current']['temperature_2m']
            temp_f = (temp_c * 9/5) + 32
            wind_kph = res['current']['wind_speed_10m']
            
            # Thermodynamic adjustment: Higher temp = thinner air = higher run factor
            temp_factor = 1.0 + max(0, (temp_f - 70) * 0.002)
            
            cursor.execute('''
            INSERT OR REPLACE INTO Ballpark_Weather (home_team, weather_multiplier)
            VALUES (?, ?)
            ''', (team, temp_factor))
            
            print(f"Weather Logged | {team}: {temp_f:.1f}°F | Multiplier: {temp_factor:.3f}")
        except Exception:
            # Fallback neutral multiplier if network fails
            cursor.execute('INSERT OR REPLACE INTO Ballpark_Weather (home_team, weather_multiplier) VALUES (?, ?)', (team, 1.0))

    conn.commit()
    conn.close()
    print("Weather thermodynamics locked.")

if __name__ == "__main__":
    fetch_ballpark_weather()
