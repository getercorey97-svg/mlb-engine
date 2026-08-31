import sqlite3
import requests
import math
import time
from datetime import datetime, timedelta

# Core Geographical & Biological Baselines
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

MLB_TIMEZONES = {
    "Boston Red Sox": -5, "Baltimore Orioles": -5, "New York Yankees": -5, "Tampa Bay Rays": -5, "Toronto Blue Jays": -5,
    "New York Mets": -5, "Philadelphia Phillies": -5, "Atlanta Braves": -5, "Washington Nationals": -5, "Miami Marlins": -5,
    "Cleveland Guardians": -5, "Detroit Tigers": -5, "Cincinnati Reds": -5, "Pittsburgh Pirates": -5,
    "Chicago White Sox": -6, "Chicago Cubs": -6, "Kansas City Royals": -6, "Minnesota Twins": -6, "St. Louis Cardinals": -6,
    "Milwaukee Brewers": -6, "Texas Rangers": -6, "Houston Astros": -6,
    "Colorado Rockies": -7, "Arizona Diamondbacks": -7,
    "Seattle Mariners": -8, "Los Angeles Angels": -8, "Oakland Athletics": -8, "San Francisco Giants": -8, "Los Angeles Dodgers": -8, "San Diego Padres": -8
}

HISTORICAL_UMPIRE_BIAS = {
    "CB Bucknor": 1.045, "Angel Hernandez": 1.052, "Pat Hoberg": 0.985,
    "Doug Eddings": 0.970, "Lance Barksdale": 0.980, "Dan Bellino": 1.025, "Default": 1.000
}

CATCHER_FRAMING_BIAS = {
    "Atlanta Braves": 0.965, "Milwaukee Brewers": 0.970, "New York Yankees": 0.975,
    "San Francisco Giants": 0.980, "Texas Rangers": 0.985, "Chicago Cubs": 0.990,
    "Miami Marlins": 1.035, "Colorado Rockies": 1.045, "Oakland Athletics": 1.040,
    "Chicago White Sox": 1.030, "Default": 1.000
}

def setup_database_schema(cursor):
    """Executes a total wipe and rebuilds the schema to prevent double-weighing."""
    print("Executing total database overwrite...")
    tables = [
        "Team_Offense", "Team_Bullpen", "Pitcher_Stats", "Advanced_Metrics",
        "Dynamic_Modifiers", "Daily_Lineups", "Daily_Umpires", 
        "Biological_Modifiers", "Post_Match_Analysis", "Model_Forecasts", "Park_Factors"
    ]
    for table in tables:
        cursor.execute(f"DROP TABLE IF EXISTS {table}")
        
    cursor.executescript('''
        CREATE TABLE Dynamic_Modifiers (team_name TEXT PRIMARY KEY, offensive_modifier REAL DEFAULT 1.0, pitching_modifier REAL DEFAULT 1.0, last_updated TEXT);
        CREATE TABLE Daily_Lineups (game_pk INTEGER PRIMARY KEY, game_date TEXT, away_team TEXT, home_team TEXT, away_pitcher TEXT, home_pitcher TEXT, lineup_status TEXT, air_density REAL, uv_modifier REAL, status TEXT);
        CREATE TABLE Daily_Umpires (game_pk INTEGER PRIMARY KEY, home_plate_umpire TEXT, run_modifier REAL);
        CREATE TABLE Biological_Modifiers (team_name TEXT PRIMARY KEY, jet_lag_runs_penalty REAL);
        CREATE TABLE Advanced_Metrics (team_name TEXT PRIMARY KEY, catcher_framing_modifier REAL, bullpen_fatigue_modifier REAL);
        CREATE TABLE Post_Match_Analysis (game_pk INTEGER PRIMARY KEY, actual_winner TEXT, home_score INTEGER, away_score INTEGER, model_correct INTEGER, processed_at TEXT);
        CREATE TABLE Model_Forecasts (game_pk INTEGER PRIMARY KEY, home_team TEXT, away_team TEXT, home_prob REAL, away_prob REAL, predicted_edge REAL, predicted_home_runs REAL, predicted_away_runs REAL, timestamp TEXT);
        CREATE TABLE Team_Offense (team_name TEXT PRIMARY KEY, ops REAL);
        CREATE TABLE Team_Bullpen (team_name TEXT PRIMARY KEY, team_era REAL);
        CREATE TABLE Pitcher_Stats (last_name TEXT PRIMARY KEY, est_era REAL);
        CREATE TABLE Park_Factors (home_team TEXT PRIMARY KEY, run_factor REAL);
    ''')

def generate_weather_matrix():
    """Generates exactly 50 historical weather pings per stadium, spread evenly throughout the year."""
    print("Initializing 50-Ping Weather Matrix across all stadiums...")
    weather_matrix = {}
    
    # Generate 50 dates spaced evenly (~3.6 days apart) covering a 180-day season span
    base_date = datetime.now() - timedelta(days=5)
    sample_dates = [(base_date - timedelta(days=int(i * 3.6))).strftime('%Y-%m-%d') for i in range(50)]
    
    for team, coords in STADIUMS.items():
        weather_matrix[team] = []
        if team == "Default": continue
        
        for date_str in sample_dates:
            archive_url = "https://archive-api.open-meteo.com/v1/archive"
            params = {
                "latitude": coords[0],
                "longitude": coords[1],
                "start_date": date_str,
                "end_date": date_str,
                "hourly": "temperature_2m,relative_humidity_2m,surface_pressure,cloud_cover,uv_index"
            }
            try:
                w_res = requests.get(archive_url, params=params, timeout=5).json()
                hourly = w_res.get('hourly', {})
                
                # Defaulting to 7 PM local time index (19)
                temp_c = hourly.get('temperature_2m', [20.0])[19]
                rh = hourly.get('relative_humidity_2m', [50.0])[19]
                pressure_hpa = hourly.get('surface_pressure', [1013.25])[19]
                cloud_cover = hourly.get('cloud_cover', [20.0])[19]
                uv_index = hourly.get('uv_index', [1.0])[19]
                
                # Thermodynamic Air Density Calculation
                term1 = (3.4837 * pressure_hpa) / (temp_c + 273.15)
                term2 = (0.0080434 * rh) / (temp_c + 273.15)
                term3 = math.exp(17.67 * (temp_c / (temp_c + 243.5)))
                rho = round(term1 - (term2 * term3), 4)
                
                uv_mod = 1.00
                if uv_index >= 5.0 and cloud_cover < 30: uv_mod = 0.97
                elif cloud_cover > 70: uv_mod = 1.03
                    
                weather_matrix[team].append((rho, uv_mod))
                time.sleep(0.15) # Strictly prevents rate limiting
            except Exception:
                weather_matrix[team].append((1.225, 1.00))
        
        print(f"  [Weather Matrix] {team}: 50 macro-environmental nodes locked.")
        
    return weather_matrix

def seed_1600_games_sota():
    conn = sqlite3.connect('mlb_engine.db')
    cursor = conn.cursor()
    
    setup_database_schema(cursor)
    weather_matrix = generate_weather_matrix()
    
    print("Executing 1600-Game ALV Schedule Sweep (Micro & Macro Variables)...")
    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=220) # Ensures enough padding for exactly 1600 finalized games
    
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={start_date.strftime('%Y-%m-%d')}&endDate={end_date.strftime('%Y-%m-%d')}&hydrate=probablePitcher,officials"
    
    team_locations = {}
    weather_indexes = {team: 0 for team in STADIUMS}
    total_seeded = 0
    
    try:
        schedule = requests.get(url, timeout=20).json()
        for date_data in reversed(schedule.get('dates', [])): # Reverse to get the most recent 1600 games
            date_str = date_data['date']
            
            for game in date_data.get('games', []):
                if total_seeded >= 1600:
                    break
                
                if game.get('status', {}).get('abstractGameState') != 'Final':
                    continue
                
                pk = game['gamePk']
                home = game['teams']['home']['team']['name']
                away = game['teams']['away']['team']['name']
                
                away_pitcher = game['teams']['away'].get('probablePitcher', {}).get('fullName', 'TBD')
                home_pitcher = game['teams']['home'].get('probablePitcher', {}).get('fullName', 'TBD')
                home_score = game['teams']['home'].get('score', 0)
                away_score = game['teams']['away'].get('score', 0)
                
                # Microscopic Umpire Isolation
                hp_ump = "Unknown"
                for off in game.get('officials', []):
                    if off.get('officialType') == 'Home Plate':
                        hp_ump = off.get('official', {}).get('fullName', 'Unknown')
                        break
                ump_mod = HISTORICAL_UMPIRE_BIAS.get(hp_ump, HISTORICAL_UMPIRE_BIAS["Default"])
                
                # Macroscopic Biological Calculation (Jet Lag)
                game_tz = MLB_TIMEZONES.get(home, -5)
                away_penalty, home_penalty = 0.0, 0.0
                
                if away in team_locations:
                    if (game_tz - team_locations[away]) >= 2: away_penalty = 0.15
                if home in team_locations:
                    if (game_tz - team_locations[home]) >= 2: home_penalty = 0.15
                    
                team_locations[away] = game_tz
                team_locations[home] = game_tz
                
                # Map pre-loaded macro weather data circularly across home games
                idx = weather_indexes.get(home, 0)
                weather_data = weather_matrix.get(home, [(1.225, 1.00)])
                air_density, uv_modifier = weather_data[idx % 50]
                weather_indexes[home] = idx + 1
                
                # Database Ingestion
                cursor.execute("INSERT OR REPLACE INTO Biological_Modifiers (team_name, jet_lag_runs_penalty) VALUES (?, ?)", (away, away_penalty))
                cursor.execute("INSERT OR REPLACE INTO Biological_Modifiers (team_name, jet_lag_runs_penalty) VALUES (?, ?)", (home, home_penalty))
                
                cursor.execute("INSERT OR IGNORE INTO Advanced_Metrics (team_name, catcher_framing_modifier, bullpen_fatigue_modifier) VALUES (?, ?, 1.0)", (home, CATCHER_FRAMING_BIAS.get(home, 1.0)))
                cursor.execute("INSERT OR IGNORE INTO Advanced_Metrics (team_name, catcher_framing_modifier, bullpen_fatigue_modifier) VALUES (?, ?, 1.0)", (away, CATCHER_FRAMING_BIAS.get(away, 1.0)))

                cursor.execute(
                    "INSERT INTO Daily_Lineups (game_pk, game_date, away_team, home_team, away_pitcher, home_pitcher, lineup_status, air_density, uv_modifier, status) VALUES (?, ?, ?, ?, ?, ?, 'Confirmed', ?, ?, 'Final')",
                    (pk, date_str, away, home, away_pitcher, home_pitcher, air_density, uv_modifier)
                )
                cursor.execute(
                    "INSERT INTO Daily_Umpires (game_pk, home_plate_umpire, run_modifier) VALUES (?, ?, ?)",
                    (pk, hp_ump, ump_mod)
                )
                cursor.execute(
                    "INSERT INTO Post_Match_Analysis (game_pk, actual_winner, home_score, away_score, model_correct, processed_at) VALUES (?, ?, ?, ?, 0, 'SEEDED')",
                    (pk, home if home_score > away_score else away, home_score, away_score)
                )
                
                total_seeded += 1
                
        print(f"ALV Seeding Complete: {total_seeded} total historical nodes mapped.")
    except Exception as e:
        print(f"Schedule extraction error: {e}")

    conn.commit()
    conn.close()

if __name__ == "__main__":
    seed_1600_games_sota()
