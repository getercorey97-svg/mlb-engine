import sqlite3
import requests
from datetime import datetime, timedelta

MLB_TIMEZONES = {
    "Boston Red Sox": -5, "Baltimore Orioles": -5, "New York Yankees": -5, "Tampa Bay Rays": -5, "Toronto Blue Jays": -5,
    "New York Mets": -5, "Philadelphia Phillies": -5, "Atlanta Braves": -5, "Washington Nationals": -5, "Miami Marlins": -5,
    "Cleveland Guardians": -5, "Detroit Tigers": -5, "Cincinnati Reds": -5, "Pittsburgh Pirates": -5,
    "Chicago White Sox": -6, "Chicago Cubs": -6, "Kansas City Royals": -6, "Minnesota Twins": -6, "St. Louis Cardinals": -6,
    "Milwaukee Brewers": -6, "Texas Rangers": -6, "Houston Astros": -6,
    "Colorado Rockies": -7, "Arizona Diamondbacks": -7,
    "Seattle Mariners": -8, "Los Angeles Angels": -8, "Oakland Athletics": -8, "San Francisco Giants": -8, "Los Angeles Dodgers": -8, "San Diego Padres": -8
}

def execute_biological_pipeline():
    print("Initializing Circadian Biology & Jet Lag Pipeline...")
    today = datetime.now()
    yesterday = today - timedelta(days=1)
    
    today_str = today.strftime('%Y-%m-%d')
    yesterday_str = yesterday.strftime('%Y-%m-%d')
    
    conn = sqlite3.connect('mlb_engine.db')
    cursor = conn.cursor()
    
    cursor.executescript('''
    CREATE TABLE IF NOT EXISTS Biological_Modifiers (
        team_name TEXT PRIMARY KEY,
        jet_lag_runs_penalty REAL
    );
    ''')
    cursor.execute("DELETE FROM Biological_Modifiers")
    
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={yesterday_str}&endDate={today_str}"
    
    try:
        response = requests.get(url, timeout=15).json()
    except Exception as e:
        print(f"API Error fetching itineraries: {e}")
        return

    team_locations = {}
    
    for date_data in response.get('dates', []):
        date = date_data['date']
        for game in date_data.get('games', []):
            away = game['teams']['away']['team']['name']
            home = game['teams']['home']['team']['name']
            game_tz = MLB_TIMEZONES.get(home, -5)
            
            if date == yesterday_str:
                team_locations[away] = {'yesterday_tz': game_tz}
                team_locations[home] = {'yesterday_tz': game_tz}
            elif date == today_str:
                if away not in team_locations: team_locations[away] = {}
                if home not in team_locations: team_locations[home] = {}
                team_locations[away]['today_tz'] = game_tz
                team_locations[home]['today_tz'] = game_tz

    print("Analyzing Eastward Travel Penalties...")
    for team, travel in team_locations.items():
        penalty = 0.0
        yesterday_tz = travel.get('yesterday_tz')
        today_tz = travel.get('today_tz')
        
        if yesterday_tz is not None and today_tz is not None:
            time_zone_shift = today_tz - yesterday_tz 
            if time_zone_shift >= 2:
                penalty = 0.15 
                print(f"  [CIRCADIAN PENALTY] {team} traveled East across {time_zone_shift} zones. (+0.15 Runs Allowed)")
        
        cursor.execute('''
            INSERT INTO Biological_Modifiers (team_name, jet_lag_runs_penalty)
            VALUES (?, ?)
        ''', (team, penalty))

    conn.commit()
    conn.close()
    print("Biological tracking complete. Jet lag parameters locked.")

if __name__ == "__main__":
    execute_biological_pipeline()
