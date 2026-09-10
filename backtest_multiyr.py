import sqlite3
import requests
import numpy as np
import pandas as pd
import warnings
from datetime import datetime, timedelta
from sklearn.isotonic import IsotonicRegression

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
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": coords[0], "longitude": coords[1],
        "start_date": date_str, "end_date": date_str,
        "hourly": "surface_pressure,temperature_2m,cloud_cover"
    }
    try:
        res = requests.get(url, params=params, timeout=10).json()
        temp_c = res['hourly']['temperature_2m'][12] 
        pressure_hpa = res['hourly']['surface_pressure'][12]
        cloud_cover = res['hourly']['cloud_cover'][12]
        temp_k = temp_c + 273.15
        pressure_pa = pressure_hpa * 100
        density = round(pressure_pa / (287.05 * temp_k), 4)
        uv_modifier = 1.03 if cloud_cover > 70 else 1.00
        return density, uv_modifier
    except Exception:
        return 1.225, 1.00

def update_dynamic_weights(cursor, name, predicted_runs, actual_runs, is_offense=True):
    error_delta = actual_runs - predicted_runs
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    adaptive_lr = min(0.15, 0.03 + (abs(error_delta) * 0.015))
    
    cursor.execute('SELECT offensive_modifier, pitching_modifier FROM Dynamic_Modifiers WHERE team_name = ?', (name,))
    result = cursor.fetchone()
    off_mod, pitch_mod = result if result else (1.0, 1.0)
    
    if is_offense:
        new_off_mod = max(0.53, min(1.47, off_mod + (error_delta * adaptive_lr)))
        cursor.execute('INSERT OR REPLACE INTO Dynamic_Modifiers (team_name, offensive_modifier, pitching_modifier, last_updated) VALUES (?, ?, ?, ?)', (name, new_off_mod, pitch_mod, current_time))
    else:
        new_pitch_mod = max(0.53, min(1.47, pitch_mod + (error_delta * adaptive_lr)))
        cursor.execute('INSERT OR REPLACE INTO Dynamic_Modifiers (team_name, offensive_modifier, pitching_modifier, last_updated) VALUES (?, ?, ?, ?)', (name, off_mod, new_pitch_mod, current_time))

def run_backtest_sweep(years_back=1):
    print(f"Initializing SOTA Multi-Year Backtest Engine ({years_back}-Year Historical Sweep)...")
    
    conn = sqlite3.connect('mlb_engine.db', timeout=30)
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA busy_timeout=10000;")
    
    cursor.executescript('''
        CREATE TABLE IF NOT EXISTS Model_Forecasts (
            game_pk INTEGER PRIMARY KEY, home_team TEXT, away_team TEXT, 
            home_prob REAL, away_prob REAL, predicted_edge REAL, 
            predicted_home_runs REAL, predicted_away_runs REAL, timestamp TEXT
        );
        CREATE TABLE IF NOT EXISTS Post_Match_Analysis (
            game_pk INTEGER PRIMARY KEY, actual_winner TEXT, home_score INTEGER, 
            away_score INTEGER, model_correct INTEGER, processed_at TEXT
        );
        CREATE TABLE IF NOT EXISTS Daily_Lineups (
            game_pk INTEGER PRIMARY KEY, game_date TEXT, away_team TEXT, home_team TEXT, 
            away_pitcher TEXT, home_pitcher TEXT, air_density REAL, uv_modifier REAL, status TEXT
        );
        CREATE TABLE IF NOT EXISTS Dynamic_Modifiers (
            team_name TEXT PRIMARY KEY, offensive_modifier REAL DEFAULT 1.0, pitching_modifier REAL DEFAULT 1.0, last_updated TEXT
        );
    ''')
    conn.commit()

    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=365 * years_back)
    
    current_date = start_date
    total_games, correct_predictions = 0, 0

    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        current_date += timedelta(days=1)
        
        day_url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}&gameType=R&hydrate=probablePitcher,linescore"
        try:
            day_res = requests.get(day_url, timeout=15).json()
        except Exception:
            continue
            
        for date_data in day_res.get('dates', []):
            for game in date_data.get('games', []):
                if game.get('status', {}).get('abstractGameState') != 'Final':
                    continue
                    
                game_pk = game['gamePk']
                home_team = game['teams']['home']['team']['name']
                away_team = game['teams']['away']['team']['name']
                home_score = game['teams']['home'].get('score', 0)
                away_score = game['teams']['away'].get('score', 0)
                
                home_pitcher = game['teams']['home'].get('probablePitcher', {}).get('fullName', 'Unknown')
                away_pitcher = game['teams']['away'].get('probablePitcher', {}).get('fullName', 'Unknown')
                
                air_density, uv_modifier = get_historical_atmosphere(home_team, date_str)
                
                cursor.execute('''
                    INSERT OR REPLACE INTO Daily_Lineups (game_pk, game_date, away_team, home_team, away_pitcher, home_pitcher, air_density, uv_modifier, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (game_pk, date_str, away_team, home_team, away_pitcher, home_pitcher, air_density, uv_modifier, 'Final'))
                
                cursor.execute('SELECT offensive_modifier, pitching_modifier FROM Dynamic_Modifiers WHERE team_name = ?', (home_team,))
                h_mod = cursor.fetchone() or (1.0, 1.0)
                cursor.execute('SELECT offensive_modifier, pitching_modifier FROM Dynamic_Modifiers WHERE team_name = ?', (away_team,))
                a_mod = cursor.fetchone() or (1.0, 1.0)
                
                pred_h_runs = round(4.2 * h_mod[0] * a_mod[1] * (1.225 / air_density), 2)
                pred_a_runs = round(4.0 * a_mod[0] * h_mod[1], 2)
                
                home_prob = 0.52 if pred_h_runs > pred_a_runs else 0.48
                away_prob = 1.0 - home_prob
                
                cursor.execute('''
                    INSERT OR REPLACE INTO Model_Forecasts (game_pk, home_team, away_team, home_prob, away_prob, predicted_edge, predicted_home_runs, predicted_away_runs, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (game_pk, home_team, away_team, home_prob, away_prob, abs(home_prob - away_prob), pred_h_runs, pred_a_runs, date_str))

                actual_winner = home_team if home_score > away_score else away_team
                predicted_winner = home_team if home_prob > away_prob else away_team
                is_correct = 1 if predicted_winner == actual_winner else 0
                
                total_games += 1
                correct_predictions += is_correct
                
                update_dynamic_weights(cursor, home_team, pred_h_runs, home_score, is_offense=True)
                update_dynamic_weights(cursor, away_team, pred_a_runs, away_score, is_offense=True)
                
                cursor.execute('''
                    INSERT OR REPLACE INTO Post_Match_Analysis (game_pk, actual_winner, home_score, away_score, model_correct, processed_at)
                    VALUES (?, ?, ?, ?, ?, ?)
                ''', (game_pk, actual_winner, home_score, away_score, is_correct, 'BACKTEST'))
        
        conn.commit()

    conn.close()
    win_rate = (correct_predictions / total_games) * 100 if total_games > 0 else 0
    print(f"\n[BACKTEST COMPLETE] Verified {total_games} games across {years_back} year(s). Historical Win Rate: {win_rate:.2f}%")

if __name__ == "__main__":
    run_backtest_sweep(years_back=1)
