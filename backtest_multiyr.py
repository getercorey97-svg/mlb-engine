import sqlite3
import requests
import warnings
import numpy as np
from datetime import datetime, timedelta

warnings.filterwarnings("ignore", category=UserWarning)

STADIUM_RHO_BASELINES = {
    "Colorado Rockies": 1.050, "Arizona Diamondbacks": 1.075, "Texas Rangers": 1.135,
    "Atlanta Braves": 1.145, "Minnesota Twins": 1.150, "Cincinnati Reds": 1.160,
    "Detroit Tigers": 1.162, "Milwaukee Brewers": 1.163, "Chicago Cubs": 1.166,
    "Chicago White Sox": 1.167, "St. Louis Cardinals": 1.168, "Washington Nationals": 1.172,
    "Tampa Bay Rays": 1.175, "Miami Marlins": 1.185, "New York Yankees": 1.188,
    "Boston Red Sox": 1.195, "Baltimore Orioles": 1.198, "San Francisco Giants": 1.205,
    "Los Angeles Dodgers": 1.210, "Los Angeles Angels": 1.212, "New York Mets": 1.215,
    "Philadelphia Phillies": 1.218, "San Diego Padres": 1.225, "Seattle Mariners": 1.225,
    "Oakland Athletics": 1.220, "Athletics": 1.220, "Houston Astros": 1.180,
    "Kansas City Royals": 1.155, "Pittsburgh Pirates": 1.170, "Cleveland Guardians": 1.165,
    "Toronto Blue Jays": 1.190, "Default": 1.225
}

DEFAULT_PARK_FACTORS = {
    "Colorado Rockies": 1.38, "Boston Red Sox": 1.09, "Cincinnati Reds": 1.08,
    "Kansas City Royals": 1.05, "Texas Rangers": 1.04, "Arizona Diamondbacks": 1.04,
    "Philadelphia Phillies": 1.03, "Washington Nationals": 1.02, "Atlanta Braves": 1.01,
    "Baltimore Orioles": 1.01, "Chicago Cubs": 1.01, "Los Angeles Angels": 1.00,
    "Milwaukee Brewers": 1.00, "Minnesota Twins": 1.00, "Toronto Blue Jays": 1.00,
    "Chicago White Sox": 0.99, "Houston Astros": 0.99, "Pittsburgh Pirates": 0.98,
    "St. Louis Cardinals": 0.98, "Detroit Tigers": 0.97, "New York Yankees": 0.97,
    "Cleveland Guardians": 0.96, "Miami Marlins": 0.95, "Oakland Athletics": 0.95,
    "San Francisco Giants": 0.95, "Tampa Bay Rays": 0.94, "New York Mets": 0.94,
    "Los Angeles Dodgers": 0.93, "San Diego Padres": 0.92, "Seattle Mariners": 0.91
}

STADIUM_COORDS = {
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
    coords = STADIUM_COORDS.get(team_name, STADIUM_COORDS["Default"])
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": coords[0], "longitude": coords[1],
        "start_date": date_str, "end_date": date_str,
        "hourly": "surface_pressure,temperature_2m,cloud_cover"
    }
    try:
        res = requests.get(url, params=params, timeout=10).json()
        hourly = res.get('hourly', {})
        temps = hourly.get('temperature_2m', [])
        pressures = hourly.get('surface_pressure', [])
        clouds = hourly.get('cloud_cover', [])

        idx = 12 if len(temps) > 12 else 0
        temp_c = temps[idx] if len(temps) > idx and temps[idx] is not None else 15.0
        pressure_hpa = pressures[idx] if len(pressures) > idx and pressures[idx] is not None else 1013.25
        cloud_cover = clouds[idx] if len(clouds) > idx and clouds[idx] is not None else 0.0

        temp_k = temp_c + 273.15
        pressure_pa = pressure_hpa * 100
        density = round(pressure_pa / (287.05 * temp_k), 4)
        uv_raw = 3.0 if cloud_cover > 70 else 7.0
        return density, uv_raw
    except Exception:
        return STADIUM_RHO_BASELINES.get(team_name, 1.225), 5.0

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
            away_score INTEGER, home_f5_score INTEGER, away_f5_score INTEGER,
            model_correct INTEGER, processed_at TEXT
        );
        CREATE TABLE IF NOT EXISTS Daily_Lineups (
            game_pk INTEGER PRIMARY KEY, game_date TEXT, away_team TEXT, home_team TEXT, 
            away_pitcher TEXT, home_pitcher TEXT, air_density REAL, uv_modifier REAL, status TEXT
        );
        CREATE TABLE IF NOT EXISTS Dynamic_Modifiers (
            team_name TEXT PRIMARY KEY, offensive_modifier REAL DEFAULT 1.0, 
            pitching_modifier REAL DEFAULT 1.0, appearance_count INTEGER DEFAULT 0, last_updated TEXT
        );
        CREATE TABLE IF NOT EXISTS Pitcher_Modifiers (
            pitcher_name TEXT PRIMARY KEY, k_modifier REAL DEFAULT 1.0, 
            f5_run_modifier REAL DEFAULT 1.0, appearance_count INTEGER DEFAULT 0, last_updated TEXT
        );
        CREATE TABLE IF NOT EXISTS Bullpen_Fatigue (
            team_name TEXT PRIMARY KEY, fatigue_multiplier REAL DEFAULT 1.00, 
            rolling_ip_3d REAL DEFAULT 8.0, last_updated TEXT
        );
    ''')
    conn.commit()

    pitcher_stats = {r[0]: (r[1], r[2]) for r in cursor.execute(
        "SELECT last_name, COALESCE(xfip, est_era, 4.20), COALESCE(throws, 'R') FROM Pitcher_Stats"
    ).fetchall()}
    platoon_ops = {r[0]: (r[1], r[2]) for r in cursor.execute(
        "SELECT team_name, COALESCE(ops_vs_rhp, 0.720), COALESCE(ops_vs_lhp, 0.720) FROM Team_Offense"
    ).fetchall()}

    end_date = datetime.now() - timedelta(days=1)
    start_date = end_date - timedelta(days=365 * years_back)
    current_date = start_date

    total_games, correct_predictions = 0, 0
    team_recent_workload = {}

    while current_date <= end_date:
        date_str = current_date.strftime('%Y-%m-%d')
        curr_dt = current_date
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
                
                if home_score == away_score:
                    continue
                
                home_pitcher = game['teams']['home'].get('probablePitcher', {}).get('fullName', 'Unknown')
                away_pitcher = game['teams']['away'].get('probablePitcher', {}).get('fullName', 'Unknown')
                
                innings = game.get('linescore', {}).get('innings', [])
                h_f5 = sum(inn.get('home', {}).get('runs') or 0 for inn in innings[:5])
                a_f5 = sum(inn.get('away', {}).get('runs') or 0 for inn in innings[:5])
                h_late = max(0, home_score - h_f5)
                a_late = max(0, away_score - a_f5)

                air_density, uv_raw = get_historical_atmosphere(home_team, date_str)
                
                cursor.execute('''
                    INSERT OR REPLACE INTO Daily_Lineups (game_pk, game_date, away_team, home_team, away_pitcher, home_pitcher, air_density, uv_modifier, status)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (game_pk, date_str, away_team, home_team, away_pitcher, home_pitcher, air_density, uv_raw, 'Final'))
                
                # Fetch learning states
                cursor.execute('SELECT offensive_modifier, pitching_modifier, appearance_count FROM Dynamic_Modifiers WHERE team_name = ?', (home_team,))
                h_team_row = cursor.fetchone() or (1.0, 1.0, 0)
                cursor.execute('SELECT offensive_modifier, pitching_modifier, appearance_count FROM Dynamic_Modifiers WHERE team_name = ?', (away_team,))
                a_team_row = cursor.fetchone() or (1.0, 1.0, 0)

                cursor.execute('SELECT f5_run_modifier, appearance_count FROM Pitcher_Modifiers WHERE pitcher_name = ?', (home_pitcher,))
                h_p_row = cursor.fetchone() or (1.0, 0)
                cursor.execute('SELECT f5_run_modifier, appearance_count FROM Pitcher_Modifiers WHERE pitcher_name = ?', (away_pitcher,))
                a_p_row = cursor.fetchone() or (1.0, 0)

                # Physical Bullpen Workload
                def compute_sim_workload(team_name):
                    recent = team_recent_workload.get(team_name, [])
                    valid = [r for r in recent if 1 <= (curr_dt - r[0]).days <= 3]
                    if not valid:
                        return 1.00
                    tot_ip = sum(r[1] for r in valid)
                    played_yest = any((curr_dt - r[0]).days == 1 for r in valid)
                    strain = (tot_ip - 8.0) * 0.025
                    b2b = 0.03 if (played_yest and len(valid) >= 2) else 0.00
                    return round(float(np.clip(1.00 + strain + b2b, 0.85, 1.25)), 4)

                h_bp_fatigue = compute_sim_workload(home_team)
                a_bp_fatigue = compute_sim_workload(away_team)

                # Bayesian Shrinkage
                w_h_t = min(1.0, h_team_row[2] / 15.0)
                w_a_t = min(1.0, a_team_row[2] / 15.0)
                h_off = w_h_t * h_team_row[0] + (1.0 - w_h_t) * 1.0
                h_pitch = w_h_t * h_team_row[1] + (1.0 - w_h_t) * 1.0
                a_off = w_a_t * a_team_row[0] + (1.0 - w_a_t) * 1.0
                a_pitch = w_a_t * a_team_row[1] + (1.0 - w_a_t) * 1.0

                w_h_p = min(1.0, h_p_row[1] / 10.0)
                w_a_p = min(1.0, a_p_row[1] / 10.0)
                h_p_mod = w_h_p * h_p_row[0] + (1.0 - w_h_p) * 1.0
                a_p_mod = w_a_p * a_p_row[0] + (1.0 - w_a_p) * 1.0

                # Matchup adjustments
                a_sp_ln = away_pitcher.split()[-1] if " " in away_pitcher else away_pitcher
                h_sp_ln = home_pitcher.split()[-1] if " " in home_pitcher else home_pitcher
                _, a_throws = pitcher_stats.get(a_sp_ln, (4.20, 'R'))
                _, h_throws = pitcher_stats.get(h_sp_ln, (4.20, 'R'))

                away_ops_rhp, away_ops_lhp = platoon_ops.get(away_team, (0.720, 0.720))
                home_ops_rhp, home_ops_lhp = platoon_ops.get(home_team, (0.720, 0.720))
                a_plat = away_ops_lhp if h_throws == 'L' else away_ops_rhp
                h_plat = home_ops_lhp if a_throws == 'L' else home_ops_rhp

                park_mult = DEFAULT_PARK_FACTORS.get(home_team, 1.00)
                air_drag = 1.000 + ((1.225 - air_density) * 1.5)
                uv_glare = 1.000 + (np.clip(uv_raw, 1.0, 11.0) - 5.0) * 0.005
                env_mult = park_mult * air_drag * uv_glare

                pred_h_runs = round(4.45 * (h_plat / 0.720) * h_off * (0.55 * a_p_mod + 0.45 * a_bp_fatigue * a_pitch) * env_mult, 2)
                pred_a_runs = round(4.25 * (a_plat / 0.720) * a_off * (0.55 * h_p_mod + 0.45 * h_bp_fatigue * h_pitch) * env_mult, 2)

                denom = (pred_h_runs ** 1.83) + (pred_a_runs ** 1.83)
                home_prob = round((pred_h_runs ** 1.83) / denom, 4) if denom > 0 else 0.50
                away_prob = round(1.0 - home_prob, 4)

                cursor.execute('''
                    INSERT OR REPLACE INTO Model_Forecasts (game_pk, home_team, away_team, home_prob, away_prob, predicted_edge, predicted_home_runs, predicted_away_runs, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ''', (game_pk, home_team, away_team, home_prob, away_prob, round(abs(home_prob - away_prob), 4), pred_h_runs, pred_a_runs, date_str))

                actual_winner = home_team if home_score > away_score else away_team
                predicted_winner = home_team if home_prob >= away_prob else away_team
                is_correct = 1 if predicted_winner == actual_winner else 0
                
                total_games += 1
                correct_predictions += is_correct

                # Log physical bullpen workload
                team_recent_workload.setdefault(home_team, []).append((curr_dt, 4.0 + max(0.0, (a_late - 2) * 0.25)))
                team_recent_workload.setdefault(away_team, []).append((curr_dt, 4.0 + max(0.0, (h_late - 2) * 0.25)))

                # Starting Pitcher F5 EWMA Updates
                pred_h_f5, pred_a_f5 = pred_h_runs * 0.55, pred_a_runs * 0.55
                for sp, p_pred, p_act, old_m, count in [(home_pitcher, pred_a_f5, a_f5, h_p_row[0], h_p_row[1]), 
                                                        (away_pitcher, pred_h_f5, h_f5, a_p_row[0], a_p_row[1])]:
                    err = p_act - p_pred
                    alpha = min(0.12, 0.03 + (abs(err) * 0.01))
                    new_mod = max(0.70, min(1.30, alpha * (old_m + err * 0.05) + (1.0 - alpha) * old_m))
                    cursor.execute('''
                        INSERT OR REPLACE INTO Pitcher_Modifiers (pitcher_name, k_modifier, f5_run_modifier, appearance_count, last_updated)
                        VALUES (?, 1.0, ?, ?, ?)
                    ''', (sp, round(new_mod, 4), count + 1, date_str))

                # Team Offense & Pitching Late-Inning Updates
                pred_h_late, pred_a_late = pred_h_runs * 0.45, pred_a_runs * 0.45
                for tm, pred_l, act_l, is_off, old_off, old_pit, n_cnt in [
                    (home_team, pred_h_late, h_late, True, h_team_row[0], h_team_row[1], h_team_row[2]),
                    (away_team, pred_h_late, h_late, False, a_team_row[0], a_team_row[1], a_team_row[2]),
                    (away_team, pred_a_late, a_late, True, a_team_row[0], a_team_row[1], a_team_row[2]),
                    (home_team, pred_a_late, a_late, False, h_team_row[0], h_team_row[1], h_team_row[2])
                ]:
                    err = act_l - pred_l
                    alpha = min(0.10, 0.02 + (abs(err) * 0.008))
                    if is_off:
                        new_off = max(0.70, min(1.30, alpha * (old_off + err * 0.04) + (1.0 - alpha) * old_off))
                        cursor.execute('''
                            INSERT OR REPLACE INTO Dynamic_Modifiers (team_name, offensive_modifier, pitching_modifier, appearance_count, last_updated)
                            VALUES (?, ?, ?, ?, ?)
                        ''', (tm, round(new_off, 4), round(old_pit, 4), n_cnt + 1, date_str))
                    else:
                        new_pit = max(0.70, min(1.30, alpha * (old_pit + err * 0.04) + (1.0 - alpha) * old_pit))
                        cursor.execute('''
                            INSERT OR REPLACE INTO Dynamic_Modifiers (team_name, offensive_modifier, pitching_modifier, appearance_count, last_updated)
                            VALUES (?, ?, ?, ?, ?)
                        ''', (tm, round(old_off, 4), round(new_pit, 4), n_cnt + 1, date_str))

                cursor.execute('''
                    INSERT OR REPLACE INTO Post_Match_Analysis (game_pk, actual_winner, home_score, away_score, home_f5_score, away_f5_score, model_correct, processed_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (game_pk, actual_winner, home_score, away_score, h_f5, a_f5, is_correct, date_str))
        
        conn.commit()

    conn.close()
    win_rate = (correct_predictions / total_games) * 100 if total_games > 0 else 0
    print(f"\n[BACKTEST COMPLETE] Verified {total_games} games across {years_back} year(s). Historical Win Rate: {win_rate:.2f}%")

if __name__ == "__main__":
    run_backtest_sweep(years_back=1)
