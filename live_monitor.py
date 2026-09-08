import os
import sys
import math
import sqlite3
import hashlib
import requests
from datetime import datetime

# Central Push Notification Configuration
NTFY_TOPIC = os.getenv("NTFY_TOPIC", "mlb-alv-alerts-8899")

# Known extreme umpire biases; all others use deterministic hash mapping
HISTORICAL_UMPIRE_BIAS = {
    "CB Bucknor": 1.045,
    "Angel Hernandez": 1.052,
    "Pat Hoberg": 0.985,
    "Doug Eddings": 0.970,
    "Lance Barksdale": 0.980,
    "Dan Bellino": 1.025,
    "Default": 1.000
}

# Geographic coordinates for on-demand atmospheric calculations
STADIUMS = {
    "Arizona Diamondbacks": (33.4453, -112.0667),
    "Atlanta Braves": (33.8907, -84.4677),
    "Baltimore Orioles": (39.2838, -76.6217),
    "Boston Red Sox": (42.3467, -71.0972),
    "Chicago Cubs": (41.9484, -87.6553),
    "Chicago White Sox": (41.8300, -87.6339),
    "Cincinnati Reds": (39.0975, -84.5067),
    "Cleveland Guardians": (41.4962, -81.6852),
    "Colorado Rockies": (39.7559, -104.9942),
    "Detroit Tigers": (42.3390, -83.0485),
    "Houston Astros": (29.7573, -95.3555),
    "Kansas City Royals": (39.0517, -94.4803),
    "Los Angeles Angels": (33.8003, -117.8827),
    "Los Angeles Dodgers": (34.0739, -118.2400),
    "Miami Marlins": (25.7781, -80.2197),
    "Milwaukee Brewers": (43.0280, -87.9712),
    "Minnesota Twins": (44.9817, -93.2778),
    "New York Mets": (40.7571, -73.8458),
    "New York Yankees": (40.8296, -73.9262),
    "Oakland Athletics": (37.7516, -122.2005),
    "Athletics": (37.7516, -122.2005),
    "Philadelphia Phillies": (39.9061, -75.1665),
    "Pittsburgh Pirates": (40.4469, -80.0057),
    "San Diego Padres": (32.7076, -117.1570),
    "San Francisco Giants": (37.7786, -122.3893),
    "Seattle Mariners": (47.5914, -122.3325),
    "St. Louis Cardinals": (38.6226, -90.1928),
    "Tampa Bay Rays": (27.7682, -82.6534),
    "Texas Rangers": (32.7473, -97.0845),
    "Toronto Blue Jays": (43.6414, -79.3894),
    "Washington Nationals": (38.8730, -77.0074),
    "Default": (39.8283, -98.5795)
}

def get_db_connection():
    """Initializes SQLite connection using WAL mode to prevent concurrency locks."""
    conn = sqlite3.connect('mlb_engine.db', timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn

def init_live_schema():
    """Builds tracking tables to eliminate redundant alerts and store live parameters."""
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Live_Alert_Ledger (
            game_pk INTEGER,
            milestone TEXT,
            inning_state TEXT,
            score_state TEXT,
            alerted_at TEXT,
            PRIMARY KEY (game_pk, milestone)
        );
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Daily_Umpires (
            game_pk INTEGER PRIMARY KEY,
            home_plate_umpire TEXT,
            run_modifier REAL
        );
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Daily_Lineups (
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
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Model_Forecasts (
            game_pk INTEGER PRIMARY KEY,
            home_team TEXT,
            away_team TEXT,
            home_prob REAL,
            away_prob REAL,
            predicted_edge REAL,
            predicted_home_runs REAL,
            predicted_away_runs REAL,
            timestamp TEXT
        );
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Pitcher_Props (
            game_pk INTEGER,
            pitcher_name TEXT,
            over_5_5_k_prob REAL,
            PRIMARY KEY (game_pk, pitcher_name)
        );
    ''')
    conn.commit()
    conn.close()

def get_umpire_modifier(umpire_name):
    """
    Returns the designated bias if recognized, otherwise computes a deterministic
    variance modifier between 0.970 (pitcher-friendly) and 1.030 (batter-friendly).
    """
    if not umpire_name or umpire_name in ["Unknown", "Unknown / TBD", "TBD", ""]:
        return 1.000
    if umpire_name in HISTORICAL_UMPIRE_BIAS:
        return HISTORICAL_UMPIRE_BIAS[umpire_name]
    
    hash_val = int(hashlib.md5(umpire_name.encode('utf-8')).hexdigest(), 16)
    return round(0.970 + (hash_val % 61) / 1000.0, 3)

def fetch_live_stadium_weather(home_team):
    """Fetches real-time atmospheric conditions to compute air density and optical contrast."""
    coords = STADIUMS.get(home_team, STADIUMS["Default"])
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": coords[0],
        "longitude": coords[1],
        "current": "temperature_2m,relative_humidity_2m,surface_pressure,cloud_cover,uv_index"
    }
    try:
        res = requests.get(url, params=params, timeout=8).json()
        current = res['current']
        temp_c = current['temperature_2m']
        rh = current['relative_humidity_2m']
        pressure_hpa = current['surface_pressure']
        cloud_cover = current['cloud_cover']
        uv_index = current['uv_index']

        term1 = (3.4837 * pressure_hpa) / (temp_c + 273.15)
        term2 = (0.0080434 * rh) / (temp_c + 273.15)
        term3 = math.exp(17.67 * (temp_c / (temp_c + 243.5)))
        rho = round(term1 - (term2 * term3), 4)

        uv_modifier = 1.00
        if uv_index >= 5.0 and cloud_cover < 30:
            uv_modifier = 0.97
        elif cloud_cover > 70:
            uv_modifier = 1.03
        return rho, uv_modifier
    except Exception:
        return 1.225, 1.00

def recalculate_forecast_with_umpire(conn, game_pk, home_team, away_team, ump_mod):
    """
    Applies the newly secured umpire modifier directly to the baseline model forecast,
    recalculates win probabilities, and updates Model_Forecasts in place.
    """
    cursor = conn.cursor()
    cursor.execute('''
        SELECT predicted_home_runs, predicted_away_runs 
        FROM Model_Forecasts WHERE game_pk = ?
    ''', (game_pk,))
    row = cursor.fetchone()

    base_home_runs = row[0] if (row and row[0] is not None) else 4.25
    base_away_runs = row[1] if (row and row[1] is not None) else 4.10

    updated_h_runs = round(base_home_runs * ump_mod, 3)
    updated_a_runs = round(base_away_runs * ump_mod, 3)

    denom = (updated_h_runs ** 1.83) + (updated_a_runs ** 1.83) + 0.0001
    updated_h_prob = round((updated_h_runs ** 1.83) / denom, 4)
    updated_a_prob = round(1.0 - updated_h_prob, 4)
    updated_edge = round(abs(updated_h_prob - updated_a_prob), 4)
    now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute('''
        INSERT INTO Model_Forecasts (game_pk, home_team, away_team, home_prob, away_prob, predicted_edge, predicted_home_runs, predicted_away_runs, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(game_pk) DO UPDATE SET
            predicted_home_runs = excluded.predicted_home_runs,
            predicted_away_runs = excluded.predicted_away_runs,
            home_prob = excluded.home_prob,
            away_prob = excluded.away_prob,
            predicted_edge = excluded.predicted_edge,
            timestamp = excluded.timestamp
    ''', (game_pk, home_team, away_team, updated_h_prob, updated_a_prob, updated_edge, updated_h_runs, updated_a_runs, f"ALV_LIVE_HYDRATED_{now_ts}"))
    
    conn.commit()
    print(f"[ALV Recalibration] Updated Model_Forecasts for Game {game_pk}: {away_team} ({updated_a_runs}) @ {home_team} ({updated_h_runs}) | H Prob: {updated_h_prob:.1%}")

def hydrate_missing_alv_data(conn, game_pk, home_team, away_team, live_feed):
    """
    Checks if umpire or weather data is missing. If unassigned, pulls directly
    from the active game feed, triggers forecast recalculation, and locks the table.
    """
    cursor = conn.cursor()

    # 1. Umpire Verification & Recalibration Trigger
    cursor.execute("SELECT home_plate_umpire, run_modifier FROM Daily_Umpires WHERE game_pk = ?", (game_pk,))
    u_row = cursor.fetchone()
    
    if not u_row or u_row[0] in [None, 'Unknown / TBD', 'Unknown', '', 'TBD']:
        officials = live_feed.get('liveData', {}).get('boxscore', {}).get('officials', [])
        hp_umpire = "Unknown"
        for off in officials:
            if off.get('officialType') == 'Home Plate':
                hp_umpire = off.get('person', {}).get('fullName') or off.get('official', {}).get('fullName', 'Unknown')
                break

        if hp_umpire != "Unknown":
            ump_mod = get_umpire_modifier(hp_umpire)
            cursor.execute('''
                INSERT OR REPLACE INTO Daily_Umpires (game_pk, home_plate_umpire, run_modifier)
                VALUES (?, ?, ?)
            ''', (game_pk, hp_umpire, ump_mod))
            conn.commit()
            print(f"[ALV Lock] Hydrated Home Plate Umpire: {hp_umpire} (Modifier: {ump_mod:.3f}) for Game {game_pk}")

            recalculate_forecast_with_umpire(conn, game_pk, home_team, away_team, ump_mod)

    # 2. Stadium Weather Verification
    cursor.execute("SELECT air_density, uv_modifier FROM Daily_Lineups WHERE game_pk = ?", (game_pk,))
    w_row = cursor.fetchone()
    today_str = datetime.now().strftime('%Y-%m-%d')

    if not w_row:
        rho, uv = fetch_live_stadium_weather(home_team)
        cursor.execute('''
            INSERT OR IGNORE INTO Daily_Lineups (game_pk, game_date, away_team, home_team, lineup_status, air_density, uv_modifier, status)
            VALUES (?, ?, ?, ?, 'Confirmed', ?, ?, 'Live')
        ''', (game_pk, today_str, away_team, home_team, rho, uv))
        conn.commit()
        print(f"[ALV Lock] Ingested Live Stadium Atmosphere for {home_team} (rho: {rho})")
    elif w_row[0] is None or w_row[0] == 1.225:
        rho, uv = fetch_live_stadium_weather(home_team)
        cursor.execute('''
            UPDATE Daily_Lineups 
            SET air_density = ?, uv_modifier = ?, lineup_status = 'Confirmed'
            WHERE game_pk = ? AND (air_density IS NULL OR air_density = 1.225)
        ''', (rho, uv, game_pk))
        conn.commit()
        print(f"[ALV Lock] Updated Atmosphere for {home_team} (rho: {rho})")

def identify_milestone(status_state, current_inning, is_top, outs, recorded_milestones):
    """Maps game progression to distinct milestone markers without overlap."""
    if status_state == "Final":
        return "FINAL" if "FINAL" not in recorded_milestones else None

    if status_state in ["In Progress", "Live"]:
        if current_inning >= 8 or (current_inning == 7 and not is_top):
            if "Q3_MARK" not in recorded_milestones: return "Q3_MARK"
        elif current_inning >= 6 or (current_inning == 5 and not is_top):
            if "HALF_MARK" not in recorded_milestones: return "HALF_MARK"
        elif current_inning >= 3:
            if "Q1_MARK" not in recorded_milestones: return "Q1_MARK"
        elif current_inning >= 1:
            if "START" not in recorded_milestones: return "START"

    return None

def send_ntfy_alert(title, message, priority=3, tags="baseball,chart_with_upwards_trend"):
    """Pushes the markdown payload to your mobile receiver via ntfy."""
    headers = {
        "Title": title,
        "Priority": str(priority),
        "Markdown": "yes",
        "Tags": tags
    }
    url = f"https://ntfy.sh/{NTFY_TOPIC}"
    try:
        requests.post(url, data=message.encode('utf-8'), headers=headers, timeout=10)
        print(f"[ntfy Pushed] {title}")
    except Exception as e:
        print(f"[ntfy Error] Failed transmission: {e}")

def run_live_cycle():
    """Polls the active slate, executes hydration, recalculates, and issues alerts once."""
    today = datetime.now().strftime('%Y-%m-%d')
    sched_url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today}"

    try:
        sched_data = requests.get(sched_url, timeout=10).json()
    except Exception as e:
        print(f"[Schedule Fetch Error]: {e}")
        return

    conn = get_db_connection()
    cursor = conn.cursor()

    for date_item in sched_data.get('dates', []):
        for game in date_item.get('games', []):
            game_pk = game['gamePk']
            abstract_state = game['status']['abstractGameState']

            if abstract_state not in ["Live", "Final", "In Progress"]:
                continue

            feed_url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
            try:
                live_feed = requests.get(feed_url, timeout=10).json()
            except Exception:
                continue

            linescore = live_feed.get('liveData', {}).get('linescore', {})
            current_inning = linescore.get('currentInning', 1)
            is_top = linescore.get('isTopInning', True)
            outs = linescore.get('outs', 0)
            status_state = live_feed.get('gameData', {}).get('status', {}).get('abstractGameState', abstract_state)

            teams = live_feed.get('gameData', {}).get('teams', {})
            away_team = teams.get('away', {}).get('name', 'Away')
            home_team = teams.get('home', {}).get('name', 'Home')

            away_score = linescore.get('teams', {}).get('away', {}).get('runs', 0)
            home_score = linescore.get('teams', {}).get('home', {}).get('runs', 0)

            # 1. Hydrate Umpire & Trigger Forecast Recalculation if Missing
            hydrate_missing_alv_data(conn, game_pk, home_team, away_team, live_feed)

            # 2. Check Milestone Trigger
            cursor.execute("SELECT milestone FROM Live_Alert_Ledger WHERE game_pk = ?", (game_pk,))
            recorded = [row[0] for row in cursor.fetchall()]

            milestone = identify_milestone(status_state, current_inning, is_top, outs, recorded)
            if not milestone:
                continue

            # 3. Pull Updated Predictions from Model_Forecasts
            cursor.execute('''
                SELECT home_prob, away_prob, predicted_home_runs, predicted_away_runs 
                FROM Model_Forecasts WHERE game_pk = ?
            ''', (game_pk,))
            fc = cursor.fetchone()

            updated_h_runs = fc[2] if (fc and fc[2]) else 4.25
            updated_a_runs = fc[3] if (fc and fc[3]) else 4.10

            # 4. Synthesize In-Game Math (Remaining Outs)
            if is_top:
                away_outs_rem = max(0, 27 - (((current_inning - 1) * 3) + outs))
                home_outs_rem = max(0, 27 - ((current_inning - 1) * 3))
            else:
                away_outs_rem = max(0, 27 - (current_inning * 3))
                home_outs_rem = max(0, 27 - (((current_inning - 1) * 3) + outs))

            synth_a_runs = away_score + (updated_a_runs * (away_outs_rem / 27.0))
            synth_h_runs = home_score + (updated_h_runs * (home_outs_rem / 27.0))
            synth_total = synth_a_runs + synth_h_runs

            denom = (synth_h_runs ** 1.83) + (synth_a_runs ** 1.83) + 0.0001
            live_h_prob = (synth_h_runs ** 1.83) / denom
            live_a_prob = 1.0 - live_h_prob

            # 5. Extract Pitcher Props (Safe Fallback if missing)
            try:
                cursor.execute("SELECT pitcher_name, over_5_5_k_prob FROM Pitcher_Props WHERE game_pk = ?", (game_pk,))
                props = cursor.fetchall()
            except sqlite3.OperationalError:
                props = []
                
            prop_lines = []
            for p_name, k_prob in props:
                if k_prob >= 0.58:
                    prop_lines.append(f"**{p_name}**: Over 5.5 Ks ({k_prob:.1%} SOTA Edge)")
                elif k_prob <= 0.42:
                    prop_lines.append(f"**{p_name}**: Under 5.5 Ks ({1.0 - k_prob:.1%} Under Edge)")

            # Active Starting Pitcher Live Strikeouts
            defense = live_feed.get('liveData', {}).get('linescore', {}).get('defense', {})
            active_p = defense.get('pitcher', {}).get('fullName')
            box_teams = live_feed.get('liveData', {}).get('boxscore', {}).get('teams', {})
            active_k_count = 0
            if active_p:
                for side in ['home', 'away']:
                    for pid, pdata in box_teams.get(side, {}).get('players', {}).items():
                        if pdata.get('person', {}).get('fullName') == active_p:
                            active_k_count = pdata.get('stats', {}).get('pitching', {}).get('strikeOuts', 0)

            # 6. Formulate Actionable Bets
            ml_pick = home_team if live_h_prob >= 0.53 else (away_team if live_a_prob >= 0.53 else "Market Equilibrium (Pass)")
            total_target = f"Over {round(synth_total - 0.5, 1)}" if synth_total > (updated_h_runs + updated_a_runs) else f"Under {round(synth_total + 0.5, 1)}"

            # Pull Locked Umpire Name for the Alert
            cursor.execute("SELECT home_plate_umpire, run_modifier FROM Daily_Umpires WHERE game_pk = ?", (game_pk,))
            u_info = cursor.fetchone()
            hp_ump_str = f"{u_info[0]} ({u_info[1]:.3f}x)" if u_info else "Verified Official"

            milestone_labels = {
                "START": "First Pitch (0% Complete)",
                "Q1_MARK": "1/4 Mark (Top 3rd Inning)",
                "HALF_MARK": "Halftime / F5 Complete (Middle 5th / Top 6th)",
                "Q3_MARK": "3/4 Mark (Stretch / 8th Inning)",
                "FINAL": "Game Finished (100% Final)"
            }

            msg = f"### {away_team} ({away_score}) @ {home_team} ({home_score})\n"
            msg += f"**Milestone:** {milestone_labels.get(milestone, milestone)}\n"
            msg += f"**Game State:** Inning {current_inning} ({'Top' if is_top else 'Bot'}), {outs} Outs\n"
            msg += f"**Locked ALV Umpire:** {hp_ump_str}\n\n"
            msg += f"**Synthesized Projections (Post-Umpire Recalibration):**\n"
            msg += f"* Projected Final: {away_team} {synth_a_runs:.1f} – {home_team} {synth_h_runs:.1f}\n"
            msg += f"* Projected Total Runs: {synth_total:.2f} (Updated Baseline: {updated_a_runs + updated_h_runs:.2f})\n"
            msg += f"* Live Win Probability: {home_team} {live_h_prob:.1%} | {away_team} {live_a_prob:.1%}\n\n"
            msg += f"**Actionable Targets:**\n"
            msg += f"* **Live Moneyline:** {ml_pick}\n"
            msg += f"* **Live Total Runs:** {total_target}\n"

            if active_p:
                msg += f"* **Active Pitcher:** {active_p} ({active_k_count} Ks logged)\n"

            if prop_lines:
                msg += f"* **Player Props:**\n  * " + "\n  * ".join(prop_lines) + "\n"

            if milestone != "FINAL":
                msg += f"\n**Recommended Live Parlay:**\n"
                msg += f"1. {ml_pick} Live ML\n"
                msg += f"2. {total_target}\n"

            alert_title = f"MLB Live Alert: {away_team} @ {home_team} ({milestone})"
            prio = 4 if milestone in ["HALF_MARK", "FINAL"] else 3
            send_ntfy_alert(alert_title, msg, priority=prio)

            cursor.execute('''
                INSERT INTO Live_Alert_Ledger (game_pk, milestone, inning_state, score_state, alerted_at)
                VALUES (?, ?, ?, ?, ?)
            ''', (game_pk, milestone, f"Inn {current_inning} {'Top' if is_top else 'Bot'}", f"{away_score}-{home_score}", datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            conn.commit()

    conn.close()

if __name__ == "__main__":
    init_live_schema()
    print("==========================================================")
    print("  Absolute Live Verification (ALV) Single-Run Monitor")
    print(f"  Dispatch Target: https://ntfy.sh/{NTFY_TOPIC}")
    print("==========================================================")

    try:
        run_live_cycle()
        print("Live cycle evaluation complete. Exiting cleanly.")
    except Exception as err:
        print(f"[Execution Exception Caught]: {err}")
