import sqlite3
import requests
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

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

def initialize_database_schemas():
    """Guarantees every table and column exists with appearance tracking migrations."""
    conn = sqlite3.connect('mlb_engine.db', timeout=30)
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA busy_timeout=10000;")
    
    cursor.executescript('''
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
        CREATE TABLE IF NOT EXISTS F5_Forecasts (
            game_pk INTEGER PRIMARY KEY,
            away_team TEXT,
            home_team TEXT,
            away_starter TEXT,
            home_starter TEXT,
            f5_away_prob REAL,
            f5_home_prob REAL,
            f5_tie_prob REAL,
            f5_exp_away_runs REAL,
            f5_exp_home_runs REAL,
            f5_total_runs REAL
        );
        CREATE TABLE IF NOT EXISTS Pitcher_Props (
            game_pk INTEGER,
            pitcher_name TEXT,
            team_name TEXT,
            projected_outs REAL,
            projected_strikeouts REAL,
            over_4_5_k_prob REAL,
            over_5_5_k_prob REAL,
            over_6_5_k_prob REAL,
            PRIMARY KEY (game_pk, pitcher_name)
        );
        CREATE TABLE IF NOT EXISTS Dynamic_Modifiers (
            team_name TEXT PRIMARY KEY,
            offensive_modifier REAL DEFAULT 1.0,
            pitching_modifier REAL DEFAULT 1.0,
            appearance_count INTEGER DEFAULT 0,
            last_updated TEXT
        );
        CREATE TABLE IF NOT EXISTS Pitcher_Modifiers (
            pitcher_name TEXT PRIMARY KEY,
            k_modifier REAL DEFAULT 1.0,
            f5_run_modifier REAL DEFAULT 1.0,
            appearance_count INTEGER DEFAULT 0,
            last_updated TEXT
        );
        CREATE TABLE IF NOT EXISTS Daily_Lineups (
            game_pk INTEGER PRIMARY KEY,
            game_date TEXT,
            away_team TEXT,
            home_team TEXT,
            away_pitcher TEXT,
            home_pitcher TEXT,
            lineup_status TEXT,
            air_density REAL DEFAULT 1.225,
            uv_modifier REAL DEFAULT 1.00,
            status TEXT
        );
        CREATE TABLE IF NOT EXISTS Daily_Umpires (
            game_pk INTEGER PRIMARY KEY,
            home_plate_umpire TEXT,
            run_modifier REAL DEFAULT 1.00,
            umpire_locked INTEGER DEFAULT 0,
            updated_at TEXT
        );
        CREATE TABLE IF NOT EXISTS Esoteric_Signals (
            game_pk INTEGER PRIMARY KEY,
            geomagnetic_kp REAL DEFAULT 2.0,
            solar_xray_flux REAL DEFAULT 1.0,
            home_media_pressure INTEGER DEFAULT 0,
            home_media_tone REAL DEFAULT 0.0,
            roster_birthday_active INTEGER DEFAULT 0,
            captured_at TEXT
        );
        CREATE TABLE IF NOT EXISTS Park_Factors (
            home_team TEXT PRIMARY KEY,
            run_factor REAL DEFAULT 1.00
        );
        CREATE TABLE IF NOT EXISTS Bullpen_Fatigue (
            team_name TEXT PRIMARY KEY,
            fatigue_multiplier REAL DEFAULT 1.00
        );
        CREATE TABLE IF NOT EXISTS Post_Match_Analysis (
            game_pk INTEGER PRIMARY KEY,
            actual_winner TEXT,
            home_score INTEGER,
            away_score INTEGER,
            home_f5_score INTEGER,
            away_f5_score INTEGER,
            model_correct INTEGER,
            processed_at TEXT
        );
    ''')

    for table, col in [("Pitcher_Modifiers", "appearance_count INTEGER DEFAULT 0"),
                       ("Dynamic_Modifiers", "appearance_count INTEGER DEFAULT 0"),
                       ("Daily_Lineups", "uv_modifier REAL DEFAULT 1.00"),
                       ("Daily_Umpires", "umpire_locked INTEGER DEFAULT 0")]:
        try:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col};")
        except sqlite3.OperationalError:
            pass

    cursor.execute("SELECT COUNT(*) FROM Park_Factors")
    if cursor.fetchone()[0] == 0:
        for team, factor in DEFAULT_PARK_FACTORS.items():
            cursor.execute("INSERT OR REPLACE INTO Park_Factors (home_team, run_factor) VALUES (?, ?)", (team, factor))

    conn.commit()
    conn.close()
    print("[PHASE 1] Schemas verified and base park factors seeded.")

def fetch_daily_matchups_and_lineups():
    """Wipes old unplayed slate and ingests strictly today's MLB schedule."""
    print("[PHASE 3] Ingesting Today's MLB Schedule & Probable Pitchers...")
    conn = sqlite3.connect('mlb_engine.db', timeout=30)
    cursor = conn.cursor()

    # Clear previous active lineup slate to prevent game stacking and repeats
    cursor.execute("DELETE FROM Daily_Lineups;")

    # Restrict ingestion strictly to today's date
    today_str = datetime.now().strftime('%Y-%m-%d')
    total_ingested = 0

    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today_str}&hydrate=probablePitcher,lineups,linescore"
    try:
        res = requests.get(url, timeout=15).json()
        for date_item in res.get('dates', []):
            for g in date_item.get('games', []):
                pk = g.get('gamePk')
                status = g.get('status', {}).get('abstractGameState', 'Scheduled')
                teams = g.get('teams', {})
                home = teams.get('home', {}).get('team', {}).get('name')
                away = teams.get('away', {}).get('team', {}).get('name')

                if not pk or not home or not away:
                    continue

                home_p = teams.get('home', {}).get('probablePitcher', {}).get('fullName', 'TBD')
                away_p = teams.get('away', {}).get('probablePitcher', {}).get('fullName', 'TBD')
                rho = STADIUM_RHO_BASELINES.get(home, 1.225)

                cursor.execute('''
                    INSERT OR REPLACE INTO Daily_Lineups 
                    (game_pk, game_date, away_team, home_team, away_pitcher, home_pitcher, lineup_status, air_density, uv_modifier, status)
                    VALUES (?, ?, ?, ?, ?, ?, 'Pending/TBD', ?, 1.00, ?)
                ''', (pk, today_str, away, home, away_p, home_p, rho, status))
                total_ingested += 1
    except Exception as e:
        print(f"Schedule pull error for {today_str}: {e}")

    conn.commit()
    conn.close()
    print(f"[PHASE 3 COMPLETE] Synchronized {total_ingested} matchups strictly for {today_str}.")

def export_prediction_markdown():
    """Generates PREDICTIONS_TODAY.md containing Full Game & F5 market values."""
    print("[PHASE 6] Exporting Consolidated Prediction Markdown...")
    conn = sqlite3.connect('mlb_engine.db', timeout=30)
    cursor = conn.cursor()

    query = '''
    SELECT 
        m.game_pk, m.away_team, m.home_team, 
        m.away_prob, m.home_prob, m.predicted_edge,
        m.predicted_away_runs, m.predicted_home_runs,
        f.f5_away_prob, f.f5_home_prob, f.f5_tie_prob, f.f5_total_runs,
        d.away_pitcher, d.home_pitcher
    FROM Model_Forecasts m
    INNER JOIN Daily_Lineups d ON m.game_pk = d.game_pk
    LEFT JOIN F5_Forecasts f ON m.game_pk = f.game_pk
    WHERE d.status != 'Final' AND m.game_pk NOT IN (SELECT game_pk FROM Post_Match_Analysis)
    ORDER BY m.predicted_edge DESC
    '''
    cursor.execute(query)
    rows = cursor.fetchall()
    conn.close()

    today_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')
    lines = [
        f"# MLB Predictive Model Forecasts ({today_str})",
        "",
        "### 🎯 Full Game Projections (Moneyline & Run Expectancies)",
        "",
        "| Matchup | Best Pick | Win Prob | Edge | Proj Score | Pitchers |",
        "| :--- | :---: | :---: | :---: | :---: | :--- |"
    ]

    if not rows:
        lines.append("| No unplayed active games scheduled | - | - | - | - | - |")
    else:
        for r in rows:
            pk, away, home, a_prob, h_prob, edge, a_runs, h_runs, f5_a, f5_h, f5_t, f5_tot, a_sp, h_sp = r
            pick = home if h_prob >= a_prob else away
            prob = max(h_prob, a_prob)
            lines.append(f"| {away} @ {home} | **{pick}** | {prob:.1%} | +{edge*100:.1f}% | {a_runs:.1f} - {h_runs:.1f} | {a_sp} vs {h_sp} |")

        lines.extend([
            "",
            "### ⚡ First 5 (F5) & Props Projections",
            "",
            "| Matchup | F5 Away Prob | F5 Home Prob | F5 Tie Prob | Expected F5 Total |",
            "| :--- | :---: | :---: | :---: | :---: |"
        ])
        for r in rows:
            pk, away, home, a_prob, h_prob, edge, a_runs, h_runs, f5_a, f5_h, f5_t, f5_tot, a_sp, h_sp = r
            f5_a_str = f"{f5_a:.1%}" if f5_a is not None else "-"
            f5_h_str = f"{f5_h:.1%}" if f5_h is not None else "-"
            f5_t_str = f"{f5_t:.1%}" if f5_t is not None else "-"
            f5_tot_str = f"{f5_tot:.2f}" if f5_tot is not None else "-"
            lines.append(f"| {away} @ {home} | {f5_a_str} | {f5_h_str} | {f5_t_str} | {f5_tot_str} runs |")

    with open("PREDICTIONS_TODAY.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("[PHASE 6 COMPLETE] PREDICTIONS_TODAY.md successfully generated.")

def main():
    print("=" * 65)
    print(f"[{datetime.now()}] Starting Unified MLB Orchestration Pipeline")
    print("=" * 65)

    # 1. Schema Validation & Baseline Setup
    initialize_database_schemas()

    # 2. Post-Match Learning Loop: Updates weights, modifiers & calibrations from completed games first
    try:
        import post_match_analysis
        print("[PHASE 2] Executing Post-Match Learning Loop...")
        post_match_analysis.run_post_match_analysis()
    except Exception as e:
        print(f"[BYPASS] Post-Match analysis skipped: {e}")

    # 3. Ingest Strictly Today's Matchups & Probable Pitchers into Daily_Lineups
    fetch_daily_matchups_and_lineups()

    # 3.5 Execute Lineup Verification natively into the Daily_Lineups table
    try:
        import lineup_verifier
        print("[PHASE 3.5] Verifying Starting Lineup Confirmations...")
        lineup_verifier.verify_starting_lineups()
    except Exception as e:
        print(f"[BYPASS] Lineup verifier skipped: {e}")

    # 4. Ingest Esoteric Signals (NOAA + GDELT)
    try:
        import open_source_discovery
        print("[PHASE 4] Executing Signal Discovery Sweep...")
        open_source_discovery.execute_discovery_ingestion()
    except Exception as e:
        print(f"[BYPASS] Discovery ingestion skipped: {e}")

    # 5. Core Full Game Monte Carlo Simulation (using updated weights)
    try:
        import engine
        print("[PHASE 5A] Executing Full Game Monte Carlo Engine...")
        engine.run_ultimate_monte_carlo()
    except Exception as e:
        print(f"[ERROR] Engine failure: {e}")

    # 6. Dedicated F5 & Props Engine (using updated weights and correct filename)
    try:
        import engine_f5_props as engine_f5
        print("[PHASE 5B] Executing First 5 & Pitcher Props Engine...")
        engine_f5.run_f5_and_props_engine()
    except Exception as e:
        print(f"[ERROR] Engine F5 failure: {e}")

    # 7. Compile Markdown Projections for Today's Slate
    export_prediction_markdown()

    print("=" * 65)
    print(f"[{datetime.now()}] Orchestration Pipeline Complete. Forecasts Synchronized.")
    print("=" * 65)

if __name__ == "__main__":
    main()
