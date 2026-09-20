import sqlite3
import requests
from datetime import datetime, timezone, timedelta
from zoneinfo import ZoneInfo

def get_db_connection():
    conn = sqlite3.connect("mlb_engine.db")
    conn.row_factory = sqlite3.Row
    return conn

def auto_repair_lineup_schema(conn):
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS Daily_Lineups (
            game_pk INTEGER PRIMARY KEY,
            away_team TEXT,
            home_team TEXT,
            away_sp TEXT,
            home_sp TEXT,
            away_pitcher TEXT,
            home_pitcher TEXT,
            lineup_status TEXT,
            status TEXT,
            game_datetime_utc TEXT,
            game_time_et TEXT,
            gatekeeper_trigger_utc TEXT,
            ingested_at TEXT
        )
    """)
    existing_dl = [r[1] for r in c.execute("PRAGMA table_info(Daily_Lineups)").fetchall()]
    required_dl = [
        ("away_team", "TEXT"), ("home_team", "TEXT"),
        ("away_sp", "TEXT"), ("home_sp", "TEXT"),
        ("away_pitcher", "TEXT"), ("home_pitcher", "TEXT"),
        ("lineup_status", "TEXT"), ("status", "TEXT"),
        ("game_datetime_utc", "TEXT"), ("game_time_et", "TEXT"),
        ("gatekeeper_trigger_utc", "TEXT"), ("ingested_at", "TEXT")
    ]
    for col_name, col_type in required_dl:
        if col_name not in existing_dl:
            c.execute(f"ALTER TABLE Daily_Lineups ADD COLUMN {col_name} {col_type}")

    c.execute("""
        CREATE TABLE IF NOT EXISTS Daily_Batters (
            game_pk INTEGER,
            player_name TEXT,
            team_name TEXT,
            batting_order INTEGER,
            is_starter INTEGER DEFAULT 1,
            is_confirmed INTEGER DEFAULT 0,
            game_datetime_utc TEXT,
            game_time_et TEXT,
            PRIMARY KEY (game_pk, player_name)
        )
    """)
    existing_db = [r[1] for r in c.execute("PRAGMA table_info(Daily_Batters)").fetchall()]
    required_db = [
        ("game_datetime_utc", "TEXT"), ("game_time_et", "TEXT"),
        ("is_starter", "INTEGER DEFAULT 1"), ("is_confirmed", "INTEGER DEFAULT 0")
    ]
    for col_name, col_type in required_db:
        if col_name not in existing_db:
            c.execute(f"ALTER TABLE Daily_Batters ADD COLUMN {col_name} {col_type}")
    conn.commit()

def format_start_times(iso_utc_str):
    now_utc = datetime.now(timezone.utc)
    if not iso_utc_str:
        return now_utc.isoformat(), now_utc.astimezone(ZoneInfo("America/New_York")).strftime("%I:%M %p EDT"), (now_utc - timedelta(minutes=30)).isoformat()

    dt_utc = datetime.fromisoformat(iso_utc_str.replace("Z", "+00:00"))
    dt_et = dt_utc.astimezone(ZoneInfo("America/New_York"))
    time_et_str = dt_et.strftime("%I:%M %p EDT")
    gatekeeper_utc = (dt_utc - timedelta(minutes=30)).isoformat()
    return dt_utc.isoformat(), time_et_str, gatekeeper_utc

def populate_slate_and_lineups():
    conn = get_db_connection()
    auto_repair_lineup_schema(conn)
    c = conn.cursor()

    now_utc = datetime.now(timezone.utc)
    today = now_utc.strftime("%Y-%m-%d")
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={today}&endDate={today}&hydrate=lineups,probablePitcher,status"

    try:
        res = requests.get(url, timeout=10).json()
    except Exception as e:
        print(f"[LINEUP VERIFIER ERROR] Failed to fetch schedule: {e}")
        conn.close()
        return

    total_batters = 0
    total_games = 0

    for date_entry in res.get("dates", []):
        for g in date_entry.get("games", []):
            pk = g["gamePk"]
            game_date_raw = g.get("gameDate")
            dt_utc_str, time_et_str, gatekeeper_str = format_start_times(game_date_raw)

            status_desc = g.get("status", {}).get("detailedState", "Scheduled")
            teams = g.get("teams", {})
            away_team = teams.get("away", {}).get("team", {}).get("name", "Away")
            home_team = teams.get("home", {}).get("team", {}).get("name", "Home")
            away_sp = teams.get("away", {}).get("probablePitcher", {}).get("fullName", "TBD")
            home_sp = teams.get("home", {}).get("probablePitcher", {}).get("fullName", "TBD")
            lineups = g.get("lineups", {})

            c.execute("""
                INSERT OR REPLACE INTO Daily_Lineups 
                (game_pk, away_team, home_team, away_sp, home_sp, away_pitcher, home_pitcher, 
                 lineup_status, status, game_datetime_utc, game_time_et, gatekeeper_trigger_utc, ingested_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (pk, away_team, home_team, away_sp, home_sp, away_sp, home_sp, 
                  status_desc, status_desc, dt_utc_str, time_et_str, gatekeeper_str, now_utc.isoformat()))
            total_games += 1

            for side in ["away", "home"]:
                team_name = away_team if side == "away" else home_team
                confirmed_lineup = lineups.get(f"{side}Players", [])

                if confirmed_lineup:
                    for slot, player in enumerate(confirmed_lineup[:9], 1):
                        p_name = player.get("fullName", f"Batter {slot}")
                        c.execute("""
                            INSERT OR REPLACE INTO Daily_Batters 
                            (game_pk, player_name, team_name, batting_order, is_starter, is_confirmed, game_datetime_utc, game_time_et)
                            VALUES (?, ?, ?, ?, 1, 1, ?, ?)
                        """, (pk, p_name, team_name, slot, dt_utc_str, time_et_str))
                        total_batters += 1
                else:
                    team_id = teams.get(side, {}).get("team", {}).get("id")
                    if team_id:
                        try:
                            roster_url = f"https://statsapi.mlb.com/api/v1/teams/{team_id}/roster?rosterType=active"
                            r_data = requests.get(roster_url, timeout=5).json()
                            hitters = [p["person"]["fullName"] for p in r_data.get("roster", []) if p.get("position", {}).get("code") != "1"]
                            for slot in range(1, 10):
                                name = hitters[slot - 1] if len(hitters) >= slot else f"{team_name} Hitter #{slot}"
                                c.execute("""
                                    INSERT OR REPLACE INTO Daily_Batters 
                                    (game_pk, player_name, team_name, batting_order, is_starter, is_confirmed, game_datetime_utc, game_time_et)
                                    VALUES (?, ?, ?, ?, 1, 0, ?, ?)
                                """, (pk, name, team_name, slot, dt_utc_str, time_et_str))
                                total_batters += 1
                        except Exception:
                            pass

    conn.commit()
    conn.close()
    print(f"[LINEUP INGESTION] Synchronized {total_games} matchups and {total_batters} batters with start times.")

if __name__ == "__main__":
    populate_slate_and_lineups()
