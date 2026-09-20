import sqlite3
import numpy as np
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

def get_db_connection():
    conn = sqlite3.connect("mlb_engine.db")
    conn.row_factory = sqlite3.Row
    return conn

def auto_repair_batter_forecast_schema(conn):
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS Batter_Hit_Forecasts (
            game_pk INTEGER,
            player_name TEXT,
            team_name TEXT,
            batting_order INTEGER,
            projected_pa REAL,
            projected_ab REAL,
            expected_hits REAL,
            over_0_5_hit_prob REAL,
            over_1_5_hit_prob REAL,
            over_2_5_hit_prob REAL,
            game_date TEXT,
            game_datetime_utc TEXT,
            game_time_et TEXT,
            forecast_timestamp_utc TEXT,
            PRIMARY KEY (game_pk, player_name)
        )
    """)
    existing = [r[1] for r in c.execute("PRAGMA table_info(Batter_Hit_Forecasts)").fetchall()]
    required = [
        ("game_date", "TEXT"),
        ("game_datetime_utc", "TEXT"),
        ("game_time_et", "TEXT"),
        ("forecast_timestamp_utc", "TEXT")
    ]
    for col_name, col_type in required:
        if col_name not in existing:
            c.execute(f"ALTER TABLE Batter_Hit_Forecasts ADD COLUMN {col_name} {col_type}")
    conn.commit()

def calculate_binomial_hit_probs(proj_ab, hit_prob_per_ab):
    n = max(1, int(round(proj_ab)))
    p = np.clip(hit_prob_per_ab, 0.05, 0.45)
    
    p0 = (1.0 - p) ** n
    p1 = n * p * ((1.0 - p) ** (n - 1)) if n >= 1 else 0.0
    p2 = (n * (n - 1) / 2.0) * (p ** 2) * ((1.0 - p) ** (n - 2)) if n >= 2 else 0.0

    p_over_0_5 = round(float(1.0 - p0), 4)
    p_over_1_5 = round(float(max(0.0, 1.0 - (p0 + p1))), 4)
    p_over_2_5 = round(float(max(0.0, 1.0 - (p0 + p1 + p2))), 4)
    return p_over_0_5, p_over_1_5, p_over_2_5

def run_batter_props_engine():
    conn = get_db_connection()
    auto_repair_batter_forecast_schema(conn)
    c = conn.cursor()

    batters = c.execute("""
        SELECT game_pk, player_name, team_name, batting_order, 
               COALESCE(game_datetime_utc, '') as game_datetime_utc, 
               COALESCE(game_time_et, 'TBD') as game_time_et
        FROM Daily_Batters
        ORDER BY game_pk, batting_order
    """).fetchall()

    if not batters:
        print("[BATTER ENGINE] No batters found in Daily_Batters to process.")
        conn.close()
        return

    now_utc = datetime.now(timezone.utc)
    today_str = now_utc.strftime("%Y-%m-%d")
    now_utc_str = now_utc.isoformat()

    c.execute("DELETE FROM Batter_Hit_Forecasts")

    generated_count = 0
    for b in batters:
        pk = b["game_pk"]
        player_name = b["player_name"]
        team_name = b["team_name"]
        slot = b["batting_order"] or 5
        game_dt_utc = b["game_datetime_utc"]
        game_time_et = b["game_time_et"]

        # Pull persistent Empirical Bayes Contact Modifier if logged
        b_mod = c.execute("SELECT contact_modifier FROM Batter_Modifiers WHERE player_name = ?", (player_name,)).fetchone()
        c_mod = float(b_mod["contact_modifier"]) if b_mod and b_mod["contact_modifier"] else 1.000

        # Project PA & AB based on lineup slot hierarchy
        proj_pa = round(max(3.6, 4.65 - (0.11 * slot)), 2)
        proj_ab = round(proj_pa * 0.89, 2)

        # Baseline MLB contact rate scaled by contact modifier
        hit_rate = np.clip(0.248 * c_mod, 0.120, 0.380)
        exp_hits = round(proj_ab * hit_rate, 2)

        p_0_5, p_1_5, p_2_5 = calculate_binomial_hit_probs(proj_ab, hit_rate)

        # Exact 14 columns matching exact 14 parameter bindings
        c.execute("""
            INSERT OR REPLACE INTO Batter_Hit_Forecasts (
                game_pk, player_name, team_name, batting_order,
                projected_pa, projected_ab, expected_hits,
                over_0_5_hit_prob, over_1_5_hit_prob, over_2_5_hit_prob,
                game_date, game_datetime_utc, game_time_et, forecast_timestamp_utc
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pk, player_name, team_name, slot,
            proj_pa, proj_ab, exp_hits,
            p_0_5, p_1_5, p_2_5,
            today_str, game_dt_utc, game_time_et, now_utc_str
        ))
        generated_count += 1

    conn.commit()
    conn.close()
    print(f"[SUCCESS] Synthesized {generated_count} calibrated Batter Hit lines with start times.")

if __name__ == "__main__":
    run_batter_props_engine()
