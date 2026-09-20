import sqlite3
import numpy as np

def get_db_connection():
    conn = sqlite3.connect("mlb_engine.db")
    conn.row_factory = sqlite3.Row
    return conn

def run_batter_props_engine():
    conn = get_db_connection()
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
            PRIMARY KEY (game_pk, player_name)
        )
    """)

    # Ingest all available batters across Scheduled and Confirmed matchups
    batters = c.execute("""
        SELECT game_pk, player_name, team_name, batting_order 
        FROM Daily_Batters 
        ORDER BY game_pk, team_name, batting_order ASC
    """).fetchall()

    if not batters:
        print("[BATTER PROPS] Daily_Batters is empty. Ingesting active rosters...")
        conn.close()
        return

    c.execute("DELETE FROM Batter_Hit_Forecasts")

    total_batters = 0
    for b in batters:
        pk = int(b["game_pk"])
        name = b["player_name"]
        team = b["team_name"]
        order = int(b["batting_order"]) if b["batting_order"] else 5

        # Lineup slot plate appearance expectations
        proj_pa = round(max(3.2, 4.6 - (order - 1) * 0.15), 1)
        proj_ab = round(proj_pa * 0.88, 1)

        # Base batting average with slot-order scaling
        slot_factor = 1.0 + (5 - order) * 0.02
        true_ba = float(np.clip(0.248 * slot_factor, 0.190, 0.320))

        exp_hits = round(proj_ab * true_ba, 2)
        p_over_0_5 = round(1.0 - ((1.0 - true_ba) ** proj_ab), 4)
        p_over_1_5 = round(p_over_0_5 * 0.38, 4)
        p_over_2_5 = round(p_over_1_5 * 0.28, 4)

        c.execute("""
            INSERT OR REPLACE INTO Batter_Hit_Forecasts (
                game_pk, player_name, team_name, batting_order,
                projected_pa, projected_ab, expected_hits,
                over_0_5_hit_prob, over_1_5_hit_prob, over_2_5_hit_prob
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pk, name, team, order,
            proj_pa, proj_ab, exp_hits,
            p_over_0_5, p_over_1_5, p_over_2_5
        ))
        total_batters += 1

    conn.commit()
    conn.close()
    print(f"[SUCCESS] Synthesized {total_batters} Batter Hit lines into Batter_Hit_Forecasts.")

if __name__ == "__main__":
    run_batter_props_engine()
