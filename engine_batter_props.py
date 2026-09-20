import sqlite3
import numpy as np
from scipy.stats import binom

LEAGUE_BA = 0.248

def get_db_connection():
    conn = sqlite3.connect("mlb_engine.db")
    conn.row_factory = sqlite3.Row
    return conn

def run_batter_props_engine():
    conn = get_db_connection()
    c = conn.cursor()

    tables = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    if "Daily_Batters" not in tables:
        print("[BATTER PROPS] Daily_Batters table not found.")
        conn.close()
        return

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

    batters = c.execute("""
        SELECT game_pk, player_name, team_name, batting_order, is_starter 
        FROM Daily_Batters 
        WHERE is_starter = 1 OR batting_order BETWEEN 1 AND 9
    """).fetchall()

    if not batters:
        print("[BATTER PROPS] No confirmed starters in Daily_Batters.")
        conn.close()
        return

    c.execute("DELETE FROM Batter_Hit_Forecasts")

    count = 0
    for b in batters:
        pk = b["game_pk"]
        name = b["player_name"]
        team = b["team_name"]
        order = int(b["batting_order"] or 5)

        # Ingest Two-Tier Decoupled Bayesian Priors
        c.execute("""
            INSERT OR IGNORE INTO Batter_Decoupled_Priors 
            (player_name, pa_contact_sample, contact_skill_mod, pa_babip_sample, babip_skill_mod) 
            VALUES (?, 0, 1.000, 0, 1.000)
        """, (name,))
        
        prior = c.execute("""
            SELECT contact_skill_mod, babip_skill_mod 
            FROM Batter_Decoupled_Priors 
            WHERE player_name = ?
        """, (name,)).fetchone()

        c_mod = float(prior["contact_skill_mod"]) if prior else 1.000
        b_mod = float(prior["babip_skill_mod"]) if prior else 1.000

        # Composite talent: Contact rate dominates (75%), BABIP regressed (25%)
        combined_hitter_mod = (c_mod * 0.75) + (b_mod * 0.25)

        # PA decay curve across lineup order
        projected_pa = max(3.1, 4.65 - (order * 0.12))
        projected_ab = projected_pa * 0.885

        true_ba = np.clip(LEAGUE_BA * combined_hitter_mod, 0.160, 0.360)
        exp_hits = round(projected_ab * true_ba, 2)

        # Binomial probability across discrete at-bats (rounded to 3 or 4)
        discrete_ab = int(round(projected_ab))
        p_over_0_5 = round(float(1.0 - binom.pmf(0, discrete_ab, true_ba)), 4)
        p_over_1_5 = round(float(1.0 - (binom.pmf(0, discrete_ab, true_ba) + binom.pmf(1, discrete_ab, true_ba))), 4)
        p_over_2_5 = round(float(1.0 - sum([binom.pmf(k, discrete_ab, true_ba) for k in range(3)])), 4)

        c.execute("""
            INSERT OR REPLACE INTO Batter_Hit_Forecasts (
                game_pk, player_name, team_name, batting_order,
                projected_pa, projected_ab, expected_hits,
                over_0_5_hit_prob, over_1_5_hit_prob, over_2_5_hit_prob
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            pk, name, team, order,
            round(projected_pa, 1), round(projected_ab, 1), exp_hits,
            p_over_0_5, p_over_1_5, p_over_2_5
        ))
        count += 1

    conn.commit()
    conn.close()
    print(f"[BATTER ENGINE] Generated {count} Decoupled Bayesian Batter Hit distributions.")

if __name__ == "__main__":
    run_batter_props_engine()
