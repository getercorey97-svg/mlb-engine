from datetime import datetime, timezone
import sqlite3
import requests
import numpy as np
from scipy.stats import binom

LEAGUE_BA = 0.248
LEAGUE_OBP = 0.318

def get_db_connection():
    conn = sqlite3.connect("mlb_engine.db")
    conn.row_factory = sqlite3.Row
    return conn

def fetch_batter_live_stats(player_name):
    """Fallback to MLB Stats API for authentic season BA, OBP, and PA if local DB lacks them."""
    try:
        search_url = f"https://statsapi.mlb.com/api/v1/people/search?names={requests.utils.quote(player_name)}&sportIds=1"
        res = requests.get(search_url, timeout=4).json()
        people = res.get("people", [])
        if not people:
            return None
        pid = people[0]["id"]

        stat_url = f"https://statsapi.mlb.com/api/v1/people/{pid}/stats?stats=season&group=hitting"
        s_res = requests.get(stat_url, timeout=4).json()
        splits = s_res.get("stats", [{}])[0].get("splits", [])
        if not splits:
            return None
        s = splits[0].get("stat", {})

        ab = int(s.get("atBats", 0))
        h = int(s.get("hits", 0))
        obp = float(s.get("obp", 0.318))
        avg = float(s.get("avg", 0.248))
        so = int(s.get("strikeOuts", 0))

        if ab > 40:
            return {
                "ba": float(np.clip(avg, 0.160, 0.370)),
                "obp": float(np.clip(obp, 0.230, 0.440)),
                "k_rate": float(np.clip(so / max(1, ab), 0.08, 0.38))
            }
    except Exception:
        pass
    return None

def calculate_log5_hit(batter_ba, opp_pitcher_ba_allowed=0.248, lg_ba=LEAGUE_BA):
    p = float(batter_ba)
    b = float(opp_pitcher_ba_allowed)
    num = (p * b) / lg_ba
    denom = num + (((1.0 - p) * (1.0 - b)) / (1.0 - lg_ba))
    return float(np.clip(num / denom, 0.140, 0.390))

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
        SELECT game_pk, player_name, team_name, batting_order 
        FROM Daily_Batters
    """).fetchall()

    if not batters:
        print("[BATTER PROPS] No starting batters found in Daily_Batters.")
        conn.close()
        return

    c.execute("DELETE FROM Batter_Hit_Forecasts")

    # Map opposing starter quality per game_pk
    opp_sp_quality = {}
    if "Pitcher_K_Forecasts" in tables:
        for r in c.execute("SELECT game_pk, team_name, expected_k FROM Pitcher_K_Forecasts").fetchall():
            # Higher strikeout expectation lowers opponent batting average
            exp_k = float(r["expected_k"] or 4.5)
            ba_allowed = float(np.clip(0.250 - ((exp_k - 4.5) * 0.008), 0.200, 0.295))
            opp_sp_quality[(r["game_pk"], r["team_name"])] = ba_allowed

    total_batters = 0
    for b in batters:
        pk = b["game_pk"]
        name = b["player_name"]
        team = b["team_name"]
        order = int(b["batting_order"] or 5)

        # 1. Check local Decoupled Priors table
        c.execute("""
            INSERT OR IGNORE INTO Batter_Decoupled_Priors 
            (player_name, pa_contact_sample, contact_skill_mod, pa_babip_sample, babip_skill_mod) 
            VALUES (?, 0, 1.000, 0, 1.000)
        """, (name,))

        prior = c.execute("SELECT contact_skill_mod, babip_skill_mod, pa_contact_sample FROM Batter_Decoupled_Priors WHERE player_name = ?", (name,)).fetchone()
        c_mod = float(prior["contact_skill_mod"]) if prior else 1.000
        b_mod = float(prior["babip_skill_mod"]) if prior else 1.000
        pa_sample = int(prior["pa_contact_sample"]) if prior else 0

        # 2. Fetch authentic MLB stats if player has low local database sample
        live_stats = None
        if pa_sample < 30 or (c_mod == 1.000 and b_mod == 1.000):
            live_stats = fetch_batter_live_stats(name)

        if live_stats:
            base_ba = live_stats["ba"]
            # Calibrate prior modifier to match authentic performance
            c_mod = round(base_ba / LEAGUE_BA, 3)
            c.execute("UPDATE Batter_Decoupled_Priors SET contact_skill_mod = ? WHERE player_name = ?", (c_mod, name))
        else:
            base_ba = np.clip(LEAGUE_BA * ((c_mod * 0.75) + (b_mod * 0.25)), 0.170, 0.350)

        # 3. Matchup adjustment against opposing starting pitcher
        opp_ba_allowed = 0.248
        for (g_pk, p_team), ba_val in opp_sp_quality.items():
            if g_pk == pk and p_team != team:
                opp_ba_allowed = ba_val
                break

        true_ba = calculate_log5_hit(base_ba, opp_ba_allowed)

        # 4. Projected plate appearances decayed by lineup slot
        projected_pa = max(3.1, round(4.65 - (order * 0.12), 2))
        projected_ab = round(projected_pa * 0.885, 2)
        exp_hits = round(projected_ab * true_ba, 2)

        # 5. Exact discrete binomial probabilities
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
            projected_pa, projected_ab, exp_hits,
            p_over_0_5, p_over_1_5, p_over_2_5
        ))
        total_batters += 1

    conn.commit()
    conn.close()
    print(f"[SUCCESS] Synthesized {total_batters} calibrated Batter Hit lines with individual player statistics.")

if __name__ == "__main__":
    run_batter_props_engine()
