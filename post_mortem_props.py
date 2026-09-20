import sqlite3
import requests
import math
from datetime import datetime, timezone

def audit_props_slate():
    conn = sqlite3.connect("mlb_engine.db")
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    # Pre-flight Check: Ensure columns exist in Batter_Modifiers
    bm_cols = [r[1] for r in c.execute("PRAGMA table_info(Batter_Modifiers)").fetchall()]
    if "sample_pa" not in bm_cols:
        c.execute("ALTER TABLE Batter_Modifiers ADD COLUMN sample_pa INTEGER DEFAULT 0")
    if "contact_modifier" not in bm_cols:
        c.execute("ALTER TABLE Batter_Modifiers ADD COLUMN contact_modifier REAL DEFAULT 1.000")
    if "last_updated" not in bm_cols:
        c.execute("ALTER TABLE Batter_Modifiers ADD COLUMN last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP")
    conn.commit()

    # Determine unique game IDs to audit
    tables = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    game_pks = set()
    if "Batter_Hit_Forecasts" in tables:
        for r in c.execute("SELECT DISTINCT game_pk FROM Batter_Hit_Forecasts").fetchall():
            if r["game_pk"]:
                game_pks.add(r["game_pk"])
    if "Pitcher_K_Forecasts" in tables:
        for r in c.execute("SELECT DISTINCT game_pk FROM Pitcher_K_Forecasts").fetchall():
            if r["game_pk"]:
                game_pks.add(r["game_pk"])

    if not game_pks:
        print("[POST-MORTEM] No active prop records found to audit.")
        conn.close()
        return

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    audited_batters = 0
    audited_pitchers = 0

    ps_cols = [r[1] for r in c.execute("PRAGMA table_info(Pitcher_Stats)").fetchall()]
    p_name_col = "pitcher_name" if "pitcher_name" in ps_cols else ("player_name" if "player_name" in ps_cols else "last_name")

    for pk in game_pks:
        url = f"https://statsapi.mlb.com/api/v1/game/{pk}/boxscore"
        try:
            res = requests.get(url, timeout=6)
            if res.status_code != 200:
                continue
            box = res.json()
        except Exception:
            continue

        teams = box.get("teams", {})

        # 1. Audit Batter Hits
        if "Batter_Hit_Forecasts" in tables:
            batter_rows = c.execute("SELECT * FROM Batter_Hit_Forecasts WHERE game_pk = ?", (pk,)).fetchall()
            for b in batter_rows:
                player_name = b["player_name"]
                team_name = b["team_name"]

                actual_hits = None
                actual_ab = 0
                actual_pa = 0

                for side in ["away", "home"]:
                    p_dict = teams.get(side, {}).get("players", {})
                    for _, pdata in p_dict.items():
                        if pdata.get("person", {}).get("fullName") == player_name:
                            b_stats = pdata.get("stats", {}).get("batting", {})
                            if b_stats:
                                actual_hits = int(b_stats.get("hits", 0))
                                actual_ab = int(b_stats.get("atBats", 0))
                                actual_pa = actual_ab + int(b_stats.get("baseOnBalls", 0)) + int(b_stats.get("hitByPitch", 0))
                            break
                    if actual_hits is not None:
                        break

                if actual_hits is not None:
                    exp_hits = float(b["expected_hits"] or 0.0)
                    p_over = float(b["over_0_5_hit_prob"] or 0.0)
                    hit_error = round(actual_hits - exp_hits, 2)
                    over_hit = 1 if actual_hits >= 1 else 0
                    brier = round((p_over - over_hit) ** 2, 4)

                    c.execute("""
                        INSERT OR REPLACE INTO Batter_Post_Mortem_Logs (
                            game_pk, player_name, team_name, game_date,
                            actual_hits, actual_ab, actual_pa, expected_hits,
                            over_0_5_prob, hit_error, brier_score, over_hit
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        pk, player_name, team_name, today_str,
                        actual_hits, actual_ab, actual_pa, exp_hits,
                        p_over, hit_error, brier, over_hit
                    ))

                    # Empirical Bayes Updates
                    c.execute("INSERT OR IGNORE INTO Batter_Modifiers (player_name, sample_pa, contact_modifier) VALUES (?, 0, 1.000)", (player_name,))
                    b_mod = c.execute("SELECT sample_pa, contact_modifier FROM Batter_Modifiers WHERE player_name = ?", (player_name,)).fetchone()
                    n = (b_mod["sample_pa"] or 0) + actual_pa
                    old_mod = b_mod["contact_modifier"] or 1.000

                    observed_ratio = (actual_hits / max(0.5, exp_hits)) if actual_pa > 0 else 1.0
                    weight = actual_pa / (actual_pa + 80.0)
                    new_mod = round((1.0 - weight) * old_mod + (weight * observed_ratio), 3)

                    c.execute("""
                        UPDATE Batter_Modifiers 
                        SET sample_pa = ?, contact_modifier = ?, last_updated = CURRENT_TIMESTAMP 
                        WHERE player_name = ?
                    """, (n, new_mod, player_name))
                    audited_batters += 1

        # 2. Audit Pitcher Ks
        if "Pitcher_K_Forecasts" in tables:
            pitcher_rows = c.execute("SELECT * FROM Pitcher_K_Forecasts WHERE game_pk = ?", (pk,)).fetchall()
            for p in pitcher_rows:
                sp_name = p["pitcher_name"]
                actual_k = None
                act_pitches = 0
                act_strikes = 0
                act_bf = 0

                for side in ["away", "home"]:
                    p_dict = teams.get(side, {}).get("players", {})
                    for _, pdata in p_dict.items():
                        if pdata.get("person", {}).get("fullName") == sp_name:
                            pitch_stats = pdata.get("stats", {}).get("pitching", {})
                            if pitch_stats:
                                act_pitches = int(pitch_stats.get("numberOfPitches", 0))
                                act_strikes = int(pitch_stats.get("strikes", 0))
                                actual_k = int(pitch_stats.get("strikeOuts", 0))
                                act_bf = int(pitch_stats.get("battersFaced", 0))
                            break
                    if actual_k is not None:
                        break

                if actual_k is not None and act_pitches > 0:
                    exp_k = float(p["expected_k"])
                    k_line = float(p["k_line"])
                    p_over = float(p["over_prob"])

                    k_error = round(actual_k - exp_k, 2)
                    over_hit = 1 if actual_k > k_line else 0
                    brier = round((p_over - over_hit) ** 2, 4)

                    c.execute("""
                        INSERT OR REPLACE INTO Pitcher_Post_Mortem_Logs (
                            game_pk, pitcher_name, team_name, game_date,
                            actual_pitches, actual_strikes, actual_whiffs,
                            actual_k, actual_bf, expected_k, k_line,
                            k_error, brier_score, over_hit
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        pk, sp_name, p["team_name"], today_str,
                        act_pitches, act_strikes, 0,
                        actual_k, act_bf, exp_k, k_line,
                        k_error, brier, over_hit
                    ))

                    # EWMA Calibration on Pitcher_Stats
                    sp_row = c.execute(f"""
                        SELECT k_modifier, sample_starts, pitches_per_bf 
                        FROM Pitcher_Stats 
                        WHERE {p_name_col} = ? OR ? LIKE '%' || {p_name_col}
                        LIMIT 1
                    """, (sp_name, sp_name)).fetchone()

                    if sp_row:
                        starts = sp_row["sample_starts"] or 1
                        curr_k_mod = sp_row["k_modifier"] or 1.000
                        curr_p_bf = sp_row["pitches_per_bf"] or 3.90

                        alpha = 1.0 / math.sqrt(starts + 1)
                        k_ratio = actual_k / max(0.5, exp_k)
                        new_k_mod = round((1.0 - alpha) * curr_k_mod + (alpha * k_ratio), 3)

                        game_p_bf = act_pitches / max(1, act_bf)
                        new_p_bf = round((0.85 * curr_p_bf) + (0.15 * game_p_bf), 2)

                        c.execute(f"""
                            UPDATE Pitcher_Stats 
                            SET k_modifier = ?, pitches_per_bf = ?, sample_starts = sample_starts + 1 
                            WHERE {p_name_col} = ? OR ? LIKE '%' || {p_name_col}
                        """, (new_k_mod, new_p_bf, sp_name, sp_name))
                    audited_pitchers += 1

    conn.commit()
    conn.close()
    print(f"[SUCCESS] Post-mortem prop audit complete ({audited_batters} batters, {audited_pitchers} pitchers updated).")

if __name__ == "__main__":
    audit_props_slate()
