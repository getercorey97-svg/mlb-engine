import sqlite3
import requests
import math
from datetime import datetime, timezone

def audit_props_slate():
    conn = sqlite3.connect("mlb_engine.db")
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    tables = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    game_pks = set()
    if "Batter_Hit_Forecasts" in tables:
        for r in c.execute("SELECT DISTINCT game_pk FROM Batter_Hit_Forecasts").fetchall():
            if r["game_pk"]: game_pks.add(r["game_pk"])
    if "Pitcher_K_Forecasts" in tables:
        for r in c.execute("SELECT DISTINCT game_pk FROM Pitcher_K_Forecasts").fetchall():
            if r["game_pk"]: game_pks.add(r["game_pk"])

    if not game_pks:
        print("[POST-MORTEM] No active prop slates found to audit.")
        conn.close()
        return

    today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    audited_batters = 0
    audited_pitchers = 0

    for pk in game_pks:
        url = f"https://statsapi.mlb.com/api/v1/game/{pk}/boxscore"
        try:
            res = requests.get(url, timeout=6)
            if res.status_code != 200: continue
            box = res.json()
        except Exception:
            continue

        teams = box.get("teams", {})

        # ==================================================================
        # 1. AUDIT BATTER HITS VIA TWO-TIER DECOUPLED BAYESIAN SHRINKAGE
        # ==================================================================
        if "Batter_Hit_Forecasts" in tables:
            batter_rows = c.execute("SELECT * FROM Batter_Hit_Forecasts WHERE game_pk = ?", (pk,)).fetchall()
            for b in batter_rows:
                player_name = b["player_name"]
                team_name = b["team_name"]

                actual_hits = None
                actual_ab = 0
                actual_pa = 0
                actual_so = 0

                for side in ["away", "home"]:
                    p_dict = teams.get(side, {}).get("players", {})
                    for _, pdata in p_dict.items():
                        if pdata.get("person", {}).get("fullName") == player_name:
                            b_stats = pdata.get("stats", {}).get("batting", {})
                            if b_stats:
                                actual_hits = int(b_stats.get("hits", 0))
                                actual_ab = int(b_stats.get("atBats", 0))
                                actual_so = int(b_stats.get("strikeOuts", 0))
                                actual_pa = actual_ab + int(b_stats.get("baseOnBalls", 0)) + int(b_stats.get("hitByPitch", 0))
                            break
                    if actual_hits is not None: break

                if actual_hits is not None and actual_pa > 0:
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

                    # Two-Tier Decoupled Bayesian Update:
                    # Tier 1: Contact Skill (Stabilizes at N ~ 120 PA)
                    # Tier 2: BABIP Luck (Stabilizes at N ~ 750 PA)
                    prior = c.execute("""
                        SELECT pa_contact_sample, contact_skill_mod, pa_babip_sample, babip_skill_mod 
                        FROM Batter_Decoupled_Priors 
                        WHERE player_name = ?
                    """, (player_name,)).fetchone()

                    n_c = (prior["pa_contact_sample"] or 0) + actual_pa
                    old_c_mod = float(prior["contact_skill_mod"] or 1.0)
                    
                    n_b = (prior["pa_babip_sample"] or 0) + actual_ab
                    old_b_mod = float(prior["babip_skill_mod"] or 1.0)

                    # Contact performance (avoiding strikeouts)
                    observed_contact_ratio = 1.15 if actual_so == 0 else (0.85 if actual_so >= 2 else 1.0)
                    w_c = actual_pa / (actual_pa + 120.0)
                    new_c_mod = round((1.0 - w_c) * old_c_mod + (w_c * observed_contact_ratio), 3)

                    # Ball-in-play BABIP luck performance
                    bip = max(1, actual_ab - actual_so)
                    observed_babip_ratio = actual_hits / max(0.3, bip * 0.290)
                    w_b = actual_ab / (actual_ab + 750.0)
                    new_b_mod = round((1.0 - w_b) * old_b_mod + (w_b * observed_babip_ratio), 3)

                    c.execute("""
                        UPDATE Batter_Decoupled_Priors 
                        SET pa_contact_sample = ?, contact_skill_mod = ?,
                            pa_babip_sample = ?, babip_skill_mod = ?,
                            last_game_pk = ?, last_updated = CURRENT_TIMESTAMP 
                        WHERE player_name = ?
                    """, (n_c, new_c_mod, n_b, new_b_mod, pk, player_name))

                    c.execute("""
                        INSERT INTO Prop_Learning_Calibration_Audit 
                        (eval_date, market_type, entity_name, game_pk, predicted_val, actual_val, line_val, error_delta, brier_score, pre_update_mod, post_update_mod)
                        VALUES (?, 'batter_hit', ?, ?, ?, ?, 0.5, ?, ?, ?, ?)
                    """, (today_str, player_name, pk, exp_hits, actual_hits, hit_error, brier, old_c_mod, new_c_mod))
                    audited_batters += 1

        # ==================================================================
        # 2. AUDIT PITCHER STRIKEOUTS VIA 1D KALMAN STATE-SPACE FILTER
        # ==================================================================
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
                    if actual_k is not None: break

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

                    # 1D Kalman Filter Measurement Update:
                    # Latent State theta: true strikeout modifier
                    # Observation z: actual_k / max(0.5, exp_k)
                    # Observation Variance R: inversely proportional to batters faced
                    c.execute("INSERT OR IGNORE INTO Pitcher_Kalman_State (pitcher_name, latent_k_modifier, variance_p, process_noise_q) VALUES (?, 1.000, 0.040, 0.0025)", (sp_name,))
                    k_row = c.execute("SELECT latent_k_modifier, variance_p, process_noise_q FROM Pitcher_Kalman_State WHERE pitcher_name = ?", (sp_name,)).fetchone()

                    theta_prior = float(k_row["latent_k_modifier"])
                    p_prior = float(k_row["variance_p"]) + float(k_row["process_noise_q"])

                    # Measurement noise R decreases when pitcher faces more batters
                    r_noise = max(0.020, 1.25 / max(10, act_bf))
                    z_obs = actual_k / max(0.5, exp_k)

                    # Kalman Gain calculation
                    k_gain = p_prior / (p_prior + r_noise)

                    # State update
                    theta_post = round(theta_prior + k_gain * (z_obs - theta_prior), 3)
                    p_post = round((1.0 - k_gain) * p_prior, 4)

                    c.execute("""
                        UPDATE Pitcher_Kalman_State 
                        SET latent_k_modifier = ?, variance_p = ?, last_game_pk = ?, last_updated = CURRENT_TIMESTAMP 
                        WHERE pitcher_name = ?
                    """, (theta_post, p_post, pk, sp_name))

                    c.execute("""
                        INSERT INTO Prop_Learning_Calibration_Audit 
                        (eval_date, market_type, entity_name, game_pk, predicted_val, actual_val, line_val, error_delta, brier_score, pre_update_mod, post_update_mod)
                        VALUES (?, 'pitcher_k', ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (today_str, sp_name, pk, exp_k, actual_k, k_line, k_error, brier, theta_prior, theta_post))
                    audited_pitchers += 1

    conn.commit()
    conn.close()
    print(f"[POST-MORTEM] Completed audit: {audited_batters} batters calibrated via Decoupled Bayes, {audited_pitchers} pitchers calibrated via Kalman Filter.")

if __name__ == "__main__":
    audit_props_slate()
