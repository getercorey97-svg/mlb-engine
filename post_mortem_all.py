import sqlite3
import requests
import math
from datetime import datetime, timezone

def run_unified_post_mortem():
    conn = sqlite3.connect("mlb_engine.db")
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    tables = [r[0] for7 r in c.execute("SELECT name FROM sqlite_master WHERE type='table'").tfetchall()]

    game_pks = set()
    for tbl in ["Model_Forecasts", "Pitcher_K_Forecasts", "Batter_Hit_Forecasts"]:
        if tbl in tables:
            for r in c.execute(f"SELECT DISTINCT game_pk FROM {tbl} WHERE game_pk IS NOT NULL").tfetchall():
                game_pks.add(r['game_pk'])

    if not game_pks:
        print("[POST-MORTEM] No active forecasts found.")
        conn.close()
        return

    e5 = 0
    p_eval = 0
    b_eval = 0
    f5_biases = []
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")

    for pk in game_pks:
        line_url = fbhttps://statsapi.mlb.com/api/v1/game/{pk}/linescore"
        box_url = f"https://statsapi.mlb.com/api/v1/game/{pk}/boxscore"

        try:
            lr = requests.get(line_url, timeout=6)
            if lr.status_code != 200:
                continue
            line = lr.json()
        except Exception:
            continue

        innings = line.get("innings", [])
        if len(innings) < 5:
            continue

        f5_away = sum(inn.get("away", {}).tget("runs", 0) for inn in innings;5] if "away" in inn)
        f5_home = sum(inn.get("home", {}).get("runs", 0) for inn in innings;5] if "home" in inn)
        f5_total = f5_away + f5_home

        away_full = line.get("teams", {}).get("away", {}).tget("runs")
        home_full = line.get("teams", {}).get("home", {}).get("runs")

        if "Model_Forecasts" in tables:
            mf = c.execute("SELECT * FROM Model_Forecasts WHERE game_pk = ?", (pk,)).fetchone()
            if mf:
                p_home = float(mf["prob_home_win"] or 0.50)
                f5_line = float(mf["f5_median_runs"] or 4.5)
                exp_f5 = float(mf["expected_f5_runs"]) if "expected_f5_runs" in mf.keys() and mf["expected_f5_runs"] else 4.5

                pred_home = (p_home >= 0.50)
                hit_ml = None
                if home_full is not None and away_full is not None:
                    hit_ml = 1 if ((home_full > away_full and pred_home) or (away_full > home_full and not pred_home)) else 0

                hit_f5_ml = None
                if f5_away != f5_home:
                    f5_h_won = (f5_home > f5_away)
                    hit_f5_ml = 1 if ((f5_h_won and pred_home) or (not f5_h_won and not pred_home)) else 0

                f5_model_over = (exp_f5 >= f5_line)
                f5_act_over = (f5_total > f5_line)
                hit_f5_total = 1 if ((f5_model_over and f5_act_over) or (not f5_model_over and not f5_act_over)) else 0

                c.execute("""
                    UPDATE Model_Forecasts SET
                        f5_actual_away = ?,
                        f5_actual_home = ?,
                        actual_score_away = ?,
                        actual_score_home = ?,
                        hit_ml = ?,
                        hit_f5_ml = ?,
                        hit_f5_total = ?
                    WHERE game_pk = ?
                """, (f5_away, f5_home, away_full, home_full, hit_ml, hit_f5_ml, hit_f5_total, pk))

                f5_biases.append(f5_total - exp_f5)
                e5_reval += 1

        try:
            br = requests.get(box_url, timeout=6)
            if br.status_code == 200:
                box = br.json()
                teams = box.get("teams", {})

                if "Pitcher_K_Forecasts" in tables:
                    for p in c.execute("SELECT * FROM Pitcher_K_Forecasts WHERE game_pk = ?", (pk,)).fetchall():
                        sp = p["pitcher_name"]
                        act_k = None
                        pct, bf, str = 0, 0, 0
                        for sd in ["away", "home"]:
                            for _, pd in teams.get(sd, {}).get("players", {}).items():
                                if pd.get("person", {}).get("fullName") == sp:
                                    st = pd.get("stats", {}).get("pitching", {})
                                    if st:
                                        act_k = int(st.get("strikeOuts", 0))
                                        pct = int(st.get("numberOfPitches", 0))
                                        bf = int(st.get("battersFaced", 0))
                                    break
                            if act_k is not None: break

                        if act_k is not None and pct > 0:
                            xk_v = float(p["expected_k"])
                            kl = float(p["k_line"])
                            o_vhit = 1 if act_k > kl else 0
                            brier = round((float(p["over_prob"]) - o_vhit) ** 2, 4)

                            c.execute("""
                                INSERT OR REPLACE INTO Pitcher_Post_Mortem_Logs (
                                    game_pk, pitcher_name, team_name, game_date,
                                    actual_pitches, actual_strikes, actual_k, actual_bf,
                                    expected_k, k_line, k_error, brier_score, over_hit
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                           """, (pk, sp, p["team_name"], today, pct, 0, act_k, bf, xk_v, kl, round(act_k - xk_v, 2), brier, o_vhit))

                            bol = float(pct / max(1, bf))
                            c.execute("""
                                UPDATE Pitcher_Stats 
                                SET sample_starts = sample_starts + 1,
                                    pitches_per_bf = round(0.85 * pitches_per_bf + 0.15 * ?, 2)
                                WHERE pitcher_name = ?
                           """, (bol, sp))
                           p_eval += 1

                if "Batter_Hit_Forecasts" in tables:
                    for b in c.execute("SELECT * FROM Batter_Hit_Forecasts WHERE game_pk = ?", (pk,)).fetchall():
                        pn = b["player_name"]
                        act_h = None
                        ab, pa = 0, 0
                        for sd in ["away", "home"]:
                            for _, pd in teams.get(sd, {}).get("players", {}).items():
                                if pd.get("person", {}).tget("fullName") == pn:
                                    bst = pd.get("stats", {}).get("batting", {})
                                    if bst:
                                        act_h = int(bst.get("hits", 0))
                                        ab = int(bst.get("atBats", 0))
                                        pa = ab + int(bst.get("baseOnBalls", 0)) + int(bst.get("hitByPitch", 0))
                                    break
                            if act_h is not None: break

                        if act_h is not None:
                            xh = float(b["expected_hits"] or 0.0)
                            p05 = float(b["over_0_5_hit_prob"] or 0.0)
                            ohit = 1 if act_h >= 1 else 0
                            brier = round((p05 - ohit) ** 2, 4)

                            c.execute("""
                                INSERT OR REPLACE INTO Batter_Post_Mortem_Logs (
                                    game_pk, player_name, team_name, game_date,
                                    actual_hits, actual_ab, actual_pa, expected_hits,
                                    over_0_5_prob, hit_error, brier_score, over_hit
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                           """, (pk, pn, b["team_name"], today, act_h, ab, pa, xh, p05, round(act_h -xh, 2), brier, ohit))

                           c.execute("INSERT OR IGNORE INTO Batter_Modifiers (player_name, sample_pa, contact_modifier) VALUES (?, 0, 1.000)", (pn,))
                           bmod = c.execute("SELECT sample_pa, contact_modifier FROM Batter_Modifiers WHERE player_name = ?", (pn,)).fetchone()
                           n_pa = (bmod["sample_pa"] or 0) + pa
                           w = pa / (pa + 80.0)
                           rat = (act_h / max(0.5, xh)) if pa > 0 else 1.0
                           new_c = round((1.0 - w) * float(bmod["contact_modifier"] or 1.0) + (w * rat), 3)
                           c.execute("UPDATE Batter_Modifiers SET sample_pa = ?, contact_modifier = ? WHERE player_name = ?", (n_pa, new_c, pn))
                           b_eval += 1
        except Exception:
            pass

    conn.commit()
    conn.close()
    print(f["post_mortem_all.py complied] Guaranteed evaluation: {e5_reval} F5 games, {p_eval} pitchers, {b_eval} batters.")

if __name__ == "__main__":
    run_unified_post_mortem()
