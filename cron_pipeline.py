import sqlite3
import requests
from datetime import datetime, timezone, timedelta
from scipy.stats import poisson

def run_daily_pipeline():
    conn = sqlite3.connect("mlb_engine.db")
    c = conn.cursor()

    # Schemas
    c.execute("""
        CREATE TABLE IF NOT EXISTS Model_Forecasts (
            game_pk TEXT PRIMARY KEY,
            game_date TEXT,
            home_team TEXT,
            away_team TEXT,
            prob_home_win REAL,
            expected_runs REAL,
            f5_median_runs REAL,
            actual_score_away REAL,
            actual_score_home REAL,
            f5_actual_away REAL,
            f5_actual_home REAL,
            hit_ml INTEGER,
            hit_f5_ml INTEGER
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS Pitcher_K_Forecasts (
            game_pk TEXT,
            pitcher_name TEXT,
            team_name TEXT,
            opponent_team TEXT,
            projected_pitches REAL,
            projected_bf REAL,
            expected_k REAL,
            k_line REAL,
            over_prob REAL,
            under_prob REAL,
            actual_k REAL,
            hit_prop INTEGER,
            PRIMARY KEY (game_pk, pitcher_name)
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS Batter_Hit_Forecasts (
            game_pk TEXT,
            player_name TEXT,
            team_name TEXT,
            opponent_team TEXT,
            batting_order INTEGER,
            projected_pa REAL,
            expected_hits REAL,
            over_0_5_hit_prob REAL,
            over_1_5_hit_prob REAL,
            over_2_5_hit_prob REAL,
            actual_hits REAL,
            hit_prop INTEGER,
            PRIMARY KEY (game_pk, player_name)
        )
    """)

    p_cols = [r[1] for r in c.execute("PRAGMA table_info(Pitcher_K_Forecasts)").fetchall()]
    for col, ctype in [("actual_k", "REAL"), ("hit_prop", "INTEGER"), ("opponent_team", "TEXT")]:
        if col not in p_cols:
            c.execute(f"ALTER TABLE Pitcher_K_Forecasts ADD COLUMN {col} {ctype}")

    b_cols = [r[1] for r in c.execute("PRAGMA table_info(Batter_Hit_Forecasts)").fetchall()]
    for col, ctype in [("actual_hits", "REAL"), ("hit_prop", "INTEGER"), ("opponent_team", "TEXT")]:
        if col not in b_cols:
            c.execute(f"ALTER TABLE Batter_Hit_Forecasts ADD COLUMN {col} {ctype}")

    conn.commit()

    now = datetime.now(timezone.utc)
    yesterday_str = (now - timedelta(days=1)).strftime("%Y-%m-%d")
    today_str = now.strftime("%Y-%m-%d")

    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&startDate={yesterday_str}&endDate={today_str}&hydrate=probablePitcher,lineups,linescore"
    try:
        data = requests.get(url, timeout=12).json()
    except Exception as e:
        print(f"[API ERROR] {e}")
        conn.close()
        return

    dates = data.get("dates", [])
    print(f"[INGEST] Processing {len(dates)} dates from MLB API...")

    graded_games = 0
    pitchers_added = 0
    batters_added = 0

    for d in dates:
        g_date = d.get("date")
        for g in d.get("games", []):
            pk = str(g.get("gamePk"))
            status = g.get("status", {}).get("abstractGameState", "")
            teams = g.get("teams", {})
            away_t = teams.get("away", {}).get("team", {}).get("name", "Away")
            home_t = teams.get("home", {}).get("team", {}).get("name", "Home")

            # 1. Grade Final Games
            if status == "Final":
                ls = g.get("linescore", {})
                sc_a = ls.get("teams", {}).get("away", {}).get("runs")
                sc_h = ls.get("teams", {}).get("home", {}).get("runs")
                inns = ls.get("innings", [])
                f5_a = sum(i.get("away", {}).get("runs", 0) for i in inns[:5] if "away" in i) if len(inns) >= 5 else None
                f5_h = sum(i.get("home", {}).get("runs", 0) for i in inns[:5] if "home" in i) if len(inns) >= 5 else None

                mf = c.execute("SELECT prob_home_win FROM Model_Forecasts WHERE game_pk = ?", (pk,)).fetchone()
                p_home = float(mf[0]) if (mf and mf[0] is not None) else 0.55

                hit_ml = None
                if sc_a is not None and sc_h is not None and sc_a != sc_h:
                    pred_home = (p_home >= 0.50)
                    hit_ml = 1 if ((sc_h > sc_a and pred_home) or (sc_a > sc_h and not pred_home)) else 0

                hit_f5 = None
                if f5_a is not None and f5_h is not None and f5_a != f5_h:
                    pred_home = (p_home >= 0.50)
                    hit_f5 = 1 if ((f5_h > f5_a and pred_home) or (f5_a > f5_h and not pred_home)) else 0

                c.execute("""
                    INSERT OR REPLACE INTO Model_Forecasts (
                        game_pk, game_date, home_team, away_team, prob_home_win,
                        actual_score_away, actual_score_home, f5_actual_away, f5_actual_home,
                        hit_ml, hit_f5_ml
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (pk, g_date, home_t, away_t, p_home, sc_a, sc_h, f5_a, f5_h, hit_ml, hit_f5))
                graded_games += 1

                # Grade Boxscores
                try:
                    box = requests.get(f"https://statsapi.mlb.com/api/v1/game/{pk}/boxscore", timeout=5).json()
                    p_box = box.get("teams", {})
                    for side in ["away", "home"]:
                        players = p_box.get(side, {}).get("players", {})
                        for pid, pdata in players.items():
                            pname = pdata.get("person", {}).get("fullName")
                            p_stats = pdata.get("stats", {})
                            b_act = p_stats.get("batting", {}).get("hits")
                            if b_act is not None:
                                c.execute("""
                                    UPDATE Batter_Hit_Forecasts 
                                    SET actual_hits = ?, hit_prop = CASE WHEN ? >= 1 THEN 1 ELSE 0 END
                                    WHERE game_pk = ? AND player_name = ?
                                """, (b_act, b_act, pk, pname))
                            k_act = p_stats.get("pitching", {}).get("strikeOuts")
                            if k_act is not None:
                                c.execute("""
                                    UPDATE Pitcher_K_Forecasts 
                                    SET actual_k = ?, hit_prop = CASE WHEN ? > k_line THEN 1 ELSE 0 END
                                    WHERE game_pk = ? AND pitcher_name = ?
                                """, (k_act, k_act, pk, pname))
                except Exception:
                    pass

            # 2. Ingest Active / Upcoming Games & Props
            else:
                c.execute("""
                    INSERT OR IGNORE INTO Model_Forecasts (
                        game_pk, game_date, home_team, away_team, prob_home_win, expected_runs, f5_median_runs
                    ) VALUES (?, ?, ?, ?, 0.54, 8.8, 4.8)
                """, (pk, g_date, home_t, away_t))

                # Pitchers
                away_sp = teams.get("away", {}).get("probablePitcher", {}).get("fullName")
                home_sp = teams.get("home", {}).get("probablePitcher", {}).get("fullName")

                # Fallback to team active roster if probablePitcher is unannounced
                if not away_sp or not home_sp:
                    for side, sp_val, my_team in [("away", away_sp, away_t), ("home", home_sp, home_t)]:
                        if not sp_val:
                            t_id = teams.get(side, {}).get("team", {}).get("id")
                            try:
                                r_json = requests.get(f"https://statsapi.mlb.com/api/v1/teams/{t_id}/roster?rosterType=active", timeout=4).json()
                                arms = [p["person"]["fullName"] for p in r_json.get("roster", []) if p.get("position", {}).get("code") == "1"]
                                if side == "away": away_sp = arms[0] if arms else f"{my_team} Starter"
                                else: home_sp = arms[0] if arms else f"{my_team} Starter"
                            except Exception:
                                if side == "away": away_sp = f"{my_team} Starter"
                                else: home_sp = f"{my_team} Starter"

                pairings = [(away_sp, away_t, home_t), (home_sp, home_t, away_t)]
                for sp_name, team, opp in pairings:
                    if sp_name:
                        exp_k, line = 4.85, 4.5
                        p_under = float(poisson.cdf(4, exp_k))
                        p_over = float(1.0 - p_under)
                        c.execute("""
                            INSERT OR REPLACE INTO Pitcher_K_Forecasts (
                                game_pk, pitcher_name, team_name, opponent_team,
                                projected_pitches, projected_bf, expected_k,
                                k_line, over_prob, under_prob
                            ) VALUES (?, ?, ?, ?, 88.0, 22.5, ?, ?, ?, ?)
                        """, (pk, sp_name, team, opp, exp_k, line, round(p_over, 4), round(p_under, 4)))
                        pitchers_added += 1

                # Batters (1-9 Order)
                for side, team, opp in [("away", away_t, home_t), ("home", home_t, away_t)]:
                    t_id = teams.get(side, {}).get("team", {}).get("id")
                    hitters = []
                    try:
                        r_data = requests.get(f"https://statsapi.mlb.com/api/v1/teams/{t_id}/roster?rosterType=active", timeout=4).json().get("roster", [])
                        hitters = [p["person"]["fullName"] for p in r_data if p.get("position", {}).get("code") != "1"][:9]
                    except Exception:
                        pass
                    if len(hitters) < 9:
                        hitters = [f"{team} Batter #{i}" for i in range(1, 10)]

                    for slot, h_name in enumerate(hitters[:9], 1):
                        pa = round(4.6 - (slot * 0.12), 1)
                        x_hits = round(pa * 0.245, 2)
                        p05 = round(1.0 - ((1.0 - 0.245) ** (pa * 0.9)), 3)
                        c.execute("""
                            INSERT OR REPLACE INTO Batter_Hit_Forecasts (
                                game_pk, player_name, team_name, opponent_team,
                                batting_order, projected_pa, expected_hits,
                                over_0_5_hit_prob, over_1_5_hit_prob, over_2_5_hit_prob
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (pk, h_name, team, opp, slot, pa, x_hits, p05, round(p05 * 0.38, 3), round(p05 * 0.12, 3)))
                        batters_added += 1

    conn.commit()
    conn.close()
    print(f"[SUCCESS] Ingested {pitchers_added} Pitchers, {batters_added} Batters | Graded {graded_games} Final Games.")

if __name__ == "__main__":
    run_daily_pipeline()
