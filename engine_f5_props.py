import sqlite3
import numpy as np

def run_f5_and_props_engine():
    print("Initializing Phase 4B: Secondary Engine (First 5 & Pitcher Props)...")
    
    conn = sqlite3.connect('mlb_engine.db', timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=10000;")
    cursor = conn.cursor()
    
    # Failsafe Schema Execution & Migrations
    cursor.executescript('''
    CREATE TABLE IF NOT EXISTS F5_Forecasts (
        game_pk INTEGER PRIMARY KEY, away_team TEXT, home_team TEXT, away_starter TEXT, home_starter TEXT, 
        f5_away_prob REAL, f5_home_prob REAL, f5_tie_prob REAL, f5_exp_away_runs REAL, f5_exp_home_runs REAL, f5_total_runs REAL
    );
    CREATE TABLE IF NOT EXISTS Pitcher_Props (
        game_pk INTEGER, pitcher_name TEXT, team_name TEXT, projected_outs REAL, projected_strikeouts REAL, 
        over_4_5_k_prob REAL, over_5_5_k_prob REAL, over_6_5_k_prob REAL, PRIMARY KEY (game_pk, pitcher_name)
    );
    CREATE TABLE IF NOT EXISTS Pitcher_Modifiers (
        pitcher_name TEXT PRIMARY KEY,
        k_modifier REAL DEFAULT 1.0,
        f5_run_modifier REAL DEFAULT 1.0,
        appearance_count INTEGER DEFAULT 0,
        last_updated TEXT
    );
    CREATE TABLE IF NOT EXISTS Dynamic_Modifiers (
        team_name TEXT PRIMARY KEY,
        offensive_modifier REAL DEFAULT 1.0,
        pitching_modifier REAL DEFAULT 1.0,
        appearance_count INTEGER DEFAULT 0,
        last_updated TEXT
    );
    ''')

    for table, col in [("Daily_Lineups", "uv_modifier REAL DEFAULT 1.0"),
                       ("Pitcher_Modifiers", "appearance_count INTEGER DEFAULT 0"),
                       ("Dynamic_Modifiers", "appearance_count INTEGER DEFAULT 0")]:
        try:
            cursor.execute(f"ALTER TABLE {table} ADD COLUMN {col}")
        except sqlite3.OperationalError:
            pass

    try:
        cursor.execute('''
            SELECT d.game_pk, d.away_team, d.home_team, d.away_pitcher, d.home_pitcher, d.air_density, d.uv_modifier, COALESCE(u.run_modifier, 1.0)
            FROM Daily_Lineups d LEFT JOIN Daily_Umpires u ON d.game_pk = u.game_pk
            WHERE d.status != 'Final' AND d.game_pk NOT IN (SELECT game_pk FROM Post_Match_Analysis)
        ''')
        matchups = cursor.fetchall()
    except Exception as e:
        print(f"CRITICAL SQL ERROR in F5 Engine: {e}")
        conn.close()
        return

    if not matchups: 
        print("No matchups found for F5 Engine.")
        conn.close()
        return

    # Ingest learning weights with appearance tracking for Bayesian Shrinkage
    pitcher_mods = {r[0]: (r[1], r[2], r[3]) for r in cursor.execute(
        "SELECT pitcher_name, COALESCE(k_modifier, 1.0), COALESCE(f5_run_modifier, 1.0), COALESCE(appearance_count, 0) FROM Pitcher_Modifiers"
    ).fetchall()}
    team_mods = {r[0]: (r[1], r[2]) for r in cursor.execute(
        "SELECT team_name, COALESCE(offensive_modifier, 1.0), COALESCE(appearance_count, 0) FROM Dynamic_Modifiers"
    ).fetchall()}
    team_ops = {r[0]: r[1] for r in cursor.execute(
        "SELECT team_name, COALESCE(ops, 0.720) FROM Team_Offense"
    ).fetchall()}
    park_factors = {r[0]: r[1] for r in cursor.execute(
        "SELECT home_team, COALESCE(run_factor, 1.0) FROM Park_Factors"
    ).fetchall()}
    pitcher_stats = {r[0]: r[1] for r in cursor.execute(
        "SELECT last_name, COALESCE(est_era, 4.20) FROM Pitcher_Stats"
    ).fetchall()}

    def get_pitcher_shrunk_modifiers(p_name):
        if not p_name or p_name == "TBD":
            return 1.0, 1.0
        k_raw, f5_raw, n = pitcher_mods.get(p_name, (1.0, 1.0, 0))
        if f5_raw == 1.0 and " " in p_name:
            last_name = p_name.split()[-1]
            for name, mods in pitcher_mods.items():
                if name.endswith(last_name):
                    k_raw, f5_raw, n = mods
                    break
        w = min(1.0, n / 10.0)
        shrunk_f5 = w * f5_raw + (1.0 - w) * 1.0
        shrunk_k = w * k_raw + (1.0 - w) * 1.0
        return shrunk_f5, shrunk_k

    def get_team_shrunk_offense(t_name):
        raw_off, n = team_mods.get(t_name, (1.0, 0))
        w = min(1.0, n / 10.0)
        shrunk_off = w * raw_off + (1.0 - w) * 1.0
        base_ops = team_ops.get(t_name, 0.720)
        return (base_ops / 0.720) * shrunk_off

    print("-" * 60)
    for game in matchups:
        pk, away, home, away_sp, home_sp, rho, uv, ump = game
        
        away_sp = away_sp if away_sp else "TBD"
        home_sp = home_sp if home_sp else "TBD"
        
        park_factor = park_factors.get(home, 1.0)
        rho_mult = 1.000 + ((1.225 - rho) * 1.5) if rho else 1.000
        env_mult = park_factor * rho_mult * ump

        away_last_name = away_sp.split()[-1] if " " in away_sp else away_sp
        home_last_name = home_sp.split()[-1] if " " in home_sp else home_sp

        raw_a_xera = pitcher_stats.get(away_last_name, 4.20)
        raw_h_xera = pitcher_stats.get(home_last_name, 4.20)
        
        a_f5_mod, a_k_mod = get_pitcher_shrunk_modifiers(away_sp)
        h_f5_mod, h_k_mod = get_pitcher_shrunk_modifiers(home_sp)

        a_xera = max(1.5, min(9.0, raw_a_xera)) * a_f5_mod
        h_xera = max(1.5, min(9.0, raw_h_xera)) * h_f5_mod

        a_off = get_team_shrunk_offense(away)
        h_off = get_team_shrunk_offense(home)

        # 5-Inning Expected Runs (5/9 allocation)
        lam_a = max(0.05, (h_xera * a_off * env_mult) * (5.0 / 9.0))
        lam_h = max(0.05, (a_xera * h_off * env_mult) * (5.0 / 9.0))

        disp = 1.25
        va, vh = max(lam_a + 0.01, lam_a * disp), max(lam_h + 0.01, lam_h * disp)
        pa, ph = max(0.01, min(0.99, lam_a / va)), max(0.01, min(0.99, lam_h / vh))
        na, nh = max(0.1, (lam_a ** 2) / (va - lam_a)), max(0.1, (lam_h ** 2) / (vh - lam_h))

        iters = 20000
        sim_a = np.random.negative_binomial(na, pa, iters)
        sim_h = np.random.negative_binomial(nh, ph, iters)

        p_a = float(np.sum(sim_a > sim_h) / iters)
        p_h = float(np.sum(sim_h > sim_a) / iters)
        p_t = float(np.sum(sim_a == sim_h) / iters)

        cursor.execute('''
            INSERT OR REPLACE INTO F5_Forecasts 
            (game_pk, away_team, home_team, away_starter, home_starter, f5_away_prob, f5_home_prob, f5_tie_prob, f5_exp_away_runs, f5_exp_home_runs, f5_total_runs)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (pk, away, home, away_sp, home_sp, p_a, p_h, p_t, lam_a, lam_h, lam_a + lam_h))

        # Pitcher Props with Umpire and Bayesian-shrunk Strikeout Modifier
        for sp, tm, xera, k_mod in [(away_sp, away, a_xera, a_k_mod), (home_sp, home, h_xera, h_k_mod)]:
            if sp == "TBD": 
                continue
            proj_bf = max(18.0, min(26.0, 26.0 - (xera * 1.2)))
            proj_outs = (proj_bf * 0.72) * (4.20 / xera) ** 0.25
            base_k = 0.225 * (4.30 / xera) ** 0.5

            adj_k = min(0.38, max(0.12, base_k * (2.0 - ump) * (uv or 1.0) * k_mod))
            k_sims = np.random.binomial(int(round(proj_bf)), adj_k, 10000)

            cursor.execute('''
                INSERT OR REPLACE INTO Pitcher_Props 
                (game_pk, pitcher_name, team_name, projected_outs, projected_strikeouts, over_4_5_k_prob, over_5_5_k_prob, over_6_5_k_prob)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (pk, sp, tm, round(proj_outs, 1), float(np.mean(k_sims)), float(np.mean(k_sims >= 5)), float(np.mean(k_sims >= 6)), float(np.mean(k_sims >= 7))))
        
        print(f"[F5 / Props] {away} ({p_a:.1%}) @ {home} ({p_h:.1%}) | Tie: {p_t:.1%}")

    conn.commit()
    cursor.execute("PRAGMA wal_checkpoint(TRUNCATE);")
    conn.close()
    print("-" * 60)
    print("Secondary Engine Complete. High-Signal Markets Exported.")

if __name__ == "__main__":
    run_f5_and_props_engine()
