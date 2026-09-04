import sqlite3
import numpy as np

def run_f5_and_props_engine():
    print("Initializing Phase 4B: Secondary Engine (First 5 & Pitcher Props)...")
    conn = sqlite3.connect('mlb_engine.db')
    cursor = conn.cursor()
    
    # Failsafe Schema Execution
    cursor.executescript('''
    CREATE TABLE IF NOT EXISTS F5_Forecasts (
        game_pk INTEGER PRIMARY KEY, away_team TEXT, home_team TEXT, away_starter TEXT, home_starter TEXT, 
        f5_away_prob REAL, f5_home_prob REAL, f5_tie_prob REAL, f5_exp_away_runs REAL, f5_exp_home_runs REAL, f5_total_runs REAL
    );
    CREATE TABLE IF NOT EXISTS Pitcher_Props (
        game_pk INTEGER, pitcher_name TEXT, team_name TEXT, projected_outs REAL, projected_strikeouts REAL, 
        over_4_5_k_prob REAL, over_5_5_k_prob REAL, over_6_5_k_prob REAL, PRIMARY KEY (game_pk, pitcher_name)
    );
    ''')

    # The Magic Failsafe: If ALV Database dropped the UV column, add it silently
    try:
        cursor.execute("ALTER TABLE Daily_Lineups ADD COLUMN uv_modifier REAL DEFAULT 1.0")
    except sqlite3.OperationalError:
        pass

    try:
        cursor.execute('''
            SELECT d.game_pk, d.away_team, d.home_team, d.away_pitcher, d.home_pitcher, d.air_density, d.uv_modifier, COALESCE(u.run_modifier, 1.0)
            FROM Daily_Lineups d LEFT JOIN Daily_Umpires u ON d.game_pk = u.game_pk
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

    def get_metric(table, col, key_col, key_val, fallback):
        try:
            cursor.execute(f"SELECT {col} FROM {table} WHERE {key_col} LIKE ?", (f'%{key_val}%',))
            res = cursor.fetchone()
            return float(res[0]) if res else fallback
        except Exception: 
            return fallback

    print("-" * 60)
    for game in matchups:
        pk, away, home, away_sp, home_sp, rho, uv, ump = game
        
        # Failsafe strings to prevent .split() NoneType errors
        away_sp = away_sp if away_sp else "TBD"
        home_sp = home_sp if home_sp else "TBD"
        
        # Environmental Base (No bullpen fatigue)
        park_factor = get_metric('Park_Factors', 'run_factor', 'home_team', home, 1.0)
        rho_mult = 1.000 + ((1.225 - rho) * 1.5) if rho else 1.000
        env_mult = park_factor * rho_mult * ump

        # Extract last names safely
        away_last_name = away_sp.split()[-1] if " " in away_sp else away_sp
        home_last_name = home_sp.split()[-1] if " " in home_sp else home_sp

        # Apply Pitcher-Specific Micro-Evolutions with a safety clamp to prevent ZeroDivisionError
        raw_a_xera = get_metric('Pitcher_Stats', 'est_era', 'last_name', away_last_name, 4.20)
        raw_h_xera = get_metric('Pitcher_Stats', 'est_era', 'last_name', home_last_name, 4.20)
        
        # Clamp xERA between 1.50 and 9.00 to prevent math blowups
        a_xera = max(1.5, min(9.0, raw_a_xera)) * get_metric('Pitcher_Modifiers', 'f5_run_modifier', 'pitcher_name', away_sp, 1.0)
        h_xera = max(1.5, min(9.0, raw_h_xera)) * get_metric('Pitcher_Modifiers', 'f5_run_modifier', 'pitcher_name', home_sp, 1.0)

        a_off = (get_metric('Team_Offense', 'ops', 'team_name', away, 0.72) / 0.72) * get_metric('Dynamic_Modifiers', 'offensive_modifier', 'team_name', away, 1.0)
        h_off = (get_metric('Team_Offense', 'ops', 'team_name', home, 0.72) / 0.72) * get_metric('Dynamic_Modifiers', 'offensive_modifier', 'team_name', home, 1.0)

        # Pure 5-Inning Expected Runs (0.555 fraction of total expectation)
        lam_a = max(0.05, (h_xera * a_off * env_mult) * (5.0 / 9.0))
        lam_h = max(0.05, (a_xera * h_off * env_mult) * (5.0 / 9.0))

        # Overdispersed F5 Monte Carlo
        disp = 1.25
        va, vh = max(lam_a + 0.01, lam_a * disp), max(lam_h + 0.01, lam_h * disp)
        pa, ph = max(0.01, min(0.99, lam_a/va)), max(0.01, min(0.99, lam_h/vh))
        na, nh = max(0.1, (lam_a**2)/(va-lam_a)), max(0.1, (lam_h**2)/(vh-lam_h))

        iters = 20000
        sim_a = np.random.negative_binomial(na, pa, iters)
        sim_h = np.random.negative_binomial(nh, ph, iters)

        p_a, p_h, p_t = float(np.sum(sim_a > sim_h)/iters), float(np.sum(sim_h > sim_a)/iters), float(np.sum(sim_a == sim_h)/iters)

        # EXPLICIT COLUMN MAPPING TO FIX THE SQLITE CRASH
        cursor.execute('''
            INSERT OR REPLACE INTO F5_Forecasts 
            (game_pk, away_team, home_team, away_starter, home_starter, f5_away_prob, f5_home_prob, f5_tie_prob, f5_exp_away_runs, f5_exp_home_runs, f5_total_runs)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (pk, away, home, away_sp, home_sp, p_a, p_h, p_t, lam_a, lam_h, lam_a+lam_h))

        # Pitcher Props (SOTA Binomial incorporating Umpire Expansion bias)
        for sp, tm, xera in [(away_sp, away, a_xera), (home_sp, home, h_xera)]:
            if sp == "TBD": continue
            proj_bf = max(18.0, min(26.0, 26.0 - (xera * 1.2)))
            proj_outs = (proj_bf * 0.72) * (4.20 / xera)**0.25
            base_k = 0.225 * (4.30 / xera)**0.5
            k_mod = get_metric('Pitcher_Modifiers', 'k_modifier', 'pitcher_name', sp, 1.0)

            # (2.0 - ump) means if Umpire is 0.97 (Pitcher friendly), the K-rate gets a 1.03 multiplier boost
            adj_k = min(0.38, max(0.12, base_k * (2.0 - ump) * (uv or 1.0) * k_mod))
            k_sims = np.random.binomial(int(round(proj_bf)), adj_k, 10000)

            # EXPLICIT COLUMN MAPPING
            cursor.execute('''
                INSERT OR REPLACE INTO Pitcher_Props 
                (game_pk, pitcher_name, team_name, projected_outs, projected_strikeouts, over_4_5_k_prob, over_5_5_k_prob, over_6_5_k_prob)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (pk, sp, tm, round(proj_outs, 1), float(np.mean(k_sims)), float(np.mean(k_sims>=5)), float(np.mean(k_sims>=6)), float(np.mean(k_sims>=7))))
        
        print(f"[F5 / Props] {away} ({p_a:.1%}) @ {home} ({p_h:.1%}) | Tie: {p_t:.1%}")

    conn.commit()
    conn.close()
    print("-" * 60)
    print("Secondary Engine Complete. High-Signal Markets Exported.")

if __name__ == "__main__":
    run_f5_and_props_engine()
