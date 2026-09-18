import sqlite3
import numpy as np
import warnings

warnings.filterwarnings('ignore')

# Times-Through-The-Order early run suppression coefficient for Innings 1-5
TTOP_SUPPRESSION_FACTOR = 0.90
# Top-of-the-order PA concentration weighting (Batters 1-4 receive 3 PAs vs 2 PAs for 7-9)
TOP_ORDER_WEIGHT = 1.03

def ensure_f5_schemas(cursor):
    """Guarantees F5 and Pitcher Props tables exist with median total schema migrations."""
    cursor.executescript('''
    CREATE TABLE IF NOT EXISTS F5_Forecasts (
        game_pk INTEGER PRIMARY KEY,
        away_team TEXT,
        home_team TEXT,
        away_starter TEXT,
        home_starter TEXT,
        f5_away_prob REAL,
        f5_home_prob REAL,
        f5_tie_prob REAL,
        f5_exp_away_runs REAL,
        f5_exp_home_runs REAL,
        f5_total_runs REAL,
        f5_median_total REAL DEFAULT 0.0
    );
    CREATE TABLE IF NOT EXISTS Pitcher_Props (
        game_pk INTEGER,
        pitcher_name TEXT,
        team_name TEXT,
        projected_outs REAL,
        projected_strikeouts REAL,
        over_4_5_k_prob REAL,
        over_5_5_k_prob REAL,
        over_6_5_k_prob REAL,
        PRIMARY KEY (game_pk, pitcher_name)
    );
    ''')
    
    cursor.execute("PRAGMA table_info(F5_Forecasts);")
    cols = [c[1] for c in cursor.fetchall()]
    if "f5_median_total" not in cols:
        cursor.execute("ALTER TABLE F5_Forecasts ADD COLUMN f5_median_total REAL DEFAULT 0.0;")

def run_f5_and_props_engine():
    conn = sqlite3.connect('mlb_engine.db', timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=10000;")
    cursor = conn.cursor()

    ensure_f5_schemas(cursor)
    conn.commit()

    # Query active slate matchups with atmospheric and umpire conditions
    cursor.execute('''
        SELECT d.game_pk, d.away_team, d.home_team, d.away_pitcher, d.home_pitcher, 
               COALESCE(d.air_density, 1.225), COALESCE(d.uv_modifier, 5.0), 
               COALESCE(u.run_modifier, 1.00), COALESCE(u.home_plate_umpire, 'TBD')
        FROM Daily_Lineups d 
        LEFT JOIN Daily_Umpires u ON d.game_pk = u.game_pk
        WHERE d.status != 'Final' AND d.game_pk NOT IN (SELECT game_pk FROM Post_Match_Analysis)
    ''')
    matchups = cursor.fetchall()
    if not matchups:
        conn.close()
        return

    # Ingest feature caches
    pitcher_mods = {r[0]: (r[1], r[2], r[3]) for r in cursor.execute(
        "SELECT pitcher_name, COALESCE(k_modifier, 1.0), COALESCE(f5_run_modifier, 1.0), COALESCE(appearance_count, 0) FROM Pitcher_Modifiers"
    ).fetchall()}
    team_mods = {r[0]: (r[1], r[2]) for r in cursor.execute(
        "SELECT team_name, COALESCE(offensive_modifier, 1.0), COALESCE(appearance_count, 0) FROM Dynamic_Modifiers"
    ).fetchall()}
    park_factors = {r[0]: r[1] for r in cursor.execute("SELECT home_team, COALESCE(run_factor, 1.0) FROM Park_Factors").fetchall()}
    pitcher_stats = {r[0]: (r[1], r[2]) for r in cursor.execute("SELECT last_name, COALESCE(est_era, 4.20), COALESCE(throws, 'R') FROM Pitcher_Stats").fetchall()}
    
    # Ingest platoon splits: team_name -> (ops_vs_rhp, ops_vs_lhp)
    platoon_ops = {r[0]: (r[1], r[2]) for r in cursor.execute(
        "SELECT team_name, COALESCE(ops_vs_rhp, 0.720), COALESCE(ops_vs_lhp, 0.720) FROM Team_Offense"
    ).fetchall()}

    def get_shrunk_pitcher(name):
        k_raw, f5_raw, n = pitcher_mods.get(name, (1.0, 1.0, 0))
        w = min(1.0, n / 10.0)
        return (w * f5_raw + (1.0 - w)), (w * k_raw + (1.0 - w))

    def get_platoon_offense(team, sp_hand):
        rhp_ops, lhp_ops = platoon_ops.get(team, (0.720, 0.720))
        target_ops = lhp_ops if sp_hand == 'L' else rhp_ops
        raw_off, n = team_mods.get(team, (1.0, 0))
        w = min(1.0, n / 10.0)
        shrunk_mod = w * raw_off + (1.0 - w)
        return (target_ops / 0.720) * shrunk_mod * TOP_ORDER_WEIGHT

    for game in matchups:
        pk, away, home, away_sp, home_sp, rho, uv_raw, ump_run, ump_name = game
        
        # 1. Atmospheric & Visual Carry Modifiers
        rho_mult = 1.000 + ((1.225 - rho) * 1.5)
        uv_glare_mod = 1.000 + (np.clip(uv_raw, 1.0, 11.0) - 5.0) * 0.005
        env_mult = park_factors.get(home, 1.0) * rho_mult * ump_run

        # 2. Pure Starter Profiling (Zero Bullpen Contamination + TTOP Suppression)
        a_ln = away_sp.split()[-1] if " " in away_sp else away_sp
        h_ln = home_sp.split()[-1] if " " in home_sp else home_sp
        a_xera_raw, a_hand = pitcher_stats.get(a_ln, (4.20, 'R'))
        h_xera_raw, h_hand = pitcher_stats.get(h_ln, (4.20, 'R'))

        a_f5_mod, a_k_mod = get_shrunk_pitcher(away_sp)
        h_f5_mod, h_k_mod = get_shrunk_pitcher(home_sp)

        # Apply 0.90x TTOP suppression for turns 1 and 2 through the batting order
        a_xera = np.clip(a_xera_raw, 1.5, 9.0) * a_f5_mod * TTOP_SUPPRESSION_FACTOR
        h_xera = np.clip(h_xera_raw, 1.5, 9.0) * h_f5_mod * TTOP_SUPPRESSION_FACTOR

        # 3. Isolated F5 Run Expectancy
        lam_a = max(0.10, (h_xera * get_platoon_offense(away, h_hand) * env_mult) * (5.0 / 9.0))
        lam_h = max(0.10, (a_xera * get_platoon_offense(home, a_hand) * env_mult) * (5.0 / 9.0))

        # 4. Negative Binomial Stochastic Sampling
        disp = 1.22
        va, vh = lam_a * disp, lam_h * disp
        pa, ph = max(0.01, min(0.99, lam_a / va)), max(0.01, min(0.99, lam_h / vh))
        na, nh = max(0.1, (lam_a ** 2) / (va - lam_a)), max(0.1, (lam_h ** 2) / (vh - lam_h))

        rng = np.random.default_rng(seed=int(pk))
        sim_a = np.clip(rng.negative_binomial(na, pa, 25000), 0, 15)
        sim_h = np.clip(rng.negative_binomial(nh, ph, 25000), 0, 15)
        sim_tot = sim_a + sim_h

        # L1-Optimized Median Point Projection
        f5_median_total = float(np.median(sim_tot))
        f5_mean_total = round(lam_a + lam_h, 2)

        f5_away_prob = float(np.mean(sim_a > sim_h))
        f5_home_prob = float(np.mean(sim_h > sim_a))
        f5_tie_prob = float(np.mean(sim_a == sim_h))

        cursor.execute('''
            INSERT OR REPLACE INTO F5_Forecasts 
            (game_pk, away_team, home_team, away_starter, home_starter, f5_away_prob, f5_home_prob, f5_tie_prob, f5_exp_away_runs, f5_exp_home_runs, f5_total_runs, f5_median_total)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (pk, away, home, away_sp, home_sp, round(f5_away_prob, 4), round(f5_home_prob, 4), 
              round(f5_tie_prob, 4), round(lam_a, 2), round(lam_h, 2), f5_mean_total, round(f5_median_total, 2)))

        # 5. Decoupled Pitcher Strikeout Props Modeling
        ump_k_mod = 1.000 - ((ump_run - 1.000) * 1.6)
        for sp, tm, xera, k_mod in [(away_sp, away, a_xera, a_k_mod), (home_sp, home, h_xera, h_k_mod)]:
            if sp in ("TBD", "Unknown"):
                continue
            proj_bf = np.clip(26.0 - (xera * 1.15), 18.0, 27.0)
            proj_outs = (proj_bf * 0.72) * ((4.20 / xera) ** 0.25)
            base_k = 0.225 * ((4.30 / xera) ** 0.50)
            adj_k = np.clip(base_k * ump_k_mod * uv_glare_mod * k_mod, 0.10, 0.42)
            
            k_sims = rng.binomial(int(round(proj_bf)), adj_k, 10000)
            cursor.execute('''
                INSERT OR REPLACE INTO Pitcher_Props 
                (game_pk, pitcher_name, team_name, projected_outs, projected_strikeouts, over_4_5_k_prob, over_5_5_k_prob, over_6_5_k_prob)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (pk, sp, tm, round(proj_outs, 1), float(np.mean(k_sims)), float(np.mean(k_sims >= 5)), 
                  float(np.mean(k_sims >= 6)), float(np.mean(k_sims >= 7))))

    conn.commit()
    cursor.execute("PRAGMA wal_checkpoint(TRUNCATE);")
    conn.close()

if __name__ == "__main__":
    run_f5_and_props_engine()
