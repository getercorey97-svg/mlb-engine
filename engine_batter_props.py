import sqlite3
import numpy as np

TTO_FACTORS = {
    1: 0.915,
    2: 1.000,
    3: 1.165,
    4: 1.080
}

def get_db_connection():
    conn = sqlite3.connect('mlb_engine.db')
    conn.row_factory = sqlite3.Row
    return conn

def get_team_offense_rating(cursor, team_name):
    """Safely extracts team offense rating regardless of exact table column naming."""
    try:
        cols = [r[1] for r in cursor.execute("PRAGMA table_info(Team_Offense)").fetchall()]
        val_col = next((c for c in cols if c.lower() not in ['team_name', 'id', 'updated_at']), None)
        if val_col:
            row = cursor.execute(f"SELECT {val_col} FROM Team_Offense WHERE team_name = ?", (team_name,)).fetchone()
            if row and row[0]:
                return float(row[0])
    except Exception:
        pass
    return 0.940

def simulate_team_lineup_paths(cursor, game_pk, team_name, opp_sp_modifier, bullpen_fatigue, n_sims=5000):
    batters = cursor.execute("""
        SELECT player_name, batting_order 
        FROM Daily_Batters 
        WHERE game_pk = ? AND team_name = ?
        ORDER BY batting_order ASC
    """, (game_pk, team_name)).fetchall()

    if len(batters) < 9:
        return {}

    base_team_offense = get_team_offense_rating(cursor, team_name)

    player_stats = {b['batting_order']: {'name': b['player_name'], 'hits': np.zeros(n_sims), 'pa': np.zeros(n_sims), 'ab': np.zeros(n_sims)} for b in batters}

    lg_ba = 0.248
    lg_obp = 0.318

    batter_probs = {}
    for b in batters:
        order = b['batting_order']
        order_skill_adj = 1.0 + (5 - order) * 0.02
        b_obp = np.clip(lg_obp * base_team_offense * order_skill_adj, 0.220, 0.410)
        b_ba = np.clip(lg_ba * base_team_offense * order_skill_adj, 0.180, 0.330)
        batter_probs[order] = {'obp': b_obp, 'ba_on_obp': b_ba / b_obp}

    for sim in range(n_sims):
        lineup_idx = 1
        starter_bf = 0
        
        for inning in range(1, 10):
            outs = 0
            while outs < 3:
                order = lineup_idx
                player_stats[order]['pa'][sim] += 1
                starter_bf += 1
                
                if starter_bf <= 9:
                    sp_eff = opp_sp_modifier * TTO_FACTORS[1]
                elif starter_bf <= 18:
                    sp_eff = opp_sp_modifier * TTO_FACTORS[2]
                elif starter_bf <= 24:
                    sp_eff = opp_sp_modifier * TTO_FACTORS[3]
                else:
                    sp_eff = bullpen_fatigue * TTO_FACTORS[4]

                p_reach = np.clip(batter_probs[order]['obp'] * sp_eff, 0.15, 0.55)
                
                if np.random.rand() < p_reach:
                    player_stats[order]['ab'][sim] += 1
                    if np.random.rand() < batter_probs[order]['ba_on_obp']:
                        player_stats[order]['hits'][sim] += 1
                else:
                    outs += 1
                    player_stats[order]['ab'][sim] += 1

                lineup_idx = (lineup_idx % 9) + 1

    return player_stats

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

    c.execute("DELETE FROM Batter_Hit_Forecasts")

    games = c.execute("SELECT DISTINCT game_pk FROM Daily_Batters").fetchall()
    if not games:
        print("[CORRELATED PROPS] No active games found in Daily_Batters.")
        conn.close()
        return

    total_batters = 0
    pks = [g['game_pk'] for g in games]

    for pk in pks:
        teams = [r[0] for r in c.execute("SELECT DISTINCT team_name FROM Daily_Batters WHERE game_pk = ?", (pk,)).fetchall()]
        if len(teams) < 2:
            continue
        
        away_team, home_team = teams[0], teams[1]
        
        away_results = simulate_team_lineup_paths(c, pk, away_team, 1.0, 1.0)
        home_results = simulate_team_lineup_paths(c, pk, home_team, 1.0, 1.0)

        for side_data, team_name in [(away_results, away_team), (home_results, home_team)]:
            for order, stats in side_data.items():
                pa_mean = float(np.mean(stats['pa']))
                ab_mean = float(np.mean(stats['ab']))
                exp_hits = float(np.mean(stats['hits']))
                p_over_0_5 = float(np.mean(stats['hits'] >= 1))
                p_over_1_5 = float(np.mean(stats['hits'] >= 2))
                p_over_2_5 = float(np.mean(stats['hits'] >= 3))

                c.execute("""
                    INSERT OR REPLACE INTO Batter_Hit_Forecasts (
                        game_pk, player_name, team_name, batting_order,
                        projected_pa, projected_ab, expected_hits,
                        over_0_5_hit_prob, over_1_5_hit_prob, over_2_5_hit_prob
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    pk, stats['name'], team_name, order,
                    round(pa_mean, 2), round(ab_mean, 2), round(exp_hits, 2),
                    round(p_over_0_5, 4), round(p_over_1_5, 4), round(p_over_2_5, 4)
                ))
                total_batters += 1

    conn.commit()
    conn.close()
    print(f"[SUCCESS] Calibrated correlated hit props for {total_batters} batters into Batter_Hit_Forecasts.")

if __name__ == "__main__":
    run_batter_props_engine()
