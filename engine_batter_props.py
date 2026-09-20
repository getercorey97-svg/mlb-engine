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

def simulate_team_lineup_paths(cursor, game_pk, team_name, opp_sp_modifier, bullpen_fatigue, n_sims=10000):
    batters = cursor.execute("""
        SELECT player_name, batting_order 
        FROM Daily_Batters 
        WHERE game_pk = ? AND team_name = ? AND is_starter = 1
        ORDER BY batting_order ASC
    """, (game_pk, team_name)).fetchall()

    if len(batters) < 9:
        return {}

    team_row = cursor.execute("SELECT offense_rating FROM Team_Offense WHERE team_name = ?", (team_name,)).fetchone()
    base_team_offense = team_row['offense_rating'] if team_row else 0.940

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

    games = c.execute("""
        SELECT game_pk, away_team, home_team, away_sp, home_sp, 
               away_sp_modifier, home_sp_modifier, bullpen_fatigue_away, bullpen_fatigue_home
        FROM Daily_Lineups
        WHERE lineup_status = 'Confirmed'
    """).fetchall()

    if not games:
        print("[CORRELATED PROPS] No confirmed games found in Daily_Lineups.")
        conn.close()
        return

    c.execute("DELETE FROM Batter_Hit_Forecasts")

    total_batters = 0
    for g in games:
        pk = g['game_pk']
        home_sp_mod = g['home_sp_modifier'] if g['home_sp_modifier'] else 1.000
        away_sp_mod = g['away_sp_modifier'] if g['away_sp_modifier'] else 1.000
        home_bp_fatigue = g['bullpen_fatigue_home'] if g['bullpen_fatigue_home'] else 1.000
        away_bp_fatigue = g['bullpen_fatigue_away'] if g['bullpen_fatigue_away'] else 1.000

        away_results = simulate_team_lineup_paths(c, pk, g['away_team'], home_sp_mod, home_bp_fatigue)
        home_results = simulate_team_lineup_paths(c, pk, g['home_team'], away_sp_mod, away_bp_fatigue)

        for side_data, team_name in [(away_results, g['away_team']), (home_results, g['home_team'])]:
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
