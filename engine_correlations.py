import sqlite3
import numpy as np

def run_cross_market_covariance():
    conn = sqlite3.connect('mlb_engine.db')
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    c.execute("""
        CREATE TABLE IF NOT EXISTS Correlated_Market_Forecasts (
            game_pk INTEGER,
            away_team TEXT,
            home_team TEXT,
            sgp_type TEXT,
            leg_1 TEXT,
            leg_2 TEXT,
            joint_prob REAL,
            uncorrelated_prob REAL,
            correlation_edge REAL,
            PRIMARY KEY (game_pk, leg_1, leg_2)
        )
    """)
    c.execute("DELETE FROM Correlated_Market_Forecasts")

    # Inspect columns dynamically
    m_cols = [r[1] for r in c.execute("PRAGMA table_info(Model_Forecasts)").fetchall()]
    
    # Identify probability and expected run columns dynamically
    prob_col = next((col for col in m_cols if 'home' in col.lower() and ('prob' in col.lower() or 'win' in col.lower())), None)
    if not prob_col:
        prob_col = next((col for col in m_cols if 'prob' in col.lower() or 'win' in col.lower()), None)
    
    exp_away = next((col for col in m_cols if 'away' in col.lower() and ('run' in col.lower() or 'exp' in col.lower())), None)
    exp_home = next((col for col in m_cols if 'home' in col.lower() and ('run' in col.lower() or 'exp' in col.lower()) and col != prob_col), None)

    forecast_rows = c.execute("SELECT * FROM Model_Forecasts").fetchall()
    forecasts = {row['game_pk']: dict(row) for row in forecast_rows}

    # Match games from Daily_Batters
    games = c.execute("SELECT DISTINCT game_pk FROM Daily_Batters").fetchall()

    inserts = []
    for g in games:
        pk = g['game_pk']
        teams = [r[0] for r in c.execute("SELECT DISTINCT team_name FROM Daily_Batters WHERE game_pk = ?", (pk,)).fetchall()]
        if len(teams) < 2:
            continue
        
        away, home = teams[0], teams[1]
        m = forecasts.get(pk, {})
        
        # Extract model probabilities
        p_home_fg = float(m.get(prob_col, 0.50)) if prob_col else 0.50
        p_away_fg = 1.0 - p_home_fg
        
        runs_a = float(m.get(exp_away, 4.5)) if exp_away else 4.5
        runs_h = float(m.get(exp_home, 4.5)) if exp_home else 4.5
        exp_total = runs_a + runs_h

        # Correlation 1: ML + Total Copula
        p_over_8_5 = 1.0 / (1.0 + np.exp(-0.45 * (exp_total - 8.5)))
        rho_away_over = 0.18
        p_away_and_over = (p_away_fg * p_over_8_5) + (rho_away_over * np.sqrt(p_away_fg * (1 - p_away_fg) * p_over_8_5 * (1 - p_over_8_5)))
        edge_away_over = p_away_and_over - (p_away_fg * p_over_8_5)

        inserts.append((
            pk, away, home, 'ML + Total',
            f"{away} ML", "Over 8.5 Runs",
            round(float(p_away_and_over), 4),
            round(float(p_away_fg * p_over_8_5), 4),
            round(float(edge_away_over), 4)
        ))

        # Correlation 2: Top Batter Hit + Team ML SGP
        top_batters = c.execute("""
            SELECT player_name, team_name, over_0_5_hit_prob, batting_order
            FROM Batter_Hit_Forecasts
            WHERE game_pk = ? AND batting_order IN (1, 2, 3)
        """, (pk,)).fetchall()

        for b in top_batters:
            b_name = b['player_name']
            b_team = b['team_name']
            p_hit = b['over_0_5_hit_prob']
            p_team_win = p_away_fg if b_team == away else p_home_fg

            rho_hit_win = 0.28
            joint_hit_win = (p_hit * p_team_win) + (rho_hit_win * np.sqrt(p_hit * (1 - p_hit) * p_team_win * (1 - p_team_win)))
            uncorrelated = p_hit * p_team_win
            edge_sgp = joint_hit_win - uncorrelated

            if edge_sgp > 0.02:
                inserts.append((
                    pk, away, home, 'Batter Hit + Team ML',
                    f"{b_name} Over 0.5 Hits", f"{b_team} ML",
                    round(float(joint_hit_win), 4),
                    round(float(uncorrelated), 4),
                    round(float(edge_sgp), 4)
                ))

    c.executemany("""
        INSERT OR REPLACE INTO Correlated_Market_Forecasts (
            game_pk, away_team, home_team, sgp_type, leg_1, leg_2,
            joint_prob, uncorrelated_prob, correlation_edge
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, inserts)

    conn.commit()
    conn.close()
    print(f"[SUCCESS] Calculated {len(inserts)} cross-market correlated edges into Correlated_Market_Forecasts.")

if __name__ == "__main__":
    run_cross_market_covariance()
