import sqlite3
import numpy as np
import pandas as pd

def run_ultimate_monte_carlo():
    conn = sqlite3.connect('mlb_engine.db')
    cursor = conn.cursor()
    
    # Pre-fetch all metrics to memory to avoid DB lag during sim
    ops_map = {r[0]: r[1] for r in conn.execute("SELECT team_name, ops FROM Team_Offense").fetchall()}
    bp_map = {r[0]: r[1] for r in conn.execute("SELECT team_name, team_era FROM Team_Bullpen").fetchall()}
    pitcher_map = {r[0]: r[1] for r in conn.execute("SELECT last_name, est_era FROM Pitcher_Stats").fetchall()}
    park_map = {r[0]: r[1] for r in conn.execute("SELECT home_team, run_factor FROM Park_Factors").fetchall()}

    cursor.execute('SELECT game_pk, away_team, home_team, away_pitcher, home_pitcher, air_density FROM Daily_Lineups WHERE status != "Final"')
    games = cursor.fetchall()

    for pk, away, home, away_p, home_p, rho in games:
        a_ops = ops_map.get(away, 0.720)
        h_ops = ops_map.get(home, 0.720)
        h_sp_era = pitcher_map.get(home_p.split(' ')[-1], 4.20)
        a_sp_era = pitcher_map.get(away_p.split(' ')[-1], 4.20)
        
        # Sim Logic (50k iterations vectorized)
        it = 50000
        away_runs = np.random.poisson((h_sp_era * (a_ops/0.72)), it)
        home_runs = np.random.poisson((a_sp_era * (h_ops/0.72)), it)
        
        a_prob = np.mean(away_runs > home_runs)
        h_prob = 1.0 - a_prob
        
        cursor.execute('UPDATE Model_Forecasts SET away_prob=?, home_prob=?, timestamp=DATETIME("now") WHERE game_pk=?', (a_prob, h_prob, pk))
        print(f"Sim Complete: {away} @ {home} | H Win: {h_prob:.1%}")

    conn.commit()
    conn.close()

if __name__ == "__main__":
    run_ultimate_monte_carlo()
