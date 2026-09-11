import sqlite3
import numpy as np

def run_ultimate_monte_carlo():
    conn = sqlite3.connect('mlb_engine.db')
    cursor = conn.cursor()
    
    # Pre-fetch all metrics to RAM to prevent database locking/lag
    ops_map = {r[0]: r[1] for r in conn.execute("SELECT team_name, ops FROM Team_Offense").fetchall()}
    pitcher_map = {r[0]: r[1] for r in conn.execute("SELECT last_name, est_era FROM Pitcher_Stats").fetchall()}

    cursor.execute('SELECT game_pk, away_team, home_team, away_pitcher, home_pitcher FROM Daily_Lineups WHERE status != "Final"')
    games = cursor.fetchall()

    for pk, away, home, away_p, home_p in games:
        a_ops = ops_map.get(away, 0.720)
        h_ops = ops_map.get(home, 0.720)
        h_sp_era = pitcher_map.get(home_p.split(' ')[-1] if home_p else "", 4.20)
        a_sp_era = pitcher_map.get(away_p.split(' ')[-1] if away_p else "", 4.20)
        
        # 50,000 Iterations Vectorized in NumPy (Lightning Fast)
        it = 50000
        away_runs = np.random.poisson((h_sp_era * (a_ops/0.72) * 0.55), it)
        home_runs = np.random.poisson((a_sp_era * (h_ops/0.72) * 0.55), it)
        
        h_prob = np.mean(home_runs > away_runs)
        a_prob = 1.0 - h_prob
        
        cursor.execute('UPDATE Model_Forecasts SET away_prob=?, home_prob=?, timestamp=DATETIME("now") WHERE game_pk=?', (a_prob, h_prob, pk))
        print(f"Engine Resolve: {away} @ {home} | Home Prob: {h_prob:.1%}")

    conn.commit()
    conn.close()

if __name__ == "__main__":
    run_ultimate_monte_carlo()
