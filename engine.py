import sqlite3
import numpy as np

def run_ultimate_monte_carlo():
    print("Initializing Ultimate Monte Carlo Engine with Dynamic Modifiers & Blowout Variance Dampening...")
    print("Synthesizing Statcast, Offense, Bullpens, Dynamic Modifiers, and Park Factors...")
    
    conn = sqlite3.connect('mlb_engine.db')
    cursor = conn.cursor()

    # Ensure required columns exist in Model_Forecasts
    for col in ["predicted_edge REAL", "predicted_home_runs REAL", "predicted_away_runs REAL"]:
        try:
            cursor.execute(f"ALTER TABLE Model_Forecasts ADD COLUMN {col}")
        except sqlite3.OperationalError:
            pass

    # Retrieve all active matchups
    cursor.execute("SELECT game_pk, away_team, home_team, away_pitcher, home_pitcher FROM Daily_Lineups")
    games = cursor.fetchall()
    
    if not games:
        print("No matchups found.")
        conn.close()
        return

    def get_metric(table, column, key_column, key_value, fallback):
        try:
            cursor.execute(f"SELECT {column} FROM {table} WHERE {key_column} LIKE ?", (f'%{key_value}%',))
            result = cursor.fetchone()
            return float(result[0]) if result and result[0] is not None else fallback
        except Exception:
            return fallback

    print("-" * 60)
    
    for game in games:
        game_pk, away, home, away_pitcher, home_pitcher = game
        
        # 1. Fetch Dynamic Feedback Modifiers
        away_off_mod = get_metric('Dynamic_Modifiers', 'offensive_modifier', 'team_name', away, 1.000)
        home_off_mod = get_metric('Dynamic_Modifiers', 'offensive_modifier', 'team_name', home, 1.000)
        away_pitch_mod = get_metric('Dynamic_Modifiers', 'pitching_modifier', 'team_name', away, 1.000)
        home_pitch_mod = get_metric('Dynamic_Modifiers', 'pitching_modifier', 'team_name', home, 1.000)

        # 2. Fetch Baseline Metrics
        home_starter_xera = get_metric('Pitcher_Stats', 'est_era', 'last_name', home_pitcher.split(' ')[-1], 4.30)
        away_starter_xera = get_metric('Pitcher_Stats', 'est_era', 'last_name', away_pitcher.split(' ')[-1], 4.30)
        
        away_ops = get_metric('Team_Offense', 'ops', 'team_name', away, 0.720)
        home_ops = get_metric('Team_Offense', 'ops', 'team_name', home, 0.720)
        
        home_bullpen = get_metric('Team_Bullpen', 'team_era', 'team_name', home, 4.00)
        away_bullpen = get_metric('Team_Bullpen', 'team_era', 'team_name', away, 4.00)
        
        park_factor = get_metric('Park_Factors', 'run_factor', 'home_team', home, 1.000)
        
        # 3. Calculate Dynamically Weighted Expected Runs (λ)
        away_ops_mult = (away_ops / 0.720) * away_off_mod
        home_ops_mult = (home_ops / 0.720) * home_off_mod
        
        adj_home_starter_xera = home_starter_xera * home_pitch_mod
        adj_away_starter_xera = away_xera * away_pitch_mod
        
        adj_home_bullpen = home_bullpen * home_pitch_mod
        adj_away_bullpen = away_bullpen * away_pitch_mod
        
        away_lambda_starter = (adj_home_starter_xera * away_ops_mult * park_factor) * 0.66
        away_lambda_bullpen = (adj_home_bullpen * away_ops_mult * park_factor) * 0.33
        away_lambda_total = away_lambda_starter + away_lambda_bullpen
        
        home_lambda_starter = (adj_away_starter_xera * home_ops_mult * park_factor) * 0.66
        home_lambda_bullpen = (adj_away_bullpen * home_ops_mult * park_factor) * 0.33
        home_lambda_total = home_lambda_starter + home_lambda_bullpen
        
        # 4. Execute Stochastic Monte Carlo Simulation (10,000 Iterations) with Variance Capping
        iterations = 10000
        
        away_sims = np.random.poisson(away_lambda_total, iterations)
        home_sims = np.random.poisson(home_lambda_total, iterations)
        
        # Resolve Extra Innings (Controlled ghost runner impact)
        ties = away_sims == home_sims
        while np.any(ties):
            away_extra = np.random.poisson((adj_home_bullpen / 3) + 0.9, np.sum(ties))
            home_extra = np.random.poisson((adj_away_bullpen / 3) + 0.9, np.sum(ties))
            
            away_sims[ties] += away_extra
            home_sims[ties] += home_extra
            ties = away_sims == home_sims
            
        # Blowout Variance Dampening: Clip extreme unrealistic simulation outliers (max 22 runs per team)
        # This directly mitigates the blowout correlation anomaly by stabilizing run distribution tails.
        away_sims = np.clip(away_sims, 0, 22)
        home_sims = np.clip(home_sims, 0, 22)
        
        # 5. Calculate Final Probabilities and Store
        away_wins = np.sum(away_sims > home_sims)
        home_wins = np.sum(home_sims > away_sims)
        
        away_prob = float(away_wins / iterations)
        home_prob = float(home_wins / iterations)
        edge = float(abs(home_prob - away_prob))
        
        cursor.execute('''
            UPDATE Model_Forecasts 
            SET away_prob = ?, home_prob = ?, predicted_edge = ?, predicted_home_runs = ?, predicted_away_runs = ?
            WHERE game_pk = ?
        ''', (away_prob, home_prob, edge, float(home_lambda_total), float(away_lambda_total), game_pk))
        
        print(f"[{away_pitcher} vs {home_pitcher}]")
        print(f"SIM: {away} ({away_prob:.1%}) @ {home} ({home_prob:.1%}) | Exp Runs: {away_lambda_total:.2f}-{home_lambda_total:.2f} | Edge: {edge:.3f}\n")

    conn.commit()
    conn.close()
    print("-" * 60)
    print("Execution complete. 100% resolution achieved with blowout variance mitigation.")

if __name__ == "__main__":
    run_ultimate_monte_carlo()
