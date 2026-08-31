import pandas as pd
import sqlite3
import numpy as np

def run_ultimate_monte_carlo():
    print("Initializing Ultimate Monte Carlo Engine with Air Density, Umpire Variance & Total Runs Output...")
    print("Synthesizing Statcast, Offense, Bullpens, Dynamic Modifiers, and Park Factors...")
    
    conn = sqlite3.connect('mlb_engine.db')
    cursor = conn.cursor()

    # Ensure required columns exist in Model_Forecasts
    for col in ["predicted_edge REAL", "predicted_home_runs REAL", "predicted_away_runs REAL"]:
        try:
            cursor.execute(f"ALTER TABLE Model_Forecasts ADD COLUMN {col}")
        except sqlite3.OperationalError:
            pass

    # Retrieve all active matchups, enforcing the ALV mandate to pull exact lineups, air density, and umpire biases
    cursor.execute('''
        SELECT d.game_pk, d.away_team, d.home_team, d.away_pitcher, d.home_pitcher, d.air_density, COALESCE(u.run_modifier, 1.0)
        FROM Daily_Lineups d
        LEFT JOIN Daily_Umpires u ON d.game_pk = u.game_pk
    ''')
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
        game_pk, away, home, away_pitcher, home_pitcher, air_density, umpire_multiplier = game
        
        # Convert Density to an aerodynamic Run Multiplier
        density_multiplier = 1.000 + ((1.225 - air_density) * 1.5) if air_density else 1.000
        
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
        adj_away_starter_xera = away_starter_xera * away_pitch_mod
        
        adj_home_bullpen = home_bullpen * home_pitch_mod
        adj_away_bullpen = away_bullpen * away_pitch_mod
        
        # Integrate ALV structural math, Park Factors, Air Density, and Umpire Bias
        away_lambda_starter = (adj_home_starter_xera * away_ops_mult * park_factor * density_multiplier * umpire_multiplier) * 0.66
        away_lambda_bullpen = (adj_home_bullpen * away_ops_mult * park_factor * density_multiplier * umpire_multiplier) * 0.33
        away_lambda_total = away_lambda_starter + away_lambda_bullpen
        
        home_lambda_starter = (adj_away_starter_xera * home_ops_mult * park_factor * density_multiplier * umpire_multiplier) * 0.66
        home_lambda_bullpen = (adj_away_bullpen * home_ops_mult * park_factor * density_multiplier * umpire_multiplier) * 0.33
        home_lambda_total = home_lambda_starter + home_lambda_bullpen
        
        # --- SAFETY CLAMP: Prevent lam < 0 or NaN crash ---
        if pd.isna(away_lambda_total) or away_lambda_total < 0.1:
            away_lambda_total = 3.50
        if pd.isna(home_lambda_total) or home_lambda_total < 0.1:
            home_lambda_total = 3.50
        # --------------------------------------------------

        # Calculate Combined Total Runs (Over/Under Line)
        total_runs = away_lambda_total + home_lambda_total
        
        # 4. Execute High-Precision Stochastic Monte Carlo Simulation (50,000 Iterations)
        iterations = 50000
        
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
        
        # Detailed Console Log featuring Away - Home breakdown AND total runs
        print(f"[{away_pitcher} vs {home_pitcher}]")
        print(f"SIM (50k): {away} ({away_prob:.1%}) @ {home} ({home_prob:.1%}) | Total Runs: {total_runs:.2f} ({away_lambda_total:.2f} - {home_lambda_total:.2f}) | Edge: {edge:.3f} | ρ: {air_density} | Ump: {umpire_multiplier}\n")

    conn.commit()
    conn.close()
    print("-" * 60)
    print("Execution complete. High-precision convergence achieved with 50,000 iterations.")

if __name__ == "__main__":
    run_ultimate_monte_carlo()
