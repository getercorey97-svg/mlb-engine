import sqlite3
import numpy as np

def run_ultimate_monte_carlo():
    print("Initializing Ultimate Monte Carlo Engine...")
    print("Synthesizing Statcast, Offense, Bullpens, and Park Factors...")
    
    conn = sqlite3.connect('mlb_engine.db')
    cursor = conn.cursor()

    # Retrieve all active matchups
    cursor.execute("SELECT game_pk, away_team, home_team, away_pitcher, home_pitcher FROM Daily_Lineups")
    games = cursor.fetchall()
    
    if not games:
        print("No matchups found.")
        return

    # Helper function to pull specific metrics from the database
    def get_metric(table, column, key_column, key_value, fallback):
        try:
            cursor.execute(f"SELECT {column} FROM {table} WHERE {key_column} LIKE ?", (f'%{key_value}%',))
            result = cursor.fetchone()
            return float(result[0]) if result else fallback
        except Exception:
            return fallback

    print("-" * 60)
    
    for game in games:
        game_pk, away, home, away_pitcher, home_pitcher = game
        
        # 1. Fetch Baseline Metrics
        away_starter_xera = get_metric('Pitcher_Stats', 'est_era', 'last_name', home_pitcher.split(' ')[-1], 4.30)
        home_starter_xera = get_metric('Pitcher_Stats', 'est_era', 'last_name', away_pitcher.split(' ')[-1], 4.30)
        
        away_ops = get_metric('Team_Offense', 'ops', 'team_name', away, 0.720)
        home_ops = get_metric('Team_Offense', 'ops', 'team_name', home, 0.720)
        
        away_bullpen = get_metric('Team_Bullpen', 'team_era', 'team_name', home, 4.00)
        home_bullpen = get_metric('Team_Bullpen', 'team_era', 'team_name', away, 4.00)
        
        park_factor = get_metric('Park_Factors', 'run_factor', 'home_team', home, 1.000)
        
        # 2. Calculate Advanced Expected Runs (λ)
        # Adjusting the pitcher's baseline by the opposing offense (League Avg OPS is ~0.720) and the park
        away_ops_mult = away_ops / 0.720
        home_ops_mult = home_ops / 0.720
        
        # Starters generally go ~6 innings (6/9 = 0.66), Bullpens go ~3 (3/9 = 0.33)
        away_lambda_starter = (away_starter_xera * away_ops_mult * park_factor) * 0.66
        away_lambda_bullpen = (away_bullpen * away_ops_mult * park_factor) * 0.33
        
        home_lambda_starter = (home_starter_xera * home_ops_mult * park_factor) * 0.66
        home_lambda_bullpen = (home_bullpen * home_ops_mult * park_factor) * 0.33
        
        # 3. Execute Stochastic Monte Carlo Simulation (10,000 Iterations)
        iterations = 10000
        
        # Simulate Regulation (9 Innings)
        away_sims = np.random.poisson(away_lambda_starter + away_lambda_bullpen, iterations)
        home_sims = np.random.poisson(home_lambda_starter + home_lambda_bullpen, iterations)
        
        # 4. Resolve Extra Innings (Tiebreakers)
        ties = away_sims == home_sims
        while np.any(ties):
            # Extra inning scoring environment (Ghost runner adds ~1.1 expected runs per inning)
            away_extra = np.random.poisson((away_lambda_bullpen / 3) + 1.1, np.sum(ties))
            home_extra = np.random.poisson((home_lambda_bullpen / 3) + 1.1, np.sum(ties))
            
            away_sims[ties] += away_extra
            home_sims[ties] += home_extra
            
            # Re-evaluate ties
            ties = away_sims == home_sims
        
        # 5. Calculate Final Win Probabilities
        away_wins = np.sum(away_sims > home_sims)
        home_wins = np.sum(home_sims > away_sims)
        
        away_prob = away_wins / iterations
        home_prob = home_wins / iterations
        edge = abs(home_prob - away_prob)
        
        cursor.execute('''
            UPDATE Model_Forecasts 
            SET away_prob = ?, home_prob = ?, predicted_edge = ?
            WHERE game_pk = ?
        ''', (away_prob, home_prob, edge, game_pk))
        
        print(f"[{away_pitcher} vs {home_pitcher}]")
        print(f"SIM: {away} ({away_prob:.1%}) @ {home} ({home_prob:.1%}) | Edge: {edge:.3f}\n")

    conn.commit()
    print("-" * 60)
    print("Execution complete. 100% resolution achieved.")
    conn.close()

if __name__ == "__main__":
    run_ultimate_monte_carlo()

