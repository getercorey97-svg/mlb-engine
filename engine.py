import sqlite3
import numpy as np

def run_ultimate_monte_carlo():
    print("Initializing Clean Monte Carlo Engine...")
    
    conn = sqlite3.connect('mlb_engine.db')
    cursor = conn.cursor()

    # Ensure required columns exist in Model_Forecasts
    for col in ["predicted_edge REAL", "predicted_home_runs REAL", "predicted_away_runs REAL"]:
        try:
            cursor.execute(f"ALTER TABLE Model_Forecasts ADD COLUMN {col}")
        except sqlite3.OperationalError:
            pass

    # Retrieve matchups and available environmental/umpire data safely
    cursor.execute('''
        SELECT d.game_pk, d.away_team, d.home_team, d.away_pitcher, d.home_pitcher, 
               COALESCE(d.air_density, 1.225), COALESCE(u.run_modifier, 1.0)
        FROM Daily_Lineups d
        LEFT JOIN Daily_Umpires u ON d.game_pk = u.game_pk
    ''')
    games = cursor.fetchall()
    
    if not games:
        print("No matchups found in Daily_Lineups.")
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
        
        # Aerodynamic run adjustment based on air density
        density_multiplier = 1.000 + ((1.225 - air_density) * 1.5)
        
        # Fetch Dynamic Modifiers (default 1.0)
        away_off_mod = get_metric('Dynamic_Modifiers', 'offensive_modifier', 'team_name', away, 1.0)
        home_off_mod = get_metric('Dynamic_Modifiers', 'offensive_modifier', 'team_name', home, 1.0)
        away_pitch_mod = get_metric('Dynamic_Modifiers', 'pitching_modifier', 'team_name', away, 1.0)
        home_pitch_mod = get_metric('Dynamic_Modifiers', 'pitching_modifier', 'team_name', home, 1.0)

        # Fetch Baselines (with safe fallbacks)
        home_starter_xera = get_metric('Pitcher_Stats', 'est_era', 'last_name', home_pitcher.split(' ')[-1], 4.30)
        away_starter_xera = get_metric('Pitcher_Stats', 'est_era', 'last_name', away_pitcher.split(' ')[-1], 4.30)
        
        away_ops = get_metric('Team_Offense', 'ops', 'team_name', away, 0.720)
        home_ops = get_metric('Team_Offense', 'ops', 'team_name', home, 0.720)
        
        home_bullpen = get_metric('Team_Bullpen', 'team_era', 'team_name', home, 4.00)
        away_bullpen = get_metric('Team_Bullpen', 'team_era', 'team_name', away, 4.00)
        
        park_factor = get_metric('Park_Factors', 'run_factor', 'home_team', home, 1.000)
        
        # Calculate Expected Runs (Lambda) for Away and Home teams
        away_ops_mult = (away_ops / 0.720) * away_off_mod
        home_ops_mult = (home_ops / 0.720) * home_off_mod
        
        adj_home_starter = home_starter_xera * home_pitch_mod
        adj_away_starter = away_starter_xera * away_pitch_mod
        adj_home_bullpen = home_bullpen * home_pitch_mod
        adj_away_bullpen = away_bullpen * away_pitch_mod
        
        # Environmental multiplier combination
        env_mult = park_factor * density_multiplier * umpire_multiplier
        
        # Away team expected runs against Home pitching (66% starter, 33% bullpen)
        away_lambda = ((adj_home_starter * away_ops_mult * env_mult) * 0.66) + \
                      ((adj_home_bullpen * away_ops_mult * env_mult) * 0.33)
                      
        # Home team expected runs against Away pitching
        home_lambda = ((adj_away_starter * home_ops_mult * env_mult) * 0.66) + \
                      ((adj_away_bullpen * home_ops_mult * env_mult) * 0.33)
        
        # Total Runs (Over/Under line)
        total_runs = away_lambda + home_lambda
        
        # Run 50,000 Monte Carlo iterations
        iterations = 50000
        away_sims = np.random.poisson(max(0.1, away_lambda), iterations)
        home_sims = np.random.poisson(max(0.1, home_lambda), iterations)
        
        # Extra innings resolution for ties
        ties = away_sims == home_sims
        while np.any(ties):
            away_sims[ties] += np.random.poisson(1.0, np.sum(ties))
            home_sims[ties] += np.random.poisson(1.0, np.sum(ties))
            ties = away_sims == home_sims
            
        # Clip extreme outliers
        away_sims = np.clip(away_sims, 0, 22)
        home_sims = np.clip(home_sims, 0, 22)
        
        away_wins = np.sum(away_sims > home_sims)
        home_wins = np.sum(home_sims > away_sims)
        
        away_prob = float(away_wins / iterations)
        home_prob = float(home_wins / iterations)
        edge = float(abs(home_prob - away_prob))
        
        # Save to database
        cursor.execute('''
            UPDATE Model_Forecasts 
            SET away_prob = ?, home_prob = ?, predicted_edge = ?, predicted_home_runs = ?, predicted_away_runs = ?
            WHERE game_pk = ?
        ''', (away_prob, home_prob, edge, float(home_lambda), float(away_lambda), game_pk))
        
        print(f"[{away} @ {home}]")
        print(f"  -> Win Prob: {away} ({away_prob:.1%}) | {home} ({home_prob:.1%})")
        print(f"  -> Exp Runs: Away {away_lambda:.2f} - Home {home_lambda:.2f} | **Total Runs: {total_runs:.2f}**\n")

    conn.commit()
    conn.close()
    print("-" * 60)
    print("Simulation execution complete.")

if __name__ == "__main__":
    run_ultimate_monte_carlo()
