import sqlite3
import numpy as np

def run_ultimate_monte_carlo():
    print("Initializing Ultimate SOTA Monte Carlo Engine...")
    
    conn = sqlite3.connect('mlb_engine.db')
    cursor = conn.cursor()

    for col in ["predicted_edge REAL", "predicted_home_runs REAL", "predicted_away_runs REAL"]:
        try:
            cursor.execute(f"ALTER TABLE Model_Forecasts ADD COLUMN {col}")
        except sqlite3.OperationalError:
            pass

    cursor.execute('''
        SELECT d.game_pk, d.away_team, d.home_team, d.away_pitcher, d.home_pitcher, 
               d.air_density, d.uv_modifier,
               COALESCE(u.run_modifier, 1.0),
               COALESCE(am_away.catcher_framing_modifier, 1.0), COALESCE(am_away.bullpen_fatigue_modifier, 1.0),
               COALESCE(am_home.catcher_framing_modifier, 1.0), COALESCE(am_home.bullpen_fatigue_modifier, 1.0),
               COALESCE(bio_away.jet_lag_runs_penalty, 0.0), COALESCE(bio_home.jet_lag_runs_penalty, 0.0)
        FROM Daily_Lineups d
        LEFT JOIN Daily_Umpires u ON d.game_pk = u.game_pk
        LEFT JOIN Advanced_Metrics am_away ON d.away_team = am_away.team_name
        LEFT JOIN Advanced_Metrics am_home ON d.home_team = am_home.team_name
        LEFT JOIN Biological_Modifiers bio_away ON d.away_team = bio_away.team_name
        LEFT JOIN Biological_Modifiers bio_home ON d.home_team = bio_home.team_name
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
        (game_pk, away, home, away_pitcher, home_pitcher, air_density, uv_modifier, umpire_multiplier,
         away_framing, away_fatigue, home_framing, home_fatigue, away_jet_lag, home_jet_lag) = game
        
        density_multiplier = 1.000 + ((1.225 - air_density) * 1.5) if air_density else 1.000
        
        away_off_mod = get_metric('Dynamic_Modifiers', 'offensive_modifier', 'team_name', away, 1.000)
        home_off_mod = get_metric('Dynamic_Modifiers', 'offensive_modifier', 'team_name', home, 1.000)
        away_pitch_mod = get_metric('Dynamic_Modifiers', 'pitching_modifier', 'team_name', away, 1.000)
        home_pitch_mod = get_metric('Dynamic_Modifiers', 'pitching_modifier', 'team_name', home, 1.000)

        home_starter_xera = get_metric('Pitcher_Stats', 'est_era', 'last_name', home_pitcher.split(' ')[-1], 4.30)
        away_starter_xera = get_metric('Pitcher_Stats', 'est_era', 'last_name', away_pitcher.split(' ')[-1], 4.30)
        
        away_ops = get_metric('Team_Offense', 'ops', 'team_name', away, 0.720)
        home_ops = get_metric('Team_Offense', 'ops', 'team_name', home, 0.720)
        
        home_bullpen = get_metric('Team_Bullpen', 'team_era', 'team_name', home, 4.00)
        away_bullpen = get_metric('Team_Bullpen', 'team_era', 'team_name', away, 4.00)
        
        park_factor = get_metric('Park_Factors', 'run_factor', 'home_team', home, 1.000)
        
        away_ops_mult = (away_ops / 0.720) * away_off_mod
        home_ops_mult = (home_ops / 0.720) * home_off_mod
        
        adj_home_starter_xera = home_starter_xera * home_pitch_mod * home_framing
        adj_away_starter_xera = away_starter_xera * away_pitch_mod * away_framing
        
        adj_home_bullpen = home_bullpen * home_pitch_mod * home_fatigue
        adj_away_bullpen = away_bullpen * away_pitch_mod * away_fatigue
        
        global_multiplier = park_factor * density_multiplier * umpire_multiplier * uv_modifier
        
        away_lambda_starter = (adj_home_starter_xera * away_ops_mult * global_multiplier) * 0.66
        away_lambda_bullpen = (adj_home_bullpen * away_ops_mult * global_multiplier) * 0.33
        away_lambda_total = away_lambda_starter + away_lambda_bullpen + home_jet_lag
        
        home_lambda_starter = (adj_away_starter_xera * home_ops_mult * global_multiplier) * 0.66
        home_lambda_bullpen = (adj_away_bullpen * home_ops_mult * global_multiplier) * 0.33
        home_lambda_total = home_lambda_starter + home_lambda_bullpen + away_jet_lag
        
        iterations = 50000
        dispersion_factor = 1.35 
        
        away_var = away_lambda_total * dispersion_factor
        home_var = home_lambda_total * dispersion_factor
        
        away_p = away_lambda_total / away_var
        away_n = (away_lambda_total ** 2) / (away_var - away_lambda_total)
        
        home_p = home_lambda_total / home_var
        home_n = (home_lambda_total ** 2) / (home_var - home_lambda_total)
        
        away_sims = np.random.negative_binomial(away_n, away_p, iterations)
        home_sims = np.random.negative_binomial(home_n, home_p, iterations)
        
        ties = away_sims == home_sims
        while np.any(ties):
            away_extra = np.random.poisson((adj_home_bullpen / 3) + 0.9, np.sum(ties))
            home_extra = np.random.poisson((adj_away_bullpen / 3) + 0.9, np.sum(ties))
            away_sims[ties] += away_extra
            home_sims[ties] += home_extra
            ties = away_sims == home_sims
            
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
        print(f"SIM (50k SOTA): {away} ({away_prob:.1%}) @ {home} ({home_prob:.1%}) | Edge: {edge:.3f}")

    conn.commit()
    conn.close()
    print("-" * 60)

if __name__ == "__main__":
    run_ultimate_monte_carlo()
