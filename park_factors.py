import sqlite3

def load_park_factors():
    print("Executing Option C: Injecting Stadium Park Factors...")
    
    # Rolling Run Factors (Base 1.000 = League Average)
    # > 1.000 favors hitters, < 1.000 favors pitchers
    park_factors = {
        "Colorado Rockies": 1.314,
        "Cincinnati Reds": 1.109,
        "Boston Red Sox": 1.087,
        "Kansas City Royals": 1.042,
        "Atlanta Braves": 1.037,
        "Chicago White Sox": 1.035,
        "Texas Rangers": 1.025,
        "Los Angeles Angels": 1.018,
        "Philadelphia Phillies": 1.016,
        "Baltimore Orioles": 1.014,
        "Houston Astros": 1.011,
        "Washington Nationals": 1.009,
        "Los Angeles Dodgers": 1.007,
        "Toronto Blue Jays": 1.003,
        "Minnesota Twins": 1.001,
        "Milwaukee Brewers": 0.998,
        "Chicago Cubs": 0.995,
        "Arizona Diamondbacks": 0.992,
        "Pittsburgh Pirates": 0.988,
        "Miami Marlins": 0.985,
        "San Francisco Giants": 0.983,
        "New York Yankees": 0.982,
        "Detroit Tigers": 0.980,
        "Tampa Bay Rays": 0.978,
        "St. Louis Cardinals": 0.975,
        "New York Mets": 0.972,
        "Cleveland Guardians": 0.968,
        "Athletics": 0.965, 
        "San Diego Padres": 0.958,
        "Seattle Mariners": 0.923
    }

    conn = sqlite3.connect('mlb_engine.db')
    cursor = conn.cursor()

    # Create the Park Factors table
    cursor.executescript('''
    CREATE TABLE IF NOT EXISTS Park_Factors (
        home_team TEXT PRIMARY KEY,
        run_factor REAL
    );
    ''')

    print("Mapping stadium environments...")
    
    for team, factor in park_factors.items():
        cursor.execute('''
        INSERT OR REPLACE INTO Park_Factors (home_team, run_factor)
        VALUES (?, ?)
        ''', (team, factor))
        
        print(f"Logged {team} (Home): {factor:.3f}x Run Multiplier")

    conn.commit()
    print("-" * 50)
    print("Park Factors safely locked into mlb_engine.db")
    conn.close()

if __name__ == "__main__":
    load_park_factors()
