import sqlite3
from pybaseball import statcast_pitcher_expected_stats

def fetch_statcast_features():
    print("FanGraphs Turnstile detected. Rerouting to MLB Statcast...")
    print("Fetching predictive exit-velocity metrics (xERA, xwOBA)...")
    
    try:
        # Fetch 2026 Statcast expected stats (minimum 50 batters faced to filter noise)
        stats = statcast_pitcher_expected_stats(2026, 50)
        
        print(f"Bypass successful! Retrieved advanced profiles for {len(stats)} pitchers.")
        
        # Connect to the local MLB Engine database
        conn = sqlite3.connect('mlb_engine.db')
        
        # Save the dataframe directly into our SQLite database
        stats.to_sql('Pitcher_Stats', conn, if_exists='replace', index=False)
        
        print("Statcast features securely saved to the 'Pitcher_Stats' table.")
        conn.close()
        
    except Exception as e:
        print(f"Error retrieving Statcast data: {e}")

if __name__ == "__main__":
    fetch_statcast_features()



