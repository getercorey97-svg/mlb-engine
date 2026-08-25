import sqlite3

def init_umpire_tendencies():
    print("Executing Extraction: Umpire Tendencies Baseline...")
    
    conn = sqlite3.connect('mlb_engine.db')
    cursor = conn.cursor()
    
    cursor.executescript('''
    CREATE TABLE IF NOT EXISTS Umpire_Tendencies (
        umpire_name TEXT PRIMARY KEY,
        zone_bias TEXT,
        run_adjustment REAL
    );
    ''')

    # Seed baseline neutral umpire profiles
    cursor.execute("INSERT OR REPLACE INTO Umpire_Tendencies (umpire_name, zone_bias, run_adjustment) VALUES ('Default Umpire', 'Neutral', 1.0)")
    
    conn.commit()
    conn.close()
    print("Umpire tendencies matrix initialized.")

if __name__ == "__main__":
    init_umpire_tendencies()
