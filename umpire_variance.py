import sqlite3
import requests
from datetime import datetime

KNOWN_UMPIRES = {
    "Derek Thomas": 0.982, "Tyler Jones": 1.003, "Jonathan Parra": 1.011,
    "Angel Hernandez": 1.065, "Pat Hoberg": 0.942, "Doug Eddings": 1.034,
    "CB Bucknor": 1.048, "Hunter Wendelstedt": 1.031, "Mark Carlson": 0.965,
    "Dan Bellino": 0.970, "Bill Miller": 0.985, "Ted Barrett": 0.990,
    "Ron Kulpa": 1.025, "Laz Diaz": 1.035, "Alan Porter": 0.975,
    "Default Umpire": 1.000
}

def execute_umpire_variance_pipeline():
    """Ingests assigned Home Plate Umpires with automated schema migration."""
    print("=" * 65)
    print(f"[{datetime.now()}] Initializing Umpire Variance & Lock Pipeline...")
    print("=" * 65)

    conn = sqlite3.connect('mlb_engine.db', timeout=30)
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA busy_timeout=10000;")

    cursor.executescript('''
    CREATE TABLE IF NOT EXISTS Daily_Umpires (
        game_pk INTEGER PRIMARY KEY,
        home_plate_umpire TEXT,
        run_modifier REAL DEFAULT 1.00,
        umpire_locked INTEGER DEFAULT 0,
        updated_at TEXT
    );
    ''')

    # Migration: Ensure umpire_locked exists
    cursor.execute("PRAGMA table_info(Daily_Umpires);")
    cols = [c[1] for c in cursor.fetchall()]
    if 'umpire_locked' not in cols:
        cursor.execute("ALTER TABLE Daily_Umpires ADD COLUMN umpire_locked INTEGER DEFAULT 0;")
    if 'updated_at' not in cols:
        cursor.execute("ALTER TABLE Daily_Umpires ADD COLUMN updated_at TEXT;")
    conn.commit()

    today = datetime.now().strftime('%Y-%m-%d')
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today}&hydrate=officials"

    try:
        data = requests.get(url, timeout=12).json()
    except Exception as e:
        print(f"[UMPIRE WARNING] Failed to connect to MLB Officials API: {e}")
        conn.close()
        return

    now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for date_item in data.get('dates', []):
        for game in date_item.get('games', []):
            game_pk = game['gamePk']
            officials = game.get('officials', [])

            cursor.execute("SELECT home_plate_umpire, run_modifier, umpire_locked FROM Daily_Umpires WHERE game_pk = ?", (game_pk,))
            existing = cursor.fetchone()

            hp_umpire = "Unknown / TBD"
            for official in officials:
                if official.get('officialType') == 'Home Plate':
                    hp_umpire = official.get('person', {}).get('fullName', 'Unknown / TBD')
                    break

            if hp_umpire != "Unknown / TBD":
                mod = KNOWN_UMPIRES.get(hp_umpire, 1.000)
                is_locked = 1
                status_label = f"LOCKED: {hp_umpire} ({mod:.3f}x)"
            else:
                if existing and len(existing) >= 3 and existing[2] == 1:
                    hp_umpire = existing[0]
                    mod = existing[1]
                    is_locked = 1
                    status_label = f"PRESERVED LOCKED: {hp_umpire} ({mod:.3f}x)"
                else:
                    mod = 1.000
                    is_locked = 0
                    status_label = "AWAITING OFFICIAL LINEUP CARD (1.000x fallback)"

            cursor.execute('''
            INSERT OR REPLACE INTO Daily_Umpires 
            (game_pk, home_plate_umpire, run_modifier, umpire_locked, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ''', (game_pk, hp_umpire, mod, is_locked, now_ts))

            print(f"Game {game_pk} | HP Umpire: {status_label}")

    conn.commit()
    conn.close()
    print("[SUCCESS] Umpire variance mapping completed.")

if __name__ == "__main__":
    execute_umpire_variance_pipeline()
