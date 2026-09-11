import os
import sys
import sqlite3
import requests
from datetime import datetime

NTFY_TOPIC = os.getenv("NTFY_TOPIC") or "mlb-alv-alerts-8899"

def send_mobile_alert(title, message, priority="default"):
    """Dispatches zero-cost instant push alerts via ntfy."""
    try:
        url = f"https://ntfy.sh/{NTFY_TOPIC}"
        headers = {
            "Title": title,
            "Priority": priority,
            "Tags": "robot,baseball"
        }
        res = requests.post(
            url,
            data=message.encode('utf-8'),
            headers=headers,
            timeout=10
        )
        print(f"[WATCHDOG ALERT] Dispatched notification to ntfy.sh/{NTFY_TOPIC} (Status: {res.status_code})")
    except Exception as e:
        print(f"Failed to dispatch mobile notification: {e}")

def run_health_checks():
    print(f"[{datetime.now()}] Running Autonomous Watchdog Health Audit...")
    db_file = "mlb_engine.db"

    # 1. Verify database file existence
    if not os.path.exists(db_file):
        send_mobile_alert("🚨 Engine Alert: Database Missing", "mlb_engine.db was not found after execution!", priority="urgent")
        sys.exit(1)

    # 2. Verify SQLite schema and page integrity
    try:
        conn = sqlite3.connect(db_file, timeout=15)
        cursor = conn.cursor()
        cursor.execute("PRAGMA integrity_check;")
        status = cursor.fetchone()[0]
        if status != "ok":
            send_mobile_alert("🚨 Database Corruption", f"PRAGMA integrity_check failed with status: {status}", priority="urgent")
            conn.close()
            sys.exit(1)

        cursor.execute("SELECT COUNT(*) FROM Model_Forecasts")
        forecast_count = cursor.fetchone()[0]

        cursor.execute("SELECT COUNT(*) FROM Daily_Lineups WHERE lineup_status = 'Confirmed'")
        confirmed_lineups = cursor.fetchone()[0]

        conn.close()
        print(f"[HEALTH OK] {forecast_count} Forecasts | {confirmed_lineups} Confirmed Lineups.")

    except Exception as e:
        send_mobile_alert("🚨 Watchdog Crash", f"Database check threw an exception: {e}", priority="high")
        sys.exit(1)

    # 3. Clean transient SQLite WAL lock files before Git push
    for lock in ["mlb_engine.db-wal", "mlb_engine.db-shm"]:
        if os.path.exists(lock):
            try:
                os.remove(lock)
                print(f"Removed transient lock file: {lock}")
            except Exception:
                pass

if __name__ == "__main__":
    run_health_checks()
