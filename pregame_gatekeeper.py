import sys
import sqlite3
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

def evaluate_pregame_triggers():
    conn = sqlite3.connect("mlb_engine.db")
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    now_utc = datetime.now(timezone.utc)
    current_et = now_utc.astimezone(ZoneInfo("America/New_York")).strftime("%I:%M %p EDT")
    print(f"[GATEKEEPER TICK] Current System Time: {current_et} ({now_utc.isoformat()})")

    games = c.execute("""
        SELECT game_pk, away_team, home_team, 
               COALESCE(lineup_status, status, '') AS match_status, 
               game_datetime_utc, game_time_et 
        FROM Daily_Lineups
    """).fetchall()

    if not games:
        print("[GATEKEEPER] No matchups found in Daily_Lineups. Triggering full slate load.")
        conn.close()
        sys.exit(0)

    pending_synthesis = []

    for g in games:
        pk = g["game_pk"]
        status = g["match_status"]
        dt_str = g["game_datetime_utc"]
        time_et = g["game_time_et"] or "TBD"

        if any(x in status.lower() for x in ["final", "game over", "completed"]):
            continue

        if not dt_str:
            continue

        try:
            start_utc = datetime.fromisoformat(dt_str)
        except Exception:
            continue

        minutes_until_first_pitch = (start_utc - now_utc).total_seconds() / 60.0

        # Check for confirmed batters in Daily_Batters
        confirmed_count = c.execute(
            "SELECT COUNT(*) FROM Daily_Batters WHERE game_pk = ? AND is_confirmed = 1", (pk,)
        ).fetchone()[0]

        # Check existing forecasts
        forecast_count = c.execute(
            "SELECT COUNT(*) FROM Batter_Hit_Forecasts WHERE game_pk = ?", (pk,)
        ).fetchone()[0]

        print(f"  • {g['away_team']} @ {g['home_team']} (PK: {pk}) | Start: {time_et} | In: {minutes_until_first_pitch:.1f}m | Confirmed: {confirmed_count}/18 | Props: {forecast_count}")

        # Trigger if game is within 45 minutes and lacks confirmed lineups or has no generated props
        if -15.0 <= minutes_until_first_pitch <= 45.0:
            if confirmed_count < 18 or forecast_count == 0:
                print(f"    --> [TRIGGER] Game starting soon requires confirmed lineup verification and props.")
                pending_synthesis.append(pk)

    conn.close()

    if pending_synthesis:
        print(f"[GATEKEEPER STATUS] Authorized execution for {len(pending_synthesis)} impending game(s).")
        sys.exit(0)
    else:
        print("[GATEKEEPER STATUS] No impending first pitches require updates. Pipeline skipped.")
        sys.exit(2)

if __name__ == "__main__":
    evaluate_pregame_triggers()
