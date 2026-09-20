import sys
import sqlite3
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

def evaluate_fluid_pregame_triggers():
    conn = sqlite3.connect("mlb_engine.db")
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    now_utc = datetime.now(timezone.utc)
    current_et = now_utc.astimezone(ZoneInfo("America/New_York")).strftime("%I:%M %p EDT")
    print(f"[GATEKEEPER TICK] Current System Time: {current_et} ({now_utc.isoformat()})")

    games = c.execute("""
        SELECT game_pk, away_team, home_team, 
               COALESCE(lineup_status, status, '') as match_status, 
               game_datetime_utc, game_time_et 
        FROM Daily_Lineups
    """).fetchall()

    if not games:
        print("[GATEKEEPER] Daily_Lineups table empty. Triggering slate load.")
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

        forecast_count = c.execute(
            "SELECT COUNT(*) FROM Batter_Hit_Forecasts WHERE game_pk = ?", (pk,)
        ).fetchone()[0]

        print(f"  • {g['away_team']} @ {g['home_team']} (PK: {pk}) | Start: {time_et} | In: {minutes_until_first_pitch:.1f} min | Cached Props: {forecast_count}")

        # Trigger if starting within 45 minutes or recently started (< 20 mins ago) without existing projections
        if -20.0 <= minutes_until_first_pitch <= 45.0 and forecast_count == 0:
            print(f"    --> [TRIGGER] Impending first pitch requires prop synthesis.")
            pending_synthesis.append(pk)

    conn.close()

    if pending_synthesis:
        print(f"[GATEKEEPER STATUS] Authorized pipeline execution for {len(pending_synthesis)} game(s).")
        sys.exit(0)
    else:
        print("[GATEKEEPER STATUS] No pending games inside the trigger window. Pipeline skipped.")
        sys.exit(2)

if __name__ == "__main__":
    evaluate_fluid_pregame_triggers()
