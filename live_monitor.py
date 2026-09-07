import os
import sqlite3
import requests
from datetime import datetime

# Uses your GitHub Secret, with your specific topic as the fallback
NTFY_TOPIC = os.getenv("NTFY_TOPIC", "mlb-alv-alerts-8899")

def init_alerts_table(cursor):
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS Live_Alerts_Sent (
            game_pk INTEGER,
            bet_key TEXT PRIMARY KEY,
            bet_recommended TEXT,
            sent_at TEXT
        );
    ''')

def send_ntfy_alert(title, message, priority="high", tags="baseball,moneybag"):
    url = f"https://ntfy.sh/{NTFY_TOPIC}"
    try:
        requests.post(
            url,
            data=message.encode("utf-8"),
            headers={
                "Title": title,
                "Priority": priority,
                "Tags": tags
            },
            timeout=10
        )
        print(f"[NTFY SENT] {title} -> {message}")
    except Exception as e:
        print(f"Failed to post to ntfy: {e}")

def run_live_monitor():
    print(f"Starting Live Monitor on topic: {NTFY_TOPIC}")
    conn = sqlite3.connect("mlb_engine.db")
    cursor = conn.cursor()
    init_alerts_table(cursor)

    # 1. Enforce the Absolute Live Verification (ALV) Mandate
    try:
        cursor.execute('''
            SELECT d.game_pk, d.away_team, d.home_team, d.air_density, 
                   u.home_plate_umpire, u.run_modifier,
                   f.home_prob, f.away_prob, f.predicted_home_runs, f.predicted_away_runs
            FROM Daily_Lineups d
            INNER JOIN Daily_Umpires u ON d.game_pk = u.game_pk
            INNER JOIN Model_Forecasts f ON d.game_pk = f.game_pk
            WHERE d.lineup_status = 'Confirmed'
              AND u.home_plate_umpire NOT IN ('Unknown / TBD', 'Unknown', '')
              AND d.air_density IS NOT NULL
        ''')
        verified_games = cursor.fetchall()
    except sqlite3.OperationalError as e:
        print(f"Database error (ALV conditions not met or tables missing): {e}")
        send_ntfy_alert("GitHub Test Failed", f"Database error encountered: {e}", priority="high", tags="warning")
        conn.close()
        return

    if not verified_games:
        print("No games currently meet the strict ALV criteria.")
        send_ntfy_alert("GitHub Pipeline Connected", "The live monitor executed perfectly on GitHub, but no games are currently live meeting the ALV criteria.", priority="default", tags="white_check_mark")
        conn.close()
        return

    for game in verified_games:
        (game_pk, away, home, air_density, hp_ump, ump_mod,
         home_prob, away_prob, exp_h_runs, exp_a_runs) = game

        live_url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
        try:
            live_data = requests.get(live_url, timeout=12).json()
        except Exception:
            continue

        game_state = live_data.get("gameData", {}).get("status", {}).get("abstractGameState")
        if game_state != "Live":
            continue

        linescore = live_data.get("liveData", {}).get("linescore", {})
        current_inning = linescore.get("currentInning", 1)
        
        # Halt all evaluations once the 9th inning begins
        if current_inning >= 9:
            continue
            
        is_top = linescore.get("isTopInning", True)
        half_str = "Top" if is_top else "Bot"
        home_score = linescore.get("teams", {}).get("home", {}).get("runs", 0)
        away_score = linescore.get("teams", {}).get("away", {}).get("runs", 0)
        total_live_runs = home_score + away_score
        projected_total = (exp_h_runs or 4.0) + (exp_a_runs or 4.0)

        # Trigger Scenario A: High-Run Shootout Validation
        if total_live_runs >= 3 and current_inning <= 3 and projected_total >= 10.5:
            bet_key = f"{game_pk}_OVER_EXPANSION"
            cursor.execute("SELECT 1 FROM Live_Alerts_Sent WHERE bet_key = ?", (bet_key,))
            if not cursor.fetchone():
                title = f"LIVE BET ALERT: Over Total Runs ({away} @ {home})"
                body = (
                    f"Game Status: {away_score}-{home_score} ({half_str} {current_inning})\n"
                    f"Model Projected Total: {projected_total:.2f} Runs\n"
                    f"ALV Context: Umpire {hp_ump} ({ump_mod:.3f}x), Air Density {air_density}\n"
                    f"ACTIONABLE BET: Bet LIVE OVER Total Runs"
                )
                send_ntfy_alert(title, body)
                cursor.execute("INSERT INTO Live_Alerts_Sent VALUES (?, ?, ?, ?)",
                               (game_pk, bet_key, "LIVE OVER", datetime.now().isoformat()))
                conn.commit()

        # Trigger Scenario B: Pre-Game Model Dominance / Buy-Low Window
        favored_team = home if home_prob > 0.58 else (away if away_prob > 0.58 else None)
        favored_prob = max(home_prob, away_prob)

        # Extended scanning logic: Evaluates all the way through the 8th inning
        if favored_team and 2 <= current_inning <= 8:
            trailing_or_tied = (favored_team == home and home_score <= away_score) or \
                               (favored_team == away and away_score <= home_score)
            run_deficit = abs(home_score - away_score)

            if trailing_or_tied and run_deficit <= 2:
                bet_key = f"{game_pk}_FAV_VALUE_{favored_team}_{current_inning}"
                cursor.execute("SELECT 1 FROM Live_Alerts_Sent WHERE bet_key = ?", (bet_key,))
                if not cursor.fetchone():
                    title = f"LIVE BET ALERT: {favored_team} Live Moneyline"
                    body = (
                        f"Game Status: {away} {away_score} @ {home} {home_score} ({half_str} {current_inning})\n"
                        f"Engine Pre-Game Probability: {favored_prob:.1%}\n"
                        f"Umpire: {hp_ump} | Deficit: {run_deficit} run(s)\n"
                        f"ACTIONABLE BET: {favored_team} Live Moneyline (Regression Buy-Low)"
                    )
                    send_ntfy_alert(title, body)
                    cursor.execute("INSERT INTO Live_Alerts_Sent VALUES (?, ?, ?, ?)",
                                   (game_pk, bet_key, f"{favored_team} Live ML", datetime.now().isoformat()))
                    conn.commit()

    conn.close()
    print("Live monitor execution complete.")

if __name__ == "__main__":
    run_live_monitor()
