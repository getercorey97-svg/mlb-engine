import os
import sys
import sqlite3
import requests
from datetime import datetime, timedelta

NTFY_TOPIC = os.getenv("NTFY_TOPIC", "mlb-alv-alerts-8899")

def get_db_connection():
    conn = sqlite3.connect('mlb_engine.db', timeout=30)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=10000;")
    return conn

def send_ntfy_alert(title, message, priority=3, tags="baseball,chart_with_upwards_trend"):
    headers = {
        "Title": title,
        "Priority": str(priority),
        "Markdown": "yes",
        "Tags": tags
    }
    url = f"https://ntfy.sh/{NTFY_TOPIC}"
    try:
        requests.post(url, data=message.encode('utf-8'), headers=headers, timeout=10)
        print(f"[ntfy Pushed] {title}")
    except Exception as e:
        print(f"[ntfy Error]: {e}")

def run_live_cycle():
    # Multi-date polling prevents West Coast evening games from dropping at UTC midnight
    now_utc = datetime.utcnow()
    dates_to_poll = [
        (now_utc - timedelta(days=1)).strftime('%Y-%m-%d'),
        now_utc.strftime('%Y-%m-%d')
    ]

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.executescript('''
    CREATE TABLE IF NOT EXISTS Live_Alert_Ledger (
        alert_id INTEGER PRIMARY KEY AUTOINCREMENT,
        game_pk INTEGER,
        milestone TEXT,
        inning_state TEXT,
        score_state TEXT,
        alerted_at TEXT
    );
    ''')
    conn.commit()

    active_games_found = 0

    for date_str in dates_to_poll:
        sched_url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date_str}"
        try:
            sched_data = requests.get(sched_url, timeout=10).json()
        except Exception:
            continue

        for date_item in sched_data.get('dates', []):
            for game in date_item.get('games', []):
                game_pk = game['gamePk']
                abstract_state = game['status']['abstractGameState']

                if abstract_state not in ["Live", "In Progress", "Final"]:
                    continue

                feed_url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live"
                try:
                    live_feed = requests.get(feed_url, timeout=10).json()
                except Exception:
                    continue

                linescore = live_feed.get('liveData', {}).get('linescore', {})
                current_inning = linescore.get('currentInning', 1)
                is_top = linescore.get('isTopInning', True)
                outs = linescore.get('outs', 0)

                teams = live_feed.get('gameData', {}).get('teams', {})
                away_team = teams.get('away', {}).get('name', 'Away')
                home_team = teams.get('home', {}).get('name', 'Home')
                away_score = linescore.get('teams', {}).get('away', {}).get('runs', 0)
                home_score = linescore.get('teams', {}).get('home', {}).get('runs', 0)

                cursor.execute("SELECT milestone FROM Live_Alert_Ledger WHERE game_pk = ?", (game_pk,))
                recorded = [row[0] for row in cursor.fetchall()]

                milestone = None
                if abstract_state == "Final" and "FINAL" not in recorded:
                    milestone = "FINAL"
                elif abstract_state in ["Live", "In Progress"]:
                    if current_inning >= 8 and "Q3_MARK" not in recorded:
                        milestone = "Q3_MARK"
                    elif current_inning >= 6 and "HALF_MARK" not in recorded:
                        milestone = "HALF_MARK"
                    elif current_inning >= 3 and "Q1_MARK" not in recorded:
                        milestone = "Q1_MARK"
                    elif current_inning == 1 and "START" not in recorded:
                        milestone = "START"

                if not milestone:
                    continue

                active_games_found += 1

                cursor.execute("SELECT home_prob, away_prob, predicted_home_runs, predicted_away_runs FROM Model_Forecasts WHERE game_pk = ?", (game_pk,))
                fc = cursor.fetchone()
                base_h = fc[2] if (fc and fc[2]) else 4.25
                base_a = fc[3] if (fc and fc[3]) else 4.10

                away_outs_rem = max(0, 27 - (((current_inning - 1) * 3) + outs)) if is_top else max(0, 27 - (current_inning * 3))
                home_outs_rem = max(0, 27 - ((current_inning - 1) * 3)) if is_top else max(0, 27 - (((current_inning - 1) * 3) + outs))

                synth_a = away_score + (base_a * (away_outs_rem / 27.0))
                synth_h = home_score + (base_h * (home_outs_rem / 27.0))
                denom = (synth_h ** 1.83) + (synth_a ** 1.83) + 0.0001
                live_h_prob = (synth_h ** 1.83) / denom

                ml_target = home_team if live_h_prob >= 0.53 else (away_team if live_h_prob <= 0.47 else "Pass / Neutral")

                msg = f"### {away_team} ({away_score}) @ {home_team} ({home_score})\n"
                msg += f"**State:** Inning {current_inning} ({'Top' if is_top else 'Bot'}), {outs} Outs | Milestone: {milestone}\n\n"
                msg += f"* **Projected Final:** {away_team} {synth_a:.1f} – {home_team} {synth_h:.1f}\n"
                msg += f"* **Live Win Probability:** {home_team} {live_h_prob:.1%} | {away_team} {(1.0 - live_h_prob):.1%}\n"
                msg += f"* **Live Actionable Target:** {ml_target}\n"

                prio = 4 if milestone in ["HALF_MARK", "FINAL"] else 3
                send_ntfy_alert(f"MLB Live: {away_team} @ {home_team} [{milestone}]", msg, priority=prio)

                cursor.execute('''
                    INSERT INTO Live_Alert_Ledger (game_pk, milestone, inning_state, score_state, alerted_at)
                    VALUES (?, ?, ?, ?, ?)
                ''', (game_pk, milestone, f"Inn {current_inning}", f"{away_score}-{home_score}", datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")))
                conn.commit()

    conn.close()
    print(f"Live monitor cycle finished. Dispatched alerts for {active_games_found} active milestones.")

if __name__ == "__main__":
    run_live_cycle()
