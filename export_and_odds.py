import sqlite3
import os
import requests
from datetime import datetime

NTFY_TOPIC = os.getenv("NTFY_TOPIC") or "mlb-alv-alerts-8899"

def prob_to_american(prob):
    if prob >= 0.5:
        return int(-round((prob / max(0.001, (1.0 - prob))) * 100))
    else:
        return int(round(((1.0 - prob) / max(0.001, prob)) * 100))

def export_forecasts_and_check_odds():
    print(f"[{datetime.now()}] Generating Betting Slips & Exporting Models...")
    conn = sqlite3.connect('mlb_engine.db', timeout=30)
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA busy_timeout=10000;")

    query = '''
    SELECT 
        m.game_pk, m.away_team, m.home_team, 
        m.away_prob, m.home_prob, m.predicted_edge,
        m.predicted_away_runs, m.predicted_home_runs,
        COALESCE(d.away_pitcher, 'TBD') as away_sp,
        COALESCE(d.home_pitcher, 'TBD') as home_sp,
        COALESCE(f.f5_away_prob, 0.0), COALESCE(f.f5_home_prob, 0.0),
        COALESCE(f.f5_exp_away_runs, 0.0), COALESCE(f.f5_exp_home_runs, 0.0)
    FROM Model_Forecasts m
    LEFT JOIN Daily_Lineups d ON m.game_pk = d.game_pk
    LEFT JOIN F5_Forecasts f ON m.game_pk = f.game_pk
    ORDER BY m.predicted_edge DESC
    '''

    try:
        cursor.execute(query)
        games = cursor.fetchall()
    except Exception as e:
        print(f"Query note: {e}")
        games = []

    conn.close()

    now_str = datetime.now().strftime('%A, %B %d, %Y')
    lines = [
        f"# ⚾ MLB Betting Slips & Model Card",
        f"*Autonomous Forecast for Slate: `{now_str}`*\n"
    ]

    tier_1_picks = []

    if not games:
        lines.append("### ℹ️ No active MLB games are currently scheduled or pending simulation for today's slate.\n")
    else:
        lines.append("## 🎟️ Primary Value Bets (Full Game Moneyline)")
        lines.append("| Matchup | Best Pick | Fair Odds | Edge | Projected Score | Pitchers |")
        lines.append("| :--- | :---: | :---: | :---: | :---: | :--- |")

        for g in games:
            pk, away, home, a_prob, h_prob, edge, a_runs, h_runs, a_sp, h_sp, f5_a, f5_h, f5_a_runs, f5_h_runs = g
            
            if h_prob >= a_prob:
                pick_team = home
                pick_prob = h_prob
            else:
                pick_team = away
                pick_prob = a_prob

            fair_odds = prob_to_american(pick_prob)
            odds_str = f"{fair_odds:+d}"
            edge_badge = f"🔥 **+{edge:.1%}**" if edge >= 0.05 else f"+{edge:.1%}"

            lines.append(f"| {away} @ {home} | **{pick_team}** | `{odds_str}` | {edge_badge} | {a_runs:.1f} - {h_runs:.1f} | {a_sp} vs {h_sp} |")

            if edge >= 0.045:
                tier_1_picks.append(f"• **{pick_team} ML** ({odds_str}) | Edge: +{edge:.1%}")

        lines.append("\n## ⏱️ First 5 Innings (F5 Isolations)")
        lines.append("| Matchup | F5 Favorite | F5 Odds | Projected F5 Total |")
        lines.append("| :--- | :---: | :---: | :---: |")

        for g in games:
            _, away, home, _, _, _, _, _, _, _, f5_a, f5_h, f5_a_r, f5_h_r = g
            if f5_a == 0.0 and f5_h == 0.0:
                continue
            fav = home if f5_h > f5_a else away
            f5_p = max(f5_h, f5_a)
            f5_odds = prob_to_american(f5_p)
            total_f5 = f5_a_r + f5_h_r
            lines.append(f"| {away} @ {home} | **{fav}** | `{f5_odds:+d}` | {total_f5:.1f} Runs |")

    content = "\n".join(lines)

    # GUARANTEED WRITE: Prevents any missing pathspec in Git
    with open("PREDICTIONS_TODAY.md", "w") as f:
        f.write(content)

    summary_path = os.environ.get('GITHUB_STEP_SUMMARY')
    if summary_path:
        with open(summary_path, "a") as f:
            f.write("\n" + content + "\n")

    # Push to mobile via ntfy
    if tier_1_picks:
        slip_msg = "🎯 **TODAY'S VALUE BETS**\n\n" + "\n".join(tier_1_picks)
        try:
            requests.post(
                f"https://ntfy.sh/{NTFY_TOPIC}",
                data=slip_msg.encode('utf-8'),
                headers={"Title": "⚾ MLB Actionable Betting Slips", "Priority": "4", "Markdown": "yes", "Tags": "ticket,baseball"},
                timeout=10
            )
            print(f"[ntfy] Value picks dispatched to ntfy.sh/{NTFY_TOPIC}")
        except Exception as e:
            print(f"ntfy error: {e}")

if __name__ == "__main__":
    export_forecasts_and_check_odds()
