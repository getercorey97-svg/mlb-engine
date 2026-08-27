import sqlite3

conn = sqlite3.connect('mlb_engine.db')
cursor = conn.cursor()

cursor.execute("SELECT game_pk, home_team, away_team, home_prob, away_prob, predicted_edge, predicted_home_runs, predicted_away_runs FROM Model_Forecasts")
for row in cursor.fetchall():
    print(f"Game PK: {row[0]} | {row[2]} ({row[4]:.1%}) @ {row[1]} ({row[3]:.1%}) | Edge: {row[5]:.3f} | Runs: {row[7]:.2f}-{row[6]:.2f}")

conn.close()
