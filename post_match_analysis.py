import sqlite3
import requests
from datetime import datetime

def run_post_match_analysis():
    print("Executing Factual Post-Mortem: Pulling daily outcomes...")
    today = datetime.now().strftime('%Y-%m-%d')
    url = f"https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={today}"
    
    response = requests.get(url).json()
    conn = sqlite3.connect('mlb_engine.db')
    cursor = conn.cursor()
    
    cursor.executescript('''
    CREATE TABLE IF NOT EXISTS Post_Match_Analysis (
        game_pk INTEGER PRIMARY KEY,
        actual_winner TEXT,
        home_score INTEGER,
        away_score INTEGER,
        model_correct INTEGER
    );
    ''')
    
    for date_data in response.get('dates', []):
        for game in date_data.get('games', []):
            game_pk = game['gamePk']
            status = game['status']['abstractGameState']
            
            if status == 'Final':
                teams = game.get('teams', {})
                home_team = teams['home']['team']['name']
                away_team = teams['away']['team']['name']
                home_score = teams['home'].get('score', 0)
                away_score = teams['away'].get('score', 0)
                
                actual_winner = home_team if home_score > away_score else away_team
                
                cursor.execute("SELECT home_prob, away_prob FROM Model_Forecasts WHERE game_pk = ?", (game_pk,))
                forecast = cursor.fetchone()
                
                if forecast:
                    home_prob, away_prob = forecast
                    predicted_winner = home_team if home_prob > away_prob else away_team
                    model_correct = 1 if predicted_winner == actual_winner else 0
                    
                    cursor.execute('''
                    INSERT OR REPLACE INTO Post_Match_Analysis (game_pk, actual_winner, home_score, away_score, model_correct)
                    VALUES (?, ?, ?, ?, ?)
                    ''', (game_pk, actual_winner, home_score, away_score, model_correct))
                    
                    print(f"Post-Mortem Game {game_pk}: Winner: {actual_winner} ({home_score}-{away_score}) | Model Correct: {bool(model_correct)}")

    conn.commit()
    conn.close()
    print("Post-match analysis completed simulation-free.")

if __name__ == "__main__":
    run_post_match_analysis()
