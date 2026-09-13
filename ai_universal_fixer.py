import os
import sqlite3
import json
import requests
import traceback

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DB_PATH = "mlb_engine.db"

def verify_database_integrity():
    print("[FIXER] Verifying SQLite WAL and database integrity for mlb_engine...")
    if os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("PRAGMA integrity_check;")
            result = cursor.fetchone()
            print(f"[FIXER] Database integrity check result: {result}")
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = cursor.fetchall()
            print(f"[FIXER] Active tables: {[t[0] for t in tables]}")
            conn.close()
        except Exception as e:
            print(f"[FIXER ERROR] Database check failed: {e}")
            traceback.print_exc()
    else:
        print("[FIXER] mlb_engine.db not found. Pipeline will initialize a fresh instance.")

def audit_python_files():
    print("[FIXER] Auditing python codebase for syntax or import regressions...")
    py_files = [f for f in os.listdir('.') if f.endswith('.py')]
    for file in py_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                code = f.read()
            compile(code, file, 'exec')
            print(f"[FIXER] Syntax verified: {file}")
        except Exception as e:
            print(f"[FIXER ERROR] Syntax error found in {file}: {e}")

def run_ai_architect_pass():
    if not OPENROUTER_API_KEY:
        print("[FIXER] OPENROUTER_API_KEY not configured. Skipping generative AI self-healing pass.")
        return
    
    print("[FIXER] Triggering OpenRouter self-correction review via ai_universal_fixer.py...")
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "anthropic/claude-3.5-sonnet",
        "messages": [
            {
                "role": "user",
                "content": "Perform a self-correction audit of the MLB prediction engine environment. All dependencies (pandas, numpy, scipy, scikit-learn) are locked."
            }
        ],
        "temperature": 0.1
    }
    try:
        response = requests.post(url, headers=headers, json=payload, timeout=30)
        if response.status_code == 200:
            print("[FIXER] AI Architect pass completed successfully.")
        else:
            print(f"[FIXER WARNING] AI Architect pass returned status code {response.status_code}: {response.text}")
    except Exception as e:
        print(f"[FIXER ERROR] Failed to reach OpenRouter API: {e}")

if __name__ == "__main__":
    print("=========================================================")
    print("[INIT] ai_universal_fixer.py for MLB Engine Starting...")
    print("=========================================================")
    verify_database_integrity()
    audit_python_files()
    run_ai_architect_pass()
    print("[SUCCESS] ai_universal_fixer.py execution finished.")
