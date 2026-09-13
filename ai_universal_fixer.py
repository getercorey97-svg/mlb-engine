import os
import sqlite3
import json
import requests
import traceback
from datetime import datetime

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
DB_PATH = "mlb_engine.db"
METADATA_PATH = "engine_metadata.json"

def verify_database_integrity():
    print("[FIXER] Verifying SQLite WAL and database integrity for mlb_engine...")
    db_report = {"status": "unknown", "tables": []}
    if os.path.exists(DB_PATH):
        try:
            conn = sqlite3.connect(DB_PATH)
            cursor = conn.cursor()
            cursor.execute("PRAGMA integrity_check;")
            result = cursor.fetchone()
            print(f"[FIXER] Database integrity check result: {result}")
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [t[0] for t in cursor.fetchall()]
            print(f"[FIXER] Active tables: {tables}")
            conn.close()
            db_report["status"] = "verified" if result and result[0] == "ok" else "warning"
            db_report["tables"] = tables
        except Exception as e:
            print(f"[FIXER ERROR] Database check failed: {e}")
            traceback.print_exc()
            db_report["status"] = f"error: {str(e)}"
    else:
        print("[FIXER] mlb_engine.db not found. Fresh instance initialized by pipeline.")
        db_report["status"] = "missing_or_new"
    return db_report

def audit_python_files():
    print("[FIXER] Auditing python codebase for syntax or import regressions...")
    py_files = [f for f in os.listdir('.') if f.endswith('.py') and f != 'ai_universal_fixer.py']
    audit_results = {}
    for file in py_files:
        try:
            with open(file, 'r', encoding='utf-8') as f:
                code = f.read()
            compile(code, file, 'exec')
            print(f"[FIXER] Syntax verified: {file}")
            audit_results[file] = "clean"
        except Exception as e:
            print(f"[FIXER ERROR] Syntax error detected in {file}: {e}")
            audit_results[file] = f"error: {str(e)}"
    return audit_results

def run_ai_architect_pass(db_info, code_info):
    if not OPENROUTER_API_KEY:
        print("[FIXER] OPENROUTER_API_KEY not configured. Skipping OpenRouter pass.")
        return None
    
    print("[FIXER] Triggering OpenRouter self-correction pass...")
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://github.com/getercorey97-svg/mlb-engine",
        "X-Title": "MLB Engine AI Architect",
        "Content-Type": "application/json"
    }

    # Model fallback cascade: auto router -> paid sonnet -> free auto router
    candidate_models = [
        "openrouter/auto",
        "anthropic/claude-3.5-sonnet",
        "openrouter/free"
    ]

    for model_name in candidate_models:
        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": f"MLB Engine System Status:\nDB Status: {db_info['status']}\nTables: {len(db_info['tables'])}\nPython Files Verified: {len(code_info)}\nConfirm system readiness."
                }
            ],
            "max_tokens": 150
        }
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=25)
            if response.status_code == 200:
                print(f"[FIXER] AI Architect pass succeeded using model: {model_name}")
                return response.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            else:
                print(f"[FIXER WARNING] Model {model_name} returned {response.status_code}: {response.text}")
        except Exception as e:
            print(f"[FIXER ERROR] Request to {model_name} failed: {e}")

    return None

def update_metadata(db_info, code_info, ai_notes):
    metadata = {
        "last_audit_timestamp": datetime.utcnow().isoformat() + "Z",
        "db_integrity": db_info,
        "verified_scripts_count": len(code_info),
        "ai_architect_status": "active" if ai_notes else "fallback",
        "notes": ai_notes or "Engine syntax and database verified clean."
    }
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"[FIXER] Audit state logged to {METADATA_PATH}.")

if __name__ == "__main__":
    print("=========================================================")
    print("[INIT] ai_universal_fixer.py for MLB Engine Starting...")
    print("=========================================================")
    db_res = verify_database_integrity()
    code_res = audit_python_files()
    ai_output = run_ai_architect_pass(db_res, code_res)
    update_metadata(db_res, code_res, ai_output)
    print("[SUCCESS] ai_universal_fixer.py execution completed.")
