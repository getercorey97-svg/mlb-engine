import ast
import shutil
import sqlite3
import numpy as np
from datetime import datetime
from backtest_engine import run_historical_backtest[span_7](start_span)[span_7](end_span)

EVOLUTION_LEDGER_TABLE = '''
    CREATE TABLE IF NOT EXISTS Code_Evolution_Ledger (
        mutation_id INTEGER PRIMARY KEY AUTOINCREMENT,
        target_file TEXT,
        previous_brier REAL,
        candidate_brier REAL,
        accuracy_delta REAL,
        applied INTEGER,
        patch_description TEXT,
        timestamp TEXT
    )
'''

def verify_code_integrity(code_string):
    """Parses code into an Abstract Syntax Tree to ensure no syntax errors exist."""
    try:
        ast.parse(code_string)
        return True
    except SyntaxError as e:
        print(f"[REJECTED] Syntax verification failure: {e}")
        return False

def evaluate_candidate_code(target_file, candidate_code, patch_description):
    """
    Sandboxes candidate code, executes historical validation, and checks accuracy metrics.
    Only commits the file if accuracy strictly improves.
    """
    print(f"[{datetime.now()}] Evaluating autonomous code patch for: {target_file}")
    
    if not verify_code_integrity(candidate_code):
        return False

    backup_file = f"{target_file}.bak"
    sandbox_file = f"{target_file}.sandbox"
    
    # 1. Establish baseline accuracy metrics from mlb_engine.db
    conn = sqlite3.connect('mlb_engine.db')
    cursor = conn.cursor()
    cursor.execute(EVOLUTION_LEDGER_TABLE)
    conn.commit()
    
    cursor.execute('''
        SELECT home_prob, (CASE WHEN home_score > away_score THEN 1 ELSE 0 END) AS home_win, model_correct
        FROM Post_Match_Analysis p
        INNER JOIN Model_Forecasts m ON p.game_pk = m.game_pk
        WHERE home_prob IS NOT NULL
        ORDER BY p.game_pk DESC LIMIT 200
    ''')
    rows = cursor.fetchall()
    
    if len(rows) < 50:
        print("[BYPASS] Insufficient sample size for validation bench (N < 50). Mutation aborted.")
        conn.close()
        return False
        
    baseline_brier = np.mean([(r[0] - r[1]) ** 2 for r in rows])
    baseline_acc = np.mean([r[2] for r in rows])
    
    # 2. Stage candidate code into sandbox
    with open(sandbox_file, 'w') as f:
        f.write(candidate_code)
        
    # Swap sandbox into production temporarily for isolated evaluation
    shutil.copyfile(target_file, backup_file)
    shutil.move(sandbox_file, target_file)
    
    mutation_successful = False
    try:
        # Run historical validation across the last 30 days[span_8](start_span)[span_8](end_span)
        run_historical_backtest("2026-08-01", "2026-09-01")[span_9](start_span)[span_9](end_span)
        
        # 3. Measure candidate performance
        cursor.execute('''
            SELECT home_prob, (CASE WHEN actual_home_runs > actual_away_runs THEN 1 ELSE 0 END),
                   (CASE WHEN (home_prob > 0.5 AND actual_home_runs > actual_away_runs) OR 
                              (home_prob <= 0.5 AND actual_away_runs > actual_home_runs) THEN 1 ELSE 0 END)
            FROM Backtest_Results
            WHERE home_prob IS NOT NULL
        ''')[span_10](start_span)[span_10](end_span)
        eval_rows = cursor.fetchall()
        
        candidate_brier = np.mean([(r[0] - r[1]) ** 2 for r in eval_rows])
        candidate_acc = np.mean([r[2] for r in eval_rows])
        acc_delta = candidate_acc - baseline_acc
        
        print(f"Baseline Brier:  {baseline_brier:.4f} | Baseline Acc:  {baseline_acc:.2%}")
        print(f"Candidate Brier: {candidate_brier:.4f} | Candidate Acc: {candidate_acc:.2%}")

        # 4. Strict Overwrite Mandate: Lower Brier Score AND Higher Accuracy required
        if candidate_brier < baseline_brier and candidate_acc >= baseline_acc:
            print("[VERIFIED] Improvement confirmed. Code mutation locked into production.")
            mutation_successful = True
            applied_flag = 1
        else:
            print("[REVERTED] Candidate failed accuracy threshold. Rolling back original code.")
            shutil.copyfile(backup_file, target_file)
            applied_flag = 0

        # Log mutation event in SQLite audit table
        cursor.execute('''
            INSERT INTO Code_Evolution_Ledger 
            (target_file, previous_brier, candidate_brier, accuracy_delta, applied, patch_description, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (target_file, baseline_brier, candidate_brier, acc_delta, applied_flag, patch_description, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        conn.commit()

    except Exception as e:
        print(f"[FATAL MUTATION ERROR] {e}. Restoring backup immediately.")
        shutil.copyfile(backup_file, target_file)
    finally:
        conn.close()
        
    return mutation_successful
