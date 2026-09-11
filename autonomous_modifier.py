import ast
import shutil
import sqlite3
import numpy as np
from datetime import datetime

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
    );
'''

def verify_code_integrity(code_string):
    """Parses code through Abstract Syntax Tree (AST) to ensure zero syntax breaks."""
    try:
        ast.parse(code_string)
        return True
    except SyntaxError as e:
        print(f"[REJECTED] Syntax verification failure: {e}")
        return False

def evaluate_candidate_code(target_file, candidate_code, patch_description):
    """
    Evaluates candidate mutations against recent production linescores.
    Adopts mutations if and only if Brier score decreases and accuracy does not drop.
    """
    print(f"[{datetime.now()}] Evaluating autonomous code patch for: {target_file}")

    if not verify_code_integrity(candidate_code):
        return False

    backup_file = f"{target_file}.bak"
    sandbox_file = f"{target_file}.sandbox"

    conn = sqlite3.connect('mlb_engine.db', timeout=30)
    cursor = conn.cursor()
    cursor.execute("PRAGMA journal_mode=WAL;")
    cursor.execute("PRAGMA busy_timeout=10000;")
    cursor.execute(EVOLUTION_LEDGER_TABLE)
    conn.commit()

    # 1. Establish baseline from Post_Match_Analysis joined with Model_Forecasts
    cursor.execute('''
        SELECT m.home_prob, 
               (CASE WHEN p.home_score > p.away_score THEN 1.0 ELSE 0.0 END) AS actual_home_win, 
               p.model_correct
        FROM Post_Match_Analysis p
        INNER JOIN Model_Forecasts m ON p.game_pk = m.game_pk
        WHERE m.home_prob IS NOT NULL AND p.home_score IS NOT NULL
        ORDER BY p.game_pk DESC LIMIT 250
    ''')
    rows = cursor.fetchall()

    if len(rows) < 30:
        print("[BYPASS] Insufficient empirical linescores (N < 30). Mutation deferred.")
        conn.close()
        return False

    baseline_brier = float(np.mean([(r[0] - r[1]) ** 2 for r in rows]))
    baseline_acc = float(np.mean([r[2] for r in rows]))

    # 2. Stage candidate code into sandbox
    with open(sandbox_file, 'w') as f:
        f.write(candidate_code)

    shutil.copyfile(target_file, backup_file)
    shutil.move(sandbox_file, target_file)

    mutation_successful = False
    try:
        import importlib
        import backtest_multiyr
        importlib.reload(backtest_multiyr)

        # Run controlled backtest sweep
        backtest_multiyr.run_backtest_sweep(years_back=1)

        # 3. Measure candidate performance
        cursor.execute('''
            SELECT m.home_prob, 
                   (CASE WHEN p.home_score > p.away_score THEN 1.0 ELSE 0.0 END),
                   p.model_correct
            FROM Post_Match_Analysis p
            INNER JOIN Model_Forecasts m ON p.game_pk = m.game_pk
            WHERE m.home_prob IS NOT NULL AND p.home_score IS NOT NULL
            ORDER BY p.game_pk DESC LIMIT 250
        ''')
        eval_rows = cursor.fetchall()

        candidate_brier = float(np.mean([(r[0] - r[1]) ** 2 for r in eval_rows]))
        candidate_acc = float(np.mean([r[2] for r in eval_rows]))
        acc_delta = candidate_acc - baseline_acc

        print(f"Baseline  | Brier: {baseline_brier:.4f} | Acc: {baseline_acc:.2%}")
        print(f"Candidate | Brier: {candidate_brier:.4f} | Acc: {candidate_acc:.2%}")

        if candidate_brier < baseline_brier and candidate_acc >= baseline_acc:
            print("[VERIFIED] Brier score improved. Mutation permanently adopted.")
            mutation_successful = True
            applied_flag = 1
        else:
            print("[REVERTED] Candidate failed accuracy gate. Restoring original code.")
            shutil.copyfile(backup_file, target_file)
            applied_flag = 0

        cursor.execute('''
            INSERT INTO Code_Evolution_Ledger 
            (target_file, previous_brier, candidate_brier, accuracy_delta, applied, patch_description, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (target_file, baseline_brier, candidate_brier, acc_delta, applied_flag, patch_description, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))
        conn.commit()

    except Exception as e:
        print(f"[MUTATION ENGINE ERROR] {e}. Restoring backup.")
        shutil.copyfile(backup_file, target_file)
    finally:
        if shutil.os.path.exists(backup_file):
            shutil.os.remove(backup_file)
        conn.close()

    return mutation_successful
