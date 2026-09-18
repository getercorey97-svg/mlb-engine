import sqlite3
import numpy as np
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

BRIER_DRIFT_THRESHOLD = 0.2520
BRIER_RETRAIN_THRESHOLD = 0.2540
F5_MAE_DRIFT_THRESHOLD = 2.50
F5_BIAS_DRIFT_THRESHOLD = 0.30

def ensure_telemetry_schemas(cursor):
    """Guarantees persistence schemas for system telemetry and rolling error auditing."""
    cursor.executescript('''
    CREATE TABLE IF NOT EXISTS System_Telemetry (
        telemetry_id INTEGER PRIMARY KEY AUTOINCREMENT,
        evaluated_games INTEGER,
        rolling_brier REAL,
        rolling_win_acc REAL,
        rolling_f5_acc REAL,
        rolling_full_run_mae REAL,
        rolling_f5_median_mae REAL,
        rolling_f5_signed_bias REAL,
        rolling_full_signed_bias REAL,
        drift_flag INTEGER DEFAULT 0,
        retrain_recommended INTEGER DEFAULT 0,
        logged_at TEXT
    );
    CREATE TABLE IF NOT EXISTS Calibration_Offsets (
        market_type TEXT PRIMARY KEY,
        active_bias_offset REAL DEFAULT 0.00,
        consecutive_drifts INTEGER DEFAULT 0,
        last_adjusted TEXT
    );
    ''')

    cursor.execute("SELECT COUNT(*) FROM Calibration_Offsets WHERE market_type = 'F5_TOTAL';")
    if cursor.fetchone()[0] == 0:
        now_ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cursor.execute("INSERT OR REPLACE INTO Calibration_Offsets (market_type, active_bias_offset, consecutive_drifts, last_adjusted) VALUES ('F5_TOTAL', 0.00, 0, ?);", (now_ts,))
        cursor.execute("INSERT OR REPLACE INTO Calibration_Offsets (market_type, active_bias_offset, consecutive_drifts, last_adjusted) VALUES ('FULL_TOTAL', 0.00, 0, ?);", (now_ts,))

def audit_rolling_telemetry(conn, cursor, window_size=50):
    """
    Computes rolling 50-game statistical calibration metrics:
    - Brier score calibration
    - Win accuracy (Full game & F5)
    - Full-game run MAE and signed bias
    - First 5 (F5) discrete median MAE and directional signed bias
    """
    ensure_telemetry_schemas(cursor)
    conn.commit()

    query = '''
    SELECT 
        p.game_pk,
        p.home_score,
        p.away_score,
        COALESCE(p.home_f5_score, 0) as home_f5,
        COALESCE(p.away_f5_score, 0) as away_f5,
        m.home_prob,
        m.predicted_home_runs,
        m.predicted_away_runs,
        COALESCE(f.f5_home_prob, 0.50),
        COALESCE(f.f5_away_prob, 0.50),
        COALESCE(f.f5_median_total, f.f5_total_runs, 4.50) as f5_median_proj
    FROM Post_Match_Analysis p
    INNER JOIN Model_Forecasts m ON p.game_pk = m.game_pk
    LEFT JOIN F5_Forecasts f ON p.game_pk = f.game_pk
    WHERE p.home_score IS NOT NULL 
      AND p.away_score IS NOT NULL 
      AND p.actual_winner IS NOT NULL
    ORDER BY p.game_pk DESC
    LIMIT ?
    '''
    cursor.execute(query, (window_size,))
    rows = cursor.fetchall()

    if len(rows) < 15:
        print(f"[TELEMETRY] Insufficient recent sample for rolling telemetry (N = {len(rows)} < 15). Skipping drift audit.")
        return 0.00, False, False

    brier_scores = []
    full_wins_correct = []
    f5_wins_correct = []
    f5_valid_games = 0
    full_run_errors = []
    full_signed_errors = []
    f5_run_errors = []
    f5_signed_errors = []

    for r in rows:
        pk, h_score, a_score, h_f5, a_f5, h_prob, pred_h, pred_a, f5_h_prob, f5_a_prob, f5_med = r
        
        act_home_win = 1.0 if h_score > a_score else 0.0
        brier_scores.append((h_prob - act_home_win) ** 2)
        
        full_pick = 1.0 if h_prob >= 0.50 else 0.0
        full_wins_correct.append(1 if full_pick == act_home_win else 0)

        act_total = h_score + a_score
        pred_total = pred_h + pred_a
        full_run_errors.append(abs(act_total - pred_total))
        full_signed_errors.append(act_total - pred_total)

        act_f5_total = h_f5 + a_f5
        f5_med_rounded = round(f5_med)
        f5_run_errors.append(abs(act_f5_total - f5_med_rounded))
        f5_signed_errors.append(act_f5_total - f5_med_rounded)

        if h_f5 != a_f5:
            act_f5_winner = 1.0 if h_f5 > a_f5 else 0.0
            f5_pick = 1.0 if f5_h_prob > f5_a_prob else 0.0
            f5_wins_correct.append(1 if f5_pick == act_f5_winner else 0)
            f5_valid_games += 1

    rolling_brier = round(float(np.mean(brier_scores)), 4)
    rolling_win_acc = round(float(np.mean(full_wins_correct)), 4)
    rolling_f5_acc = round(float(np.mean(f5_wins_correct)), 4) if f5_valid_games > 0 else 0.00
    rolling_full_mae = round(float(np.mean(full_run_errors)), 2)
    rolling_full_bias = round(float(np.mean(full_signed_errors)), 2)
    rolling_f5_mae = round(float(np.mean(f5_run_errors)), 2)
    rolling_f5_bias = round(float(np.mean(f5_signed_errors)), 2)

    drift_flag = 1 if (rolling_brier > BRIER_DRIFT_THRESHOLD or 
                       rolling_f5_mae > F5_MAE_DRIFT_THRESHOLD or 
                       abs(rolling_f5_bias) >= F5_BIAS_DRIFT_THRESHOLD) else 0
                       
    retrain_recommended = 1 if (rolling_brier >= BRIER_RETRAIN_THRESHOLD or drift_flag == 1) else 0
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    cursor.execute('''
    INSERT INTO System_Telemetry 
    (evaluated_games, rolling_brier, rolling_win_acc, rolling_f5_acc, rolling_full_run_mae, 
     rolling_f5_median_mae, rolling_f5_signed_bias, rolling_full_signed_bias, drift_flag, retrain_recommended, logged_at)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (len(rows), rolling_brier, rolling_win_acc, rolling_f5_acc, rolling_full_mae, 
          rolling_f5_mae, rolling_f5_bias, rolling_full_bias, drift_flag, retrain_recommended, now_str))

    cursor.execute("SELECT consecutive_drifts FROM Calibration_Offsets WHERE market_type = 'F5_TOTAL';")
    prev_drifts = cursor.fetchone()[0]

    if drift_flag == 1:
        new_drifts = prev_drifts + 1
        # Dynamic L1 feedback controller: nudge expectancy by 35% of observed signed bias
        bias_adjustment = float(np.clip(rolling_f5_bias * 0.35, -0.40, 0.40))
        cursor.execute('''
        UPDATE Calibration_Offsets 
        SET active_bias_offset = ?, consecutive_drifts = ?, last_adjusted = ? 
        WHERE market_type = 'F5_TOTAL';
        ''', (bias_adjustment, new_drifts, now_str))
    else:
        new_drifts = max(0, prev_drifts - 1)
        cursor.execute('''
        UPDATE Calibration_Offsets 
        SET active_bias_offset = 0.00, consecutive_drifts = ?, last_adjusted = ? 
        WHERE market_type = 'F5_TOTAL';
        ''', (new_drifts, now_str))

    conn.commit()

    print("=" * 65)
    print(f"[{now_str}] SYSTEM TELEMETRY & AUTO-CALIBRATION AUDIT (N = {len(rows)})")
    print(f"• Rolling Brier Score     : {rolling_brier:.4f} (Threshold: < {BRIER_DRIFT_THRESHOLD})")
    print(f"• Full Game Outright Acc  : {rolling_win_acc:.1%}")
    print(f"• First 5 (F5) Win Acc    : {rolling_f5_acc:.1%}")
    print(f"• Full Game Run Delta     : {rolling_full_mae:.2f} runs (Signed Bias: {rolling_full_bias:+.2f})")
    print(f"• First 5 (F5) Median MAE : {rolling_f5_mae:.2f} runs (Signed Bias: {rolling_f5_bias:+.2f})")
    if drift_flag == 1:
        print(f"🚨 [DRIFT DETECTED] Auto-calibration offset injected: {bias_adjustment:+.3f} runs (Drift Count: {new_drifts})")
    else:
        print("✅ [CALIBRATION STABLE] Performance within operational confidence boundaries.")
    print("=" * 65)

    return rolling_f5_bias, bool(drift_flag), bool(retrain_recommended)

def get_active_f5_bias_offset(cursor) -> float:
    """Retrieves active L1 median bias correction offset for simulation engines."""
    try:
        cursor.execute("SELECT active_bias_offset FROM Calibration_Offsets WHERE market_type = 'F5_TOTAL';")
        row = cursor.fetchone()
        return float(row[0]) if row else 0.00
    except Exception:
        return 0.00

def compute_adaptive_alpha(base_alpha: float, err_history: list) -> float:
    """
    Momentum and oscillation elasticity for EWMA updates:
    - Scales alpha up by 1.4x if recent errors have consistent directional sign.
    - Dampens alpha down by 0.75x if errors oscillate across zero.
    """
    if len(err_history) < 3:
        return base_alpha
    
    last_three = err_history[-3:]
    all_positive = all(e > 0.35 for e in last_three)
    all_negative = all(e < -0.35 for e in last_three)
    
    if all_positive or all_negative:
        return float(min(0.20, base_alpha * 1.40))
    elif (last_three[-1] * last_three[-2]) < 0:
        return float(max(0.015, base_alpha * 0.75))
        
    return base_alpha

if __name__ == "__main__":
    conn = sqlite3.connect('mlb_engine.db', timeout=30)
    cursor = conn.cursor()
    audit_rolling_telemetry(conn, cursor)
    conn.close()
