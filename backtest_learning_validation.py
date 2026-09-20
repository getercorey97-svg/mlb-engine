import sqlite3
import numpy as np

def run_calibration_benchmark():
    conn = sqlite3.connect("mlb_engine.db")
    conn.row_factory = sqlite3.Row
    c = conn.cursor()

    tables = [r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()]
    if "Prop_Learning_Calibration_Audit" not in tables:
        print("[CALIBRATION] Prop_Learning_Calibration_Audit table not found. Run post_mortem_props.py first.")
        conn.close()
        return

    rows = c.execute("SELECT * FROM Prop_Learning_Calibration_Audit").fetchall()
    if not rows:
        print("[CALIBRATION] No audit records logged yet. Running validation over historical tables...")
        # Check Batter_Post_Mortem_Logs and Pitcher_Post_Mortem_Logs
        if "Batter_Post_Mortem_Logs" in tables:
            b_logs = c.execute("SELECT brier_score, hit_error FROM Batter_Post_Mortem_Logs").fetchall()
            if b_logs:
                b_briers = [r["brier_score"] for r in b_logs]
                b_maes = [abs(r["hit_error"]) for r in b_logs]
                print(f"--- BATTER POST-MORTEM BENCHMARK (N = {len(b_logs)}) ---")
                print(f"Mean Absolute Hit Error (MAE) : {np.mean(b_maes):.3f} hits")
                print(f"Over 0.5 Hit Brier Score      : {np.mean(b_briers):.4f}")

        if "Pitcher_Post_Mortem_Logs" in tables:
            p_logs = c.execute("SELECT brier_score, k_error FROM Pitcher_Post_Mortem_Logs").fetchall()
            if p_logs:
                p_briers = [r["brier_score"] for r in p_logs]
                p_maes = [abs(r["k_error"]) for r in p_logs]
                print(f"\n--- PITCHER POST-MORTEM BENCHMARK (N = {len(p_logs)}) ---")
                print(f"Mean Absolute K Error (MAE)   : {np.mean(p_maes):.3f} strikeouts")
                print(f"Strikeout Over/Under Brier    : {np.mean(p_briers):.4f}")
        conn.close()
        return

    b_audits = [r for r in rows if r["market_type"] == "batter_hit"]
    p_audits = [r for r in rows if r["market_type"] == "pitcher_k"]

    print("=" * 65)
    print("STATE-OF-THE-ART LEARNING MECHANISM CALIBRATION REPORT")
    print("=" * 65)
    if b_audits:
        b_brier = np.mean([r["brier_score"] for r in b_audits])
        b_mae = np.mean([abs(r["error_delta"]) for r in b_audits])
        print(f"Batter Decoupled Bayes (N={len(b_audits)}):")
        print(f"  • Hit MAE   : {b_mae:.3f} hits/game")
        print(f"  • Hit Brier : {b_brier:.4f} (Benchmark: < 0.2350)")

    if p_audits:
        p_brier = np.mean([r["brier_score"] for r in p_audits])
        p_mae = np.mean([abs(r["error_delta"]) for r in p_audits])
        print(f"\nPitcher Kalman State-Space (N={len(p_audits)}):")
        print(f"  • K MAE     : {p_mae:.3f} strikeouts/game")
        print(f"  • K Brier   : {p_brier:.4f} (Benchmark: < 0.2200)")
    print("=" * 65)
    conn.close()

if __name__ == "__main__":
    run_calibration_benchmark()
