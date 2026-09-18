# MLB Engine Empirical Backtest Report (2026-09-18 05:01:42)

### 📊 Walk-Forward Simulation Summary

| Metric | Result | Target Benchmark |
| :--- | :---: | :---: |
| **Sample Size (Games Evaluated)** | `2000` | > 2,000 |
| **Full Game Win Accuracy** | `53.30%` | > 54.0% |
| **First 5 (F5) Win Accuracy** | `54.92%` | > 55.0% |
| **Brier Score Calibration** | `0.2481` | < 0.2500 |
| **Full Game Mean Run Error** | `3.74 runs` | < 3.80 |
| **First 5 (F5) Median Run Error** | `2.57 runs` | <= 2.50 |

### ⚙️ Engine State
- **Execution Model**: Deterministic Dual-Engine Monte Carlo (Dynamic Starter Length + L1 Median F5 NegBinomial).
- **Ensemble**: Stacking Classifier (RandomForest + XGBoost -> Logistic Regression).
- **Lookahead Isolation**: Strict point-in-time progression (no data leakage).
- **F5 Calibration**: Decoupled from bullpen noise, scaled with 0.5155 empirical run share, 0.90x TTOP suppression, and discrete L1 median scoring.
