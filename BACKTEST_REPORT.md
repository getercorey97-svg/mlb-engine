# MLB Engine Empirical Backtest Report (2026-09-18 04:33:11)

### 📊 Walk-Forward Simulation Summary

| Metric | Result | Target Benchmark |
| :--- | :---: | :---: |
| **Sample Size (Games Evaluated)** | `2000` | > 2,000 |
| **Full Game Win Accuracy** | `52.90%` | > 54.0% |
| **First 5 (F5) Win Accuracy** | `54.21%` | > 55.0% |
| **Brier Score Calibration** | `0.2482` | < 0.2500 |
| **Full Game Mean Run Error** | `3.79 runs` | < 3.80 |
| **First 5 (F5) Median Run Error** | `2.55 runs` | <= 2.50 |

### ⚙️ Engine State
- **Execution Model**: Deterministic Dual-Engine Monte Carlo (Dynamic Starter Length + L1 Median F5 NegBinomial).
- **Ensemble**: Stacking Classifier (RandomForest + XGBoost -> Logistic Regression).
- **Lookahead Isolation**: Strict point-in-time progression (no data leakage).
- **F5 Calibration**: Decoupled from bullpen noise, scaled with 0.90x TTOP suppression and 1.03x top-order PA weighting.
