# MLB Engine Empirical Backtest Report (2026-09-18 03:49:18)

### 📊 Walk-Forward Simulation Summary

| Metric | Result | Target Benchmark |
| :--- | :---: | :---: |
| **Sample Size (Games Evaluated)** | `2000` | > 2,000 |
| **Full Game Win Accuracy** | `53.20%` | > 54.0% |
| **First 5 (F5) Win Accuracy** | `53.44%` | > 55.0% |
| **Brier Score Calibration** | `0.2478` | < 0.2500 |
| **Mean Absolute Run Error** | `3.80 runs` | < 3.20 |

### ⚙️ Engine State
- **Execution Model**: Deterministic Dual-Engine Monte Carlo (Physical Bullpen + Negative Binomial).
- **Ensemble**: Stacking Classifier (RandomForest + XGBoost -> Logistic Regression).
- **Lookahead Isolation**: Strict point-in-time progression (no data leakage).
