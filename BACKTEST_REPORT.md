# MLB Engine Empirical Backtest Report (2026-09-18 04:14:50)

### 📊 Walk-Forward Simulation Summary

| Metric | Result | Target Benchmark |
| :--- | :---: | :---: |
| **Sample Size (Games Evaluated)** | `2293` | > 2,000 |
| **Full Game Win Accuracy** | `54.03%` | > 54.0% |
| **First 5 (F5) Win Accuracy** | `54.63%` | > 55.0% |
| **Brier Score Calibration** | `0.2472` | < 0.2500 |
| **Mean Absolute Run Error** | `3.74 runs` | < 3.20 |

### ⚙️ Engine State
- **Execution Model**: Deterministic Dual-Engine Monte Carlo (Dynamic Starter Length + NegBinomial).
- **Ensemble**: Stacking Classifier (RandomForest + XGBoost -> Logistic Regression).
- **Lookahead Isolation**: Strict point-in-time progression (no data leakage).
