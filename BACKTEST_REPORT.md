# MLB Engine Empirical Backtest Report (2026-09-18 16:43:02)

### 📊 Walk-Forward Simulation Summary (50,000 Iterations)

| Metric | Result | Target Benchmark |
| :--- | :---: | :---: |
| **Sample Size (Games Evaluated)** | `2000` | > 2,000 |
| **Full Game Win Accuracy** | `53.65%` | > 54.0% |
| **First 5 (F5) Win Accuracy** | `53.50%` | > 55.0% |
| **Probability Brier Score** | `0.2481` | < 0.2500 |
| **Full Game Mean Run Error** | `3.74 runs` | < 3.80 |
| **First 5 (F5) Median Run Error** | `2.57 runs` | <= 2.50 |
| **Batter Hit Prop MAE** | `0.70 hits` | < 0.70 |
| **Batter Over 0.5 Hit Brier** | `0.2515` | < 0.2250 |

### ⚙️ Engine State
- **Execution Model**: Deterministic Dual-Engine Monte Carlo (50,000 Iterations).
- **Ensemble**: Stacking Classifier (RandomForest + XGBoost -> Logistic Regression).
- **Lookahead Isolation**: Strict point-in-time progression (zero data leakage).
- **F5 Scoring**: Evaluated against continuous median with 5.0 / 9.7 volume scalar.
- **Player Hit Calibration**: Empirical Bayes log-odds shrinkage toward 60.5% starter hit rate.
