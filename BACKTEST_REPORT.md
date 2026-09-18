# MLB Engine Empirical Backtest Report (2026-09-18 08:20:19)

### 📊 Walk-Forward Simulation Summary (50,000 Iterations)

| Metric | Result | Target Benchmark |
| :--- | :---: | :---: |
| **Sample Size (Games Evaluated)** | `2000` | > 2,000 |
| **Full Game Win Accuracy** | `53.15%` | > 54.0% |
| **First 5 (F5) Win Accuracy** | `53.74%` | > 55.0% |
| **Probability Brier Score** | `0.2482` | < 0.2500 |
| **Full Game Mean Run Error** | `3.72 runs` | < 3.80 |
| **First 5 (F5) Median Run Error** | `2.58 runs` | <= 2.50 |
| **Batter Hit Prop MAE** | `0.70 hits` | < 0.70 |
| **Batter Over 0.5 Hit Brier** | `0.2509` | < 0.2250 |

### ⚙️ Engine State
- **Execution Model**: Deterministic Dual-Engine Monte Carlo (50,000 Iterations).
- **Ensemble**: Stacking Classifier (RandomForest + XGBoost -> Logistic Regression).
- **Lookahead Isolation**: Strict point-in-time progression (zero data leakage).
- **F5 Scoring**: Evaluated against continuous median with 0.5000 volume scalar.
- **Player Hit Props**: Endogenous plate appearance simulation with Log5 batter-vs-pitcher contact mixture.
