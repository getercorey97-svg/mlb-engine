# MLB Engine Empirical Backtest Report (2026-09-19 01:30:57)

### 📊 Walk-Forward Simulation Summary (50,000 Iterations)

| Metric | Result | Target Benchmark |
| :--- | :---: | :---: |
| **Sample Size (Games Evaluated)** | `2000` | > 2,000 |
| **Full Game Win Accuracy** | `53.90%` | > 54.0% |
| **First 5 (F5) Win Accuracy** | `52.91%` | > 55.0% |
| **Probability Brier Score** | `0.2481` | < 0.2500 |
| **Full Game Mean Run Error** | `3.90 runs` | < 3.80 |
| **First 5 (F5) Median Run Error** | `2.57 runs` | <= 2.50 |
| **Batter Hit Prop MAE** | `0.70 hits` | < 0.70 |
| **Batter Over 0.5 Hit Brier** | `0.2463` | < 0.2250 |

### ⚙️ Enhanced Quantitative Engine State
- **Markov Analytical Chain**: 24-state base-out fundamental matrix inversion replacing blunt Poisson approximations.
- **Arsenal Matching**: Pitcher repertoire decomposition (Sinker/Cutter vs Four-Seam/Sweeper) mapped in Log5 space.
- **Tiered Bullpen Leverage**: Binary tracking of high-leverage availability (closer/setup workload) with dynamic run taxes.
- **Catcher Shadow-Zone Integration**: Starting catcher framing metrics blended with home plate umpire strike zone edges.
- **Expanded Empirical Memory**: 500-matchup boxscore hydration driving mature Bayesian shrinkage weights.
