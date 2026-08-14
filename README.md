# Quant-Research-Fundamental-MultiFactor

China A-share quant research project - **fundamental multi-factor monthly rotation strategy**.

Within the CSI 300 universe, five fundamental factors (value EP/BP, quality ROE, small size, momentum) are combined with equal weights into a composite score. The portfolio is rebalanced at each month end, holding the 30 highest-scoring stocks, and evaluated against the CSI 300 benchmark.

## Strategy Logic

1. **Universe**: CSI 300 constituents (~300 stocks)
2. **Factor construction**:
   | Factor | Definition | Direction |
   |------|------|------|
   | EP | 1 / PE(TTM); loss-making stocks excluded | higher is better |
   | BP | 1 / PB; negative (below book value) excluded | higher is better |
   | ROE | ROE(TTM) reconstructed from cumulative weighted ROE, point-in-time aligned via disclosure lag | higher is better |
   | Size | -ln(total market cap) | smaller is better |
   | Momentum | 12-1 month momentum (252-day return excluding the most recent 21 days) | higher is better |
3. **Factor preprocessing**: cross-sectional MAD winsorization + zscore standardization at each month end
4. **Combination**: equal-weight aggregation of the five factors (weights configurable in `config/config.yaml`)
5. **Rebalancing**: on the last trading day of each month, buy the top 30 stocks by score and hold them equal-weighted
6. **Costs**: 0.03% commission + 0.1% stamp tax (sell side) + 0.1% slippage, charged proportionally to turnover

**ROE point-in-time alignment**: TTM ROE is reconstructed as `current cumulative ROE + prior annual ROE - prior same-period cumulative ROE`, and takes effect after the statutory disclosure deadline (Q1 +30d, semi-annual +62d, Q3 +31d, annual +121d) to avoid look-ahead bias.

## Data Sources

- Valuation data (daily PE/PB/market cap etc., available since 2018): akshare `stock_value_em` (Eastmoney valuation analysis)
- Financial indicators (quarterly weighted ROE): akshare `stock_financial_analysis_indicator_em` (Eastmoney financial analysis)
- Benchmark index: akshare `stock_zh_index_daily` (CSI 300)
- Returns are constructed by compounding official daily percentage changes, which is automatically ex-dividend/split adjusted

## Backtest Results (2020-01 ~ 2026-07)

### Long Portfolio vs Benchmark

| Metric | Long Portfolio (Top 30) | CSI 300 |
|------|------|------|
| Cumulative return | **+206.72%** | +14.59% |
| Annualized return | **18.56%** | 2.09% |
| Annualized volatility | 15.36% | - |
| Sharpe ratio | 1.21 | - |
| Max drawdown | -10.83% | -39.92% |
| Calmar ratio | 1.71 | - |
| Avg monthly turnover | 12.91% | - |

![NAV curve](results/plots/nav_curve.png)

### Quantile Backtest (5 groups, group 1 has the highest scores)

| Group | Annualized return | Sharpe | Max drawdown |
|------|------|------|------|
| group_1 (top) | 23.33% | 1.50 | -11.43% |
| group_2 | 23.13% | 1.16 | -16.84% |
| group_3 | 20.08% | 1.07 | -17.72% |
| group_4 | 15.73% | 0.73 | -20.89% |
| group_5 (bottom) | 10.47% | 0.45 | -25.86% |

Group returns decrease monotonically from top to bottom, indicating good factor discriminating power.

![Quantile NAV](results/plots/quantile_nav.png)

### Factor Rank IC

| Factor | IC mean | ICIR | IC positive ratio |
|------|------|------|------|
| size | +0.0540 | +0.301 | 60.3% |
| bp | +0.0292 | +0.102 | 56.4% |
| ep | +0.0234 | +0.083 | 47.4% |
| momentum | +0.0123 | +0.050 | 53.8% |
| roe | -0.0020 | -0.014 | 52.6% |

![Cumulative IC](results/plots/factor_ic_cumsum.png)

## Factor Weight Optimization

In addition to the default equal-weight combination, the project implements two generations of nine weight optimization methods (all walk-forward, strictly using only point-in-time information):

**v1 baseline methods**

| Method | Idea |
|------|------|
| Dynamic ICIR weighting | Weight proportional to rolling 12-month ICIR (negatives clipped to zero), automatically down-weighting decayed factors |
| Portfolio ICIR maximization | Constrained optimization (scipy SLSQP) on factor IC mean/covariance matrix (with shrinkage) |
| Ridge walk-forward | Rolling 36-month ridge regression directly predicts next-period returns; coefficients act as adaptive weights |
| GBDT walk-forward | Histogram gradient boosting (sklearn HistGradientBoosting) to capture nonlinear interactions |

**v2 improved methods**

| Method | Idea |
|------|------|
| EWMA-IC weighting | Exponentially decayed (6-month half-life) ICIR weighting; more weight on recent signals |
| Orthogonalization + EWMA | Sequentially regress out preceding factors in the order size -> value -> quality -> momentum, then weight the residuals |
| Mean-variance weighting | True monthly returns of factor long-short portfolios + Ledoit-Wolf shrunk covariance, maximizing Sharpe |
| ML ensemble (shrinkage) | Expanding-window Ridge+GBDT ensemble; scores shrunk 30% toward the equal-weight score after cross-sectional standardization to reduce variance |
| Inverse-volatility weighting | Within the Top-N, weight by the inverse of 63-day realized volatility (composable with any scoring method) |

```bash
python scripts/run_weight_optimization.py
```

### Comparison (2020-01 ~ 2026-07, monthly rebalance, Top 30)

| Method | Annualized return | Sharpe | Max drawdown | Annualized volatility | Monthly turnover |
|------|------|------|------|------|------|
| Equal weight (baseline) | 18.56% | 1.21 | **-10.83%** | **15.36%** | 12.9% |
| Dynamic ICIR weighting | 3.85% | 0.20 | -29.50% | 19.71% | 25.8% |
| Portfolio ICIR maximization | 8.43% | 0.49 | -26.48% | 17.28% | 14.2% |
| Ridge walk-forward | 19.89% | 0.81 | -28.72% | 24.52% | 16.1% |
| GBDT walk-forward | 21.04% | 0.82 | -27.77% | 25.76% | 28.7% |
| EWMA-IC weighting | 1.24% | 0.06 | -27.92% | 19.70% | 29.1% |
| Orthogonalization + EWMA | 13.78% | 0.66 | -29.72% | 20.78% | 29.4% |
| Mean-variance weighting | 9.66% | 0.50 | -30.32% | 19.21% | 14.5% |
| **ML ensemble (shrinkage)** | **27.64%** | **1.21** | -19.50% | 22.93% | 19.8% |
| Equal weight + inverse volatility | 11.46% | 0.82 | **-9.54%** | 14.04% | 12.9% |
| ML ensemble + inverse volatility | 23.32% | 1.21 | -19.01% | 19.34% | 19.8% |

Out-of-sample (after the training warm-up) for the ML ensemble: annualized 41.99%, Sharpe 1.56; adding inverse-volatility weighting reduces volatility to 19.3%.

![Weight optimization comparison](results/plots/weight_optimization_nav.png)

**Conclusions**:
- **The ML ensemble (shrinkage) is the best risk-adjusted option**: it matches the equal-weight Sharpe (1.21) over the full sample while delivering ~9 percentage points higher returns,
  and its out-of-sample Sharpe of 1.56 leads across the board. The key improvements are: expanding-window training (avoiding rolling-window history forgetting),
  target winsorization, the Ridge/GBDT ensemble, and 30% shrinkage toward the equal-weight score for variance reduction
- Inverse-volatility weighting significantly reduces volatility (ML ensemble: 22.9% -> 19.3%, drawdown -19.5% -> -19.0%),
  but dilutes the weights of high-beta small-cap names, lowering absolute returns
- Short-window signal methods (ICIR/EWMA/orthogonalization/mean-variance) all clearly underperform equal weight in this sample, suggesting monthly IC is too noisy;
  dynamic factor-level weight timing is less effective than stock-level nonlinear modeling

## Project Structure

```
Quant-Research-Fundamental-MultiFactor/
├── config/config.yaml          # Global configuration (universe / factor weights / backtest parameters)
├── data/
│   ├── raw/                    # Raw data (parquet, gitignored)
│   └── processed/              # Processed data
├── notebooks/                  # Research notebooks
├── results/
│   ├── reports/                # Backtest reports and metric summaries
│   └── plots/                  # Charts
├── scripts/
│   ├── fetch_data.py           # Data download
│   ├── run_backtest.py         # One-click backtest
│   ├── run_weight_optimization.py  # Weight optimization comparison
│   └── show_holdings.py        # Latest holdings based on the most recent data
├── src/
│   ├── data/data_loader.py     # Data fetching and preprocessing
│   ├── factors/base.py         # Winsorization / standardization
│   ├── factors/fundamental.py  # EP/BP/ROE/Size/Momentum five factors
│   ├── factors/weight_optimizer.py  # Weight optimization (ICIR / constrained optimization / ML walk-forward)
│   ├── backtest/engine.py      # Monthly rotation backtest engine
│   ├── backtest/analyzers.py   # IC analysis and report generation
│   └── utils/helpers.py        # Utility functions
├── tests/                      # Unit tests (factors + weight optimization)
├── requirements.txt
└── setup.py
```

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# 1. Download data (CSI 300 valuation + financials + benchmark index, ~5 minutes)
python scripts/fetch_data.py

# 2. Run the backtest (generates reports, plots and metrics)
python scripts/run_backtest.py

# 3. Compare weight optimization methods (ICIR weighting / constrained optimization / ML walk-forward)
python scripts/run_weight_optimization.py

# 4. Output the current holdings (ML ensemble scoring by default; add --equal-weight to switch)
python scripts/show_holdings.py

# 5. Run unit tests
pytest tests/ -v
```

## Caveats

- Constituents use the current snapshot list, introducing survivorship bias; historical backtest results are optimistic
- Financial disclosure lag uses statutory deadlines as an approximation, not actual disclosure dates
- NAV is sampled at month ends; intra-month drawdowns may be understated
- This project is for quant research purposes only and does not constitute any investment advice
