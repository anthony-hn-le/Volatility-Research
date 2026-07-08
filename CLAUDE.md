# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

10-week summer 2026 research project comparing ML vs. econometric models for equity volatility forecasting, with economic evaluation via portfolio backtesting. Supervisors: Emily Marshall & Tyler Wake (Denison Economics). See `claude-handoff.md` for the full research specification and weekly plan.

## Environment Setup

This project has its own `.venv` scoped to this folder — separate from the shared one at the `Quant Projects/` root. Always activate this local one before running anything here.

```bash
# Activate the virtual environment (from inside Volatility Research/)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Python version:** 3.14 (note: TensorFlow unsupported — use PyTorch for LSTM; `pandas_datareader` has a 3.14 bug — fetch VIX via `yfinance` ticker `^VIX`).

**API key:** Alpha Vantage key is stored in `.env` as `ALPHA_VANTAGE_KEY`. Never hard-code it.

## Common Commands

```bash
# Launch JupyterLab for notebooks
jupyter lab

# Run a module directly (e.g., download all data)
python -c "from src.data import download_prices; download_prices()"

# Run a specific notebook as a script (for reproduction)
jupyter nbconvert --to notebook --execute notebooks/01_eda.ipynb
```

There is no test suite yet. Validate pipeline steps interactively in notebooks or via `python -c "..."`.

## Architecture

### Data Flow
```
yfinance / Alpha Vantage
    → src/data.py         (download, clean, compute RV, tag regimes)
    → data/raw/           (one CSV per asset: {TICKER}_daily.csv)
    → data/processed/     (prices_clean.csv, realized_vol.csv, features.csv)
    → src/features.py     (engineer 40–50 predictors, correlation filter)
    → src/models.py       (walk-forward fit + forecast)
    → src/evaluation.py   (RMSE, MAE, QLIKE, directional accuracy)
    → src/backtest.py     (3 portfolio strategies + performance_metrics)
    → results/            (forecast_accuracy.csv, portfolio_performance.csv, figures/)
```

### Module Responsibilities

**`src/data.py`** — Download OHLCV via yfinance, compute `rv_21d` (21-day rolling log-return std, annualized to %), tag `regime` (low <15 / medium 15–25 / high >25). Key constants: `ASSETS` (18 tickers), `TRAIN_END = '2017-12-31'`, `TEST_START = '2018-01-01'`.

**`src/features.py`** — Builds ~40–50 predictors grouped into: historical vol lags, return dynamics, microstructure (high-low range, volume ratio, gap frequency), calendar dummies, and market-wide features (VIX, SPY RV). The top-level `engineer_features(df, vix_df, spy_rv)` runs all groups, drops NaNs, and applies correlation filtering (|r| > 0.95 threshold). **All features use `.shift(1)` to prevent look-ahead.**

**`src/models.py`** — Thin wrappers with a consistent `fit(X, y)` / `forecast(X_new)` interface:
- `GARCHModel(vol, p, o, q)` — wraps `arch` library; `vol='Garch'` + `o=0` → GARCH, `o=1` → GJR-GARCH; `vol='EGARCH'` → EGARCH
- `HARRVModel` — OLS on daily/weekly/monthly RV lags (Corsi 2009)
- `RandomForestModel` — includes `StandardScaler` internally
- `XGBoostModel` — includes early stopping with 15% validation split
- `LSTMModel` — PyTorch, 2-layer LSTM + linear head, 30-day lookback, early stopping (patience=10)
- `GARCHHybridModel(ml_model_type='RF'|'XGB')` — appends GARCH fitted vol + standardized residuals as extra features before delegating to `RandomForestModel`/`XGBoostModel`

**`src/evaluation.py`** — `evaluate_forecasts(y_true, forecasts_dict)` returns a DataFrame of RMSE/MAE/QLIKE/DirAcc for all models at once. `regime_evaluation(...)` slices by low/medium/high regime. `diebold_mariano(loss_a, loss_b, h)` and `model_confidence_set(losses_df, size, reps)` (wraps `arch.bootstrap.MCS`) provide statistical-significance testing across models.

**`src/backtest.py`** — Four strategies, all returning a `cost_drag` column and expecting **log returns** as input (matching `src/data.py`'s convention):
1. `vol_timing_strategy` — SPY/SHY allocation targeting 15% vol (`w = target_vol / forecast_vol`, capped at 1.5×)
2. `risk_parity_strategy` — inverse-vol weights across all 15 stocks, monthly rebalance
3. `regime_strategy` — discrete 100/60/20% equity depending on regime
4. `vrp_strategy` — delta-neutral short-straddle proxy exploiting the IV-vs-forecast spread (VRP)

All strategies apply `$5/trade` transaction cost drag and rebalance monthly (`resample('ME')`). `performance_metrics(returns, rf_rate=0.04, cost_drag=None)` computes Sharpe, Sortino, max drawdown, and Calmar from the log-return series (`cum = exp(returns.cumsum())`), plus annualized `Ann. Cost Drag` when a `cost_drag` series is passed.

### Walk-Forward Validation

- **Training:** Jan 2010 – Dec 2017 (expanding window, not rolling)
- **Test:** Jan 2018 – Dec 2025 (~96 monthly refits)
- **Horizons:** 1-day, 5-day, 20-day ahead
- GARCH `forecast(horizon=n)` returns an array of length `n`; ML models forecast one step at a time

### Hybrid Models

GARCH-RF and GARCH-XGBoost feed GARCH fitted values + GARCH residuals as extra features into the ML models. Implement by concatenating GARCH outputs to the `X` matrix from `engineer_features` before calling `RandomForestModel.fit`.

## Notebooks Convention

Numbered with 2-digit prefixes in `notebooks/`: `01_eda.ipynb`, `02_garch_baseline.ipynb`, `03_ml_models.ipynb`, `04_results.ipynb`, `05_manuscript.ipynb`. Figures saved to `results/figures/`.

## Key Design Decisions

- **Realized volatility proxy:** 21-day rolling std of daily log returns, annualized to percent. Not intraday RV (unavailable at scale).
- **No `pandas_datareader` for VIX:** fetch `^VIX` via yfinance instead.
- **PyTorch, not TensorFlow:** Python 3.14 incompatibility.
- **Alpha Vantage is backup only:** free tier is rate-limited to 5 req/min; yfinance is primary.
- **Gitignored:** `data/raw/`, `data/processed/`, `models/*.pkl`, `models/*.pt`, `results/` — regenerate from scripts, never commit data.
