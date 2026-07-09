# Equity Volatility Forecasting: ML vs. Econometric Models

**Author:** Anthony Le (Denison University, Class of 2027)  
**Supervisors:** Dr. Emily Marshall & Dr. Tyler Wake (Denison Economics)  
**Timeline:** 10-week summer research, May–August 2026

---

## Research Questions

1. **Statistical Accuracy** — Do ML and hybrid models significantly reduce out-of-sample forecast errors (RMSE, QLIKE) vs. parametric benchmarks?
2. **Economic Utility** — Does superior statistical accuracy translate into higher risk-adjusted returns in portfolio applications?
3. **Regime Resilience** — How do models perform during market stress vs. calm periods?

---

## Models Compared

| Category | Models |
|---|---|
| Econometric | GARCH(1,1), EGARCH(1,1), GJR-GARCH(1,1), HAR-RV |
| Machine Learning | Random Forest, XGBoost, LSTM (PyTorch) |
| Hybrid | GARCH-RF, GARCH-XGBoost |

All models are evaluated on 18 equity assets (SPY, QQQ, IWM + 15 stocks) using walk-forward validation: trained on 2010–2017, tested on 2018–2025.

---

## Repository Structure

```
Volatility Research/
├── notebooks/
│   ├── 01_EDA.ipynb              # Exploratory analysis, regime visualization
│   ├── 02_garch_baseline.ipynb   # GARCH family + HAR-RV walk-forward, residual diagnostics
│   ├── 03_ml_models.ipynb        # RF/XGBoost/LSTM/hybrid models, MCS + Diebold-Mariano
│   ├── 04_results.ipynb          # Economic evaluation (Week 8) + robustness (Week 9)
│   └── 05_manuscript.ipynb       # Full manuscript write-up (Week 10)
├── src/
│   ├── data.py                   # Download (yfinance), clean, compute RV, tag regimes
│   ├── features.py               # Engineer 40–50 predictors, correlation filter
│   ├── models.py                 # Model wrappers: fit() / forecast() interface
│   ├── evaluation.py             # RMSE, MAE, QLIKE, directional accuracy, DM test, MCS
│   └── backtest.py               # 4 portfolio strategies + performance metrics
├── data/
│   ├── raw/                      # {TICKER}_daily.csv — regenerate via src/data.py
│   └── processed/                # prices_clean.csv, realized_vol.csv, features.csv 
├── results/
│   ├── forecast_accuracy_econometric.csv
│   ├── forecast_accuracy_all_models.csv
│   ├── model_confidence_set.csv / diebold_mariano_vs_garch.csv
│   ├── portfolio_performance.csv
│   ├── feature_importance.csv
│   └── figures/                  # Publication-quality plots
└── requirements.txt
```

---

## Setup

**Requirements:** Python 3.14, macOS/Linux

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Set up API key (Alpha Vantage backup — yfinance is primary)
echo "ALPHA_VANTAGE_KEY=your_key_here" > .env
```

> **Notes:**
> - TensorFlow does not support Python 3.14 — PyTorch is used for LSTM instead.
> - `pandas_datareader` has a Python 3.14 bug — VIX is fetched via `yfinance` (`^VIX` ticker).

---

## Reproducing Results

Run notebooks in order. Data and model artifacts are gitignored; re-download them first.

```bash
# 1. Download all raw data (18 assets, 2010–2025)
python -c "from src.data import download_prices; download_prices()"

# 2. Launch JupyterLab
jupyter lab
```

Then execute notebooks sequentially: `01 → 02 → 03 → 04 → 05`.

To run a notebook non-interactively:

```bash
jupyter nbconvert --to notebook --execute notebooks/01_EDA.ipynb
```

> **Runtime note:** most notebooks finish in minutes, but `03_ml_models.ipynb`'s LSTM
> walk-forward (96 refits × 18 assets) takes roughly **24 hours** of wall-clock time on a
> single machine (see `run_walk_forward.log`). All notebooks cache their forecast CSVs to
> `data/processed/forecasts_*.csv` and skip recomputation on a subsequent run if that file
> already exists — a fresh clone's first run is the slow one.

---

## Data

- **Target variable:** Realized Volatility (`rv_21d`) — 21-day rolling std of daily log returns, annualized to %.
- **Volatility regimes:** Low < 15%, Medium 15–25%, High > 25%.
- **Features:** ~40–50 predictors across lagged volatility, return dynamics, market microstructure, VIX/SPY market-wide signals, and calendar effects. All features use `.shift(1)` to prevent look-ahead bias.
- **Primary source:** [yfinance](https://github.com/ranaroussi/yfinance). Alpha Vantage is backup only (rate-limited).

---

## Evaluation

**Statistical metrics:** RMSE, MAE, QLIKE, Directional Accuracy  
**Economic metrics:** Sharpe Ratio, Sortino Ratio, Max Drawdown, Calmar Ratio

**Portfolio strategies:**
1. **Volatility Timing** — Dynamic SPY/SHY allocation targeting 15% annualized vol
2. **Risk Parity** — Inverse-vol weights across 15 stocks, monthly rebalance
3. **Regime-Based** — Discrete 100/60/20% equity exposure by vol regime
4. **VRP** — Variance-risk-premium strategy exploiting the implied-vs-forecast vol spread

All strategies apply $5/trade transaction cost drag (reported as its own `Ann. Cost Drag`
metric) and rebalance monthly. Statistical significance across all 9 models is assessed via
Diebold-Mariano tests and a 90%-confidence Model Confidence Set.

---

## Key References

- Engle (1982); Bollerslev (1986); Nelson (1991) — ARCH/GARCH/EGARCH foundations
- Corsi (2009) — HAR-RV model
- Hansen & Lunde (2005) — "Does Anything Beat GARCH(1,1)?"
- Gu, Kelly & Xiu (2020) — ML for asset pricing
- Moreira & Muir (2017) — Volatility-managed portfolios
- Fleming, Kirby & Ostdiek (2001, 2003) — Economic value of volatility timing

---

## Current Progress

| Notebook | Status |
|---|---|
| 01_EDA | Complete |
| 02_garch_baseline (econometric walk-forward + residual diagnostics) | Complete |
| 03_ml_models (RF, XGBoost, LSTM, hybrids, MCS/Diebold-Mariano) | Complete |
| 04_results (economic evaluation + Week 9 robustness) | Complete |
| 05_manuscript (full write-up) | Complete |

Weeks 1–10 of the original 10-week research plan are complete; see `results/` for every
generated table and figure.

---

## Development Notes

Research design, modeling decisions, and all analysis are the author's. An AI coding
assistant (Claude Code) was used during development to help implement and debug portions
of the codebase under the author's direction and review.
