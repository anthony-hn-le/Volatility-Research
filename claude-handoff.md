# Claude Handoff: Volatility Forecasting Research

**Project:** Machine Learning and Traditional Econometrics for Volatility Forecasting  
**Duration:** 10-week summer research (May 2026)  
**Supervisor:** Emily Marshall & Tyler Wake (Denison Economics)  
**Goal:** Compare ML models vs. econometric benchmarks for equity volatility prediction + test economic value in portfolios  

---

## 📋 Current Status

**Phase:** Week 1 – Foundation & Setup  
**Date:** May 21, 2026  
**Progress:** Initial planning complete; 6 foundational tasks created

### Completed Work
- ✅ Research proposal drafted and reviewed
- ✅ Literature review completed (GARCH, LSTM, Risk-Parity literature)
- ✅ Work plan and timeline established
- ✅ Initial architectural decisions made
- ✅ Task list created for Phase 1

### Active Tasks (Use TaskUpdate to track)
1. Set up Python environment and dependencies
2. Test API connections and data availability
3. Design data collection and storage strategy
4. Exploratory Data Analysis (EDA) and baseline calculations
5. Implement baseline GARCH(1,1) model
6. Create project documentation and README

---

## 🎯 Research Questions

1. **Statistical Accuracy:** Do ML and hybrid models significantly reduce out-of-sample forecast errors (RMSE, QLIKE) vs. parametric benchmarks?
2. **Economic Utility:** Does superior statistical accuracy translate into higher risk-adjusted returns in portfolio applications?
3. **Regime Resilience:** How do models perform during market stress vs. calm periods?

---

## 📊 Data Specification

### Assets (18 Total)
**Indices (3):**
- SPY (S&P 500)
- QQQ (Nasdaq-100)
- IWM (Russell 2000)

**Individual Stocks (15):**
- Tech: AAPL, MSFT, NVDA
- Finance: JPM, GS, BAC
- Healthcare: JNJ, UNH, PFE
- Consumer: AMZN, WMT, HD
- Energy/Industrials: XOM, CVX, CAT

### Time Period
- **In-sample (training):** Jan 2010 – Dec 2017 (8 years)
- **Out-of-sample (forecast/backtest):** Jan 2018 – Dec 2025 (8 years)
- **Total:** ~4,000 observations per asset (daily)

### Data Sources
| Source | Primary Use | Coverage | Notes |
|--------|------------|----------|-------|
| **yfinance** | Daily prices, primary | All 18 assets, 2010-2025 | Fast, reliable, free |
| **Alpha Vantage** | Backup/high-freq data | All assets | API Key: `0N9ORVW2U03KPT40` |
| **CBOE** | VIX levels/changes | Index-level feature | Manual or pandas-datareader |
| **FRED** | Market breadth | Macro features | Via pandas-datareader |

### Key Metrics to Compute
- **Realized Volatility (RV):** 21-day rolling standard deviation of daily returns, annualized
- **Volatility Regimes:** Low (<15%), Medium (15–25%), High (>25%)
- **Machine Learning Features:** ~40–50 predictors across 5 categories (see Feature Engineering below)

---

## 🏗️ Architectural Decisions

### Data Pipeline
```
Raw Data (yfinance) 
    ↓
Data Cleaning & Validation
    ↓
Feature Engineering (40-50 predictors)
    ↓
Walk-Forward Train/Test Splits
    ↓
Model Training & Forecasting
```

### Walk-Forward Validation Framework
- **Initial training window:** Jan 2010 – Dec 2017
- **Refit frequency:** Monthly (end of each month)
- **Forecast horizon:** 1-day, 5-day, 20-day ahead
- **Test period:** Jan 2018 – Dec 2025 (96 monthly refits)
- **Procedure:** Expanding window (retrain on all data to date, not rolling)

### Model Comparison Structure
Eight models to compare in three categories:

**Traditional Econometric (3):**
1. GARCH(1,1) – baseline
2. EGARCH(1,1) – leverage effects
3. GJR-GARCH(1,1) – threshold asymmetry

**Machine Learning (3):**
4. Random Forest (500–1,000 trees)
5. XGBoost (with early stopping)
6. LSTM (2–3 layers, 64–128 units, dropout 0.2–0.3)

**Hybrid (2):**
7. GARCH-RF (GARCH fitted values + residuals → Random Forest)
8. GARCH-XGBoost (GARCH fitted values + residuals → XGBoost)

### Feature Engineering Categories (40–50 predictors)

| Category | Features | Count |
|----------|----------|-------|
| **Historical Volatility** | Lagged RV (1,2,3,5,10,20,60d), rolling avgs, momentum | ~8 |
| **Return Dynamics** | Lagged returns, abs returns, rolling skew/kurtosis, extremes | ~8 |
| **Market Microstructure** | High-low range, volume (scaled), gap frequency, bid-ask spread proxy | ~6 |
| **Market-Wide Info** | VIX level/change, S&P 500 indicators, market breadth | ~5 |
| **Calendar Effects** | Day-of-week, month-of-year, post-FOMC dummies | ~3–5 |
| **Redundancy Filtering** | Remove features with |correlation| > 0.95 | Applied after construction |

### Evaluation Metrics

**Statistical (accuracy):**
- RMSE (Root Mean Squared Error)
- MAE (Mean Absolute Error)
- QLIKE (Quasi-Likelihood, volatility-specific)
- Directional Accuracy (sign prediction)
- Model Confidence Set (Hansen & Lunde 2006)

**Economic (portfolio performance):**
- Sharpe Ratio
- Sortino Ratio
- Maximum Drawdown
- Calmar Ratio
- Transaction Cost Drag

### Portfolio Strategies (3)
1. **Volatility-Timing:** Dynamic SPY/SHY allocation targeting 15% risk; monthly rebalance; $5/trade costs
2. **Risk-Parity:** 15-stock portfolio weighted inversely by forecasted vol; monthly rebalance; turnover constraints
3. **Regime-Based Allocation:** Discrete equity-bond shift across low/medium/high vol regimes; less frequent trading

---

## 🛠️ Technology Stack

### Environment
```bash
Python 3.9+
Virtual environment: volatility_research

# Core data & ML
pandas, numpy
yfinance, alpha_vantage, pandas_datareader
scikit-learn, xgboost, tensorflow/keras
arch (GARCH), statsmodels

# Backtesting & Analysis
scipy, scikit-optimize (hyperparameter tuning)
matplotlib, seaborn (visualization)

# Optional: Parallelization
joblib (for walk-forward loop)
```

### Project Structure
```
Volatility Research/
├── data/
│   ├── raw/                    # Downloaded price data
│   │   └── {asset}_daily.csv   # One file per asset
│   └── processed/
│       ├── prices_clean.csv    # Merged, deduplicated
│       ├── features.csv        # All 40-50 predictors
│       └── realized_vol.csv    # Target variable (RV)
├── models/
│   ├── garch_baseline.pkl
│   ├── rf_model.pkl
│   ├── xgboost_model.pkl
│   └── lstm_model.h5
├── notebooks/
│   ├── 01_eda.ipynb           # Exploratory analysis (Figures 1-4)
│   ├── 02_garch_baseline.ipynb
│   ├── 03_ml_models.ipynb
│   ├── 04_results.ipynb        # Statistical & economic evaluation
│   └── 05_manuscript.ipynb     # Write-up
├── src/
│   ├── __init__.py
│   ├── data.py                 # Data collection, cleaning
│   ├── features.py             # Feature engineering
│   ├── models.py               # Model wrappers & training
│   ├── evaluation.py           # Loss functions, metrics
│   └── backtest.py             # Portfolio strategies
├── results/
│   ├── forecast_accuracy.csv   # RMSE, QLIKE by model, asset
│   ├── portfolio_performance.csv # Sharpe, drawdown, etc.
│   └── figures/                # Publication-quality plots
├── claude-handoff.md           # This file
├── README.md                   # Project overview & reproduction
├── requirements.txt
└── .gitignore
```

---

## 📅 Weekly Breakdown (10-Week Plan)

### Week 1: Foundation *(IN PROGRESS)*
- [ ] Python environment & dependencies
- [ ] API testing (yfinance, Alpha Vantage, CBOE, FRED)
- [ ] Data collection for all 18 assets
- [ ] Data cleaning & storage pipeline
- [ ] Exploratory analysis (replicating Figures 1-4)

**Deliverable:** EDA notebook + cleaned data files

---

### Week 2: Baseline Setup
- [ ] Implement GARCH(1,1) with `arch` library
- [ ] Set up walk-forward validation framework
- [ ] Generate 1-day, 5-day, 20-day forecasts
- [ ] Compute RMSE, MAE, QLIKE loss functions
- [ ] Model diagnostics (residual tests)

**Deliverable:** Baseline model + forecast accuracy metrics

---

### Week 3: Traditional Econometric Extensions
- [ ] EGARCH(1,1) implementation
- [ ] GJR-GARCH(1,1) implementation
- [ ] HAR-RV model
- [ ] Compare statistical accuracy vs GARCH(1,1)
- [ ] Identify best traditional benchmark

**Deliverable:** Econometric model comparison table

---

### Week 4: Feature Engineering & ML Preparation
- [ ] Construct 40–50 predictor features
- [ ] Handle missing data & scaling
- [ ] Train/validation/test splits (walk-forward ready)
- [ ] Correlation filtering (|r| > 0.95)
- [ ] Feature importance analysis (preliminary)

**Deliverable:** Feature dataset + engineering pipeline

---

### Week 5: Tree-Based ML Models
- [ ] Random Forest (500–1,000 trees, hyperparameter tuning)
- [ ] XGBoost (early stopping, feature importance)
- [ ] Generate out-of-sample forecasts
- [ ] Compare to econometric benchmarks

**Deliverable:** ML forecast accuracy; feature importance rankings

---

### Week 6: LSTM & Hybrid Models
- [ ] LSTM architecture design (2–3 layers, 64–128 units, dropout)
- [ ] Time-series data preparation (20–60 day lookback)
- [ ] Training with early stopping
- [ ] GARCH-RF & GARCH-XGBoost hybrid models
- [ ] Residual analysis

**Deliverable:** LSTM forecasts + hybrid model results

---

### Week 7: Statistical Evaluation
- [ ] RMSE, MAE, QLIKE across all models
- [ ] Model Confidence Set (Hansen & Lunde 2006)
- [ ] Regime-specific performance (crisis vs. calm)
- [ ] Directional accuracy assessment
- [ ] Visualization of forecast comparisons

**Deliverable:** Statistical results table + regime analysis

---

### Week 8: Economic Evaluation & Backtesting
- [ ] Volatility-timing strategy (SPY/SHY)
- [ ] Risk-parity cross-sectional strategy
- [ ] Regime-based tactical allocation
- [ ] Sharpe ratios, max drawdown, transaction costs
- [ ] Performance attribution (forecast accuracy → returns)

**Deliverable:** Portfolio backtest results + Sharpe ratio comparison

---

### Week 9: Robustness & Extensions
- [ ] Cross-asset validation (verify results hold across all 18 assets)
- [ ] Sensitivity analysis (transaction costs, leverage caps, constraints)
- [ ] Out-of-sample stability checks
- [ ] Cost-benefit synthesis (ML complexity vs. economic gains)

**Deliverable:** Robustness tables + cost-benefit summary

---

### Week 10: Documentation & Manuscript
- [ ] Publication-quality visualizations (Figures 1-10)
- [ ] Write full research manuscript (intro, methods, results, discussion)
- [ ] Fall Research Symposium poster
- [ ] Code documentation & reproducibility guide

**Deliverable:** Research manuscript (5,000–8,000 words) + poster

---

## 🚀 Next Immediate Actions (What to do in Claude Code)

### Session 1: Data Foundation
```python
# 1. Collect & validate data
import yfinance as yf
import pandas as pd

assets = ['SPY', 'QQQ', 'IWM', 'AAPL', 'MSFT', 'NVDA', 'JPM', 'GS', 'BAC', 
          'JNJ', 'UNH', 'PFE', 'AMZN', 'WMT', 'HD', 'XOM', 'CVX', 'CAT']

for ticker in assets:
    df = yf.download(ticker, start='2010-01-01', end='2025-12-31', progress=False)
    df.to_csv(f'data/raw/{ticker}_daily.csv')

# 2. Compute realized volatility
for ticker in assets:
    df = pd.read_csv(f'data/raw/{ticker}_daily.csv', index_col=0, parse_dates=True)
    df['returns'] = df['Adj Close'].pct_change()
    df['rv_21d'] = df['returns'].rolling(21).std() * np.sqrt(252) * 100  # Annualized %
    df.to_csv(f'data/processed/{ticker}_with_rv.csv')

# 3. Create merged dataset for ML
# (Combine all 18 assets into one feature matrix)
```

### Session 2: Baseline GARCH Model
```python
# Implement walk-forward GARCH(1,1)
# Train on 2010-2017, forecast Jan 2018 onward
# Monthly refitting

from arch import arch_model
import numpy as np

def garch_forecast(returns, n_forecast=20):
    model = arch_model(returns, vol='Garch', p=1, q=1)
    res = model.fit(disp='off')
    forecast = res.forecast(horizon=n_forecast)
    return forecast.variance.values[-1, :]  # Next n_forecast days
```

### Session 3: Feature Engineering
```python
# Construct 40-50 predictors
# Lagged volatility, returns, skew, kurtosis, VIX, market breadth, etc.
# Use data.py and features.py modules

def engineer_features(df, target_col='rv_21d'):
    """
    Create lagged RV, returns, and market features.
    Apply correlation filtering & standardization.
    """
    pass  # Implement in src/features.py
```

---

## 💾 Credentials & Configuration

**Alpha Vantage API Key:**
```
0N9ORVW2U03KPT40
```
⚠️ Store in environment variable or `.env` file (not in git repo)

**File Paths (in Cowork):**
```
Project folder: /Users/anthonyle/Documents/Denison/Volatility Research
Output folder: /Users/anthonyle/Library/Application Support/Claude/local-agent-mode-sessions/.../outputs
```

---

## 📚 Key References & Literature

**Foundational GARCH Papers:**
- Engle (1982): ARCH framework
- Bollerslev (1986): GARCH model
- Nelson (1991): EGARCH (leverage effects)
- Hansen & Lunde (2005): "Does Anything Beat GARCH(1,1)?"

**Realized Volatility:**
- Andersen et al. (2003): Realized Volatility framework
- Corsi (2009): HAR-RV model

**ML in Finance:**
- Gu, Kelly & Xiu (2020): ML for asset pricing
- Krauss et al. (2017): Deep learning for S&P 500 arbitrage
- Kristjanpoller & Minutolo (2018): Hybrid volatility forecasting

**Economic Evaluation:**
- Fleming, Kirby & Ostdiek (2001, 2003): Volatility-timing economic value
- Moreira & Muir (2017): Volatility-managed portfolios

---

## 🔗 Dependencies Checklist

- [ ] Python 3.9+ installed
- [ ] Virtual environment created (`volatility_research`)
- [ ] `requirements.txt` installed
- [ ] yfinance tested (download SPY data)
- [ ] `arch` library installed (for GARCH)
- [ ] Project folder structure created
- [ ] `.env` file with API key (Alpha Vantage backup)
- [ ] Git repo initialized (optional but recommended)

---

## ⚠️ Known Challenges & Solutions

| Challenge | Solution |
|-----------|----------|
| Alpha Vantage rate limits (5 req/min free) | Use yfinance as primary; Alpha Vantage for backup |
| LSTM overfitting on small datasets | Dropout (0.2–0.3), early stopping, modest architecture |
| Walk-forward computational cost | Parallelize monthly refits with `joblib` |
| Missing intraday data for realized vol | Use 21-day rolling std of daily returns as proxy |
| Multiple model comparisons (look-ahead bias) | Use Model Confidence Set; report 90% & 95% CI |

---

## 📞 Handoff Notes for Claude Code Sessions

**When picking up this project:**
1. Check the task list (`TaskList`) – update status as work progresses
2. Read `src/data.py` comments to understand data pipeline
3. Refer to "Weekly Breakdown" section for context on which week you're in
4. Use `claude-handoff.md` (this file) as your north star for architecture & decisions
5. All model implementations should follow the structure in `src/models.py`
6. Save notebooks in `notebooks/` with 2-digit prefixes (01_, 02_, etc.)

**Git Workflow (if using version control):**
```bash
git add -A
git commit -m "Week X: [brief description]"
git push
```

---

**Last Updated:** May 21, 2026 (Week 1 Initial Planning)  
**Next Review:** After Week 2 baseline GARCH implementation
