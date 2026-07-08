# Backtesting Ideas: from `build-a-quant-trading-strategy/` to this project

Source material reviewed: `build-a-quant-trading-strategy/{video1,video2,video3}.ipynb`, `research.py`, `models.py`, `binance.py` (all executed, outputs read). Target: `src/backtest.py`, `src/evaluation.py`, `claude-handoff.md` Week 8 (Economic Evaluation & Backtesting).

This is a research/reading document only — no code was changed to produce it.

## Summary table

| # | Idea | Source | Target here | Priority |
|---|---|---|---|---|
| 1 | Standalone transaction-cost-drag metric | `research.py:886-893` (`add_tx_fees_log`) | `performance_metrics` (`src/backtest.py:160-181`) | **High** |
| 2 | `regime_strategy` doesn't charge transaction costs | (gap, not a notebook idea) | `regime_strategy` (`src/backtest.py:57-74`) | **High** |
| 3 | Overfitting/multiple-comparisons caution | `research.py:1016-1032` (`benchmark_linear_models`), video1 cell 88 (Sharpe 10.04 from brute-force combo search) | Week 7/8 evaluation generally, alongside planned Diebold-Mariano/MCS work | **High** |
| 4 | Vol-scaled sizing vs. flat leverage | video2 cells 99-108 (flat 4x/8x leverage) vs. `vol_timing_strategy` (`src/backtest.py:12-32`) | No action needed — this project is already ahead | Info only |
| 5 | Intrabar/worst-case risk check (liquidation pattern) | video2 cells 113, 126-128 (`long_liquidation_price`, high/low check) | Optional stop-out/drawdown-breach check for `vol_timing_strategy`/`regime_strategy` | Medium |
| 6 | Compounding equity curve correctness check | video2 cells 71-89 (log-return additivity) | Confirm `(1+returns).cumprod()` in `performance_metrics` matches how returns are computed in `src/data.py`/`src/backtest.py` inputs | Medium |
| 7 | Sharpe-annualization parameterized by trading calendar | `research.py:289-324` (`sharpe_annualization_factor`) | `performance_metrics` (`src/backtest.py:162-163`) | Low |
| 8 | Event-driven Strategy/Exchange/Account framework | video3.ipynb cells 60-92 | Not recommended for this project | Rejected |
| 9 | Binance download/caching, liquidation formulas, 24/7-hours assumption | `binance.py`, video2 liquidation formulas | Not applicable | Rejected |

---

## 1. Standalone transaction-cost-drag metric (High priority)

**What the crypto repo does:** `research.py` computes both a gross and a fee-adjusted equity curve side by side (`add_tx_fees_log`, `research.py:886-893`), so the cost of trading is always visible as its own number, not just silently subtracted into the return series:

```python
def add_tx_fees_log(trades, maker_fee, taker_fee):
    maker_roundtrip = np.log(1 - 2*maker_fee)
    taker_roundtrip = np.log(1 - 2*taker_fee)
    return trades.with_columns([
        (pl.col('trade_log_return') + maker_roundtrip).alias('maker_net_log_return'),
        (pl.col('trade_log_return') + taker_roundtrip).alias('taker_net_log_return'),
    ])
```

**Why it matters here:** `claude-handoff.md:140` lists "Transaction Cost Drag" as its own economic-evaluation metric, on par with Sharpe/Sortino/MaxDD/Calmar. But in `src/backtest.py`, every strategy function computes `cost_drag` internally and subtracts it directly into `strategy_returns` (e.g. `src/backtest.py:25`: `strategy_returns = weights.shift(1) * spy_returns - cost_drag`) — the drag is never returned or reported separately. Right now there's no way to answer "how much of this strategy's underperformance vs. benchmark is turnover cost vs. bad timing?"

**Concrete suggestion:** Have each strategy function also return the `cost_drag` series (it's already computed locally in all three cost-charging strategies), and add a small helper — e.g. `annualized_cost_drag(cost_drag_series) = cost_drag_series.mean() * 252` — either as a new function in `src/backtest.py` or as an extra key in `performance_metrics`'s output dict when a cost series is passed in.

## 2. `regime_strategy` doesn't charge transaction costs (High priority — gap, not a notebook idea)

Not something from the crypto notebooks, but surfaced while reading `src/backtest.py` for comparison: `vol_timing_strategy` (`src/backtest.py:22-23`) and `risk_parity_strategy` (`src/backtest.py:45-46`) both compute `turnover` and subtract `cost_drag`. `regime_strategy` (`src/backtest.py:57-74`) rebalances monthly just like the other two but never computes turnover or a cost_drag term at all — line 73 has no cost term:

```python
strategy_returns = equity_weight.shift(1) * spy_returns + bond_weight.shift(1) * bond_returns
```

This makes `regime_strategy` look artificially better than the other two strategies in any side-by-side comparison, since it's the only one trading for free. Worth fixing before any cross-strategy comparison table goes into the manuscript — same 5bp-per-turnover-unit pattern as the other two strategies would make it consistent.

## 3. Overfitting / multiple-comparisons caution (High priority)

**What happened in the crypto repo:** `benchmark_linear_models` (`research.py:1016-1032`) retrains a fresh `LinearModel` for every combination of up to 3 lag features via `itertools.combinations`, ranks purely by Sharpe on one static 25-30% hold-out slice, and keeps the winner. This produced a "Sharpe 10.04" model (video1 cell 88, 12h bars, 3-lag combo) that was saved and reused as the flagship model in videos 2-3. There is no walk-forward re-estimation anywhere in that codebase (`timeseries_split`, `research.py:768-793`, is a single non-shuffled chronological split, used everywhere). A Sharpe of 10 from combinatorial search on one small test set (a few hundred trades) is a textbook case of backtest overfitting / selection bias, not a real edge.

**Why it's directly relevant here:** this project's pipeline walk-forward fits 9 models (GARCH, GJR-GARCH, EGARCH, HAR-RV, RF, XGBoost, LSTM, GARCH-RF, GARCH-XGB) × 18 assets × 3 horizons (per `run_walk_forward.log` and `src/models.py`). That's a much larger comparison surface than the crypto repo's feature-combo search, and `claude-handoff.md:133` already calls for a Model Confidence Set (Hansen & Lunde 2006) specifically to guard against this. Neither DM nor MCS exists in the codebase yet (confirmed via grep — no hits for "diebold", "mariano", "confidence set" anywhere in `src/` or the notebooks).

**Concrete suggestion:** when Week 7/8 work resumes, treat the crypto repo's inflated Sharpe as a concrete illustration to cite for *why* the MCS/DM step isn't optional — and apply the same skepticism to any single "best Sharpe" strategy/model that emerges from the 9×18×3 grid before reporting it, exactly as the crypto repo's "Sharpe 10" model should not be taken at face value.

## 4. Vol-scaled sizing vs. flat leverage (info only — no action needed)

The crypto repo sizes positions with a flat leverage multiplier (4x or 8x, chosen manually, video2 cells 99-103) with **no** volatility targeting — leverage doesn't adapt to predicted or realized vol at all, and this is explicitly flagged as unaddressed in the notebook's own closing markdown cell. This project's `vol_timing_strategy` (`src/backtest.py:19`: `weights = (target_vol / vol_forecasts).clip(0, max_leverage)`) already does the more sophisticated thing — inverse-vol-scaled weighting, capped rather than flat. No change needed; noting this so it's clear the comparison was made and this project isn't missing anything here.

## 5. Intrabar / worst-case risk check pattern (Medium priority)

**What the crypto repo does:** rather than checking risk only at the close, video2's liquidation check (cells 113, 126-128) computes a worst-case adverse price per position and checks it against the bar's **`high`/`low`**, not just `close`:

```python
def long_liquidation_price(p, l, mmr):
    return (p * l) / (l + 1 - mmr * l)
# then, per bar:
# pl.when(dir_signal == 1).then(low <= liquidation_price).when(dir_signal == -1).then(high >= liquidation_price)
```

The specific formula (perpetual-futures margin calls) doesn't apply to equities, but the **pattern** — "check whether a risk threshold was breached at any point during the period, not just at the sampling boundary" — is general.

**Concrete suggestion:** this project's monthly-rebalance strategies only observe vol forecasts/weights at month-end. A cheap adaptation: check whether daily drawdown *within* a month ever breached some threshold (e.g. a 2× target-vol daily move) between rebalances, as a robustness diagnostic for `vol_timing_strategy`/`regime_strategy` — i.e., "did the monthly rebalance cadence miss a risk event that a more frequent check would have caught?" This is a Week 9 (Robustness & Extensions) style question, not required for the base Week 8 deliverable.

## 6. Compounding equity curve correctness (Medium priority)

The crypto repo makes a point of demonstrating log-return time-additivity explicitly (video2 cells 71-77: `capital*exp(r1+r2+r3) == ((capital*exp(r1))*exp(r2))*exp(r3)`) before building compounding equity curves on it. `performance_metrics` here uses `cum = (1 + returns).cumprod()` (`src/backtest.py:169`), which is the correct construction **only if** `returns` are simple (arithmetic) returns, not log returns. Worth a quick sanity check against `src/data.py`/`src/features.py` to confirm `spy_returns`/`asset_returns` fed into `backtest.py` are simple returns (they likely are, since RV features elsewhere in the project use log returns for `rv_21d` but that's a different series) — if any log-return series were ever passed into `performance_metrics` by mistake, drawdown/Calmar would be silently wrong. No code changed here; just flagging it as a one-line thing to verify before Week 8 results go final.

## 7. Sharpe-annualization parameterized by trading calendar (Low priority)

`sharpe_annualization_factor(interval, trading_days_per_year, trading_hours_per_day)` (`research.py:289-324`) is a clean, reusable pattern: it converts a bar-level Sharpe to annualized Sharpe generically from an interval string, rather than hardcoding `sqrt(252)`. Since this project only ever uses daily bars, `performance_metrics`'s hardcoded `np.sqrt(252)` (`src/backtest.py:163,166`) is already correct and doesn't need to change — but if any intraday or non-daily-bar work is ever added (e.g. testing weekly rebalancing sensitivity in Week 9), this parameterized-helper pattern is worth copying rather than hardcoding a new constant.

## 8. Event-driven Strategy/Exchange/Account framework — not recommended

video3.ipynb builds a full ABC-based event-driven simulator (`Tick`/`Order`/`Trade`/`Position`/`Account`/`Exchange`/`Strategy`, cells 60-92) for live/paper trading. This is a good pattern in general but is solving a different problem (live tick-by-tick execution) than this project needs (monthly-rebalance research backtesting on daily bars). Adopting it here would be over-engineering — the existing vectorized pandas approach in `src/backtest.py` is the right tool for a research paper's economic-evaluation section. Included here only for completeness, so it's clear this was considered and deliberately not recommended.

## 9. Explicitly not transferable

- `binance.py`'s tick-data download/parquet-caching logic — Binance Futures API-specific, no analog needed (this project already has `src/data.py` for yfinance/Alpha Vantage).
- The perpetual-futures liquidation-price formulas (`long_liquidation_price`/`short_liquidation_price`) — margin-call mechanics don't exist in the same form for the long-only equity/bond strategies here.
- `sharpe_annualization_factor`'s default `trading_hours_per_day=24` — crypto-specific; would need `6.5` for any equity intraday work, not relevant at daily-bar frequency.
- Sign-only signal generation with flat leverage (video1/video2's `dir_signal = y_hat.sign()`) — not applicable; this project's strategies already use continuous, vol-scaled weights rather than binary long/short signals.
