"""
Portfolio backtesting: four strategies using volatility forecasts.

All return series (spy_returns, asset_returns, bond_returns) are expected to be
LOG returns, matching the project-wide convention in src/data.py (`np.log(Close
/ Close.shift(1))`). performance_metrics() compounds them via exp(cumsum()),
not (1+r).cumprod(), which is only valid for simple returns.
"""
import numpy as np
import pandas as pd


TRANSACTION_COST = 5.0  # $ per trade (applied as bp drag approximation)
TARGET_VOL = 15.0       # % annualized


def vol_timing_strategy(spy_returns, vol_forecasts, target_vol=TARGET_VOL,
                        max_leverage=1.5, cost_per_trade=TRANSACTION_COST):
    """Volatility-timing: scale SPY exposure to target constant annualized risk.

    Implements the classic vol-managed-portfolio construction (Fleming, Kirby
    & Ostdiek 2001; Moreira & Muir 2017): weight SPY inversely to its
    forecast volatility so realized portfolio risk is approximately constant
    over time, with the residual allocated to cash/short-duration bonds (SHY).

        w_SPY = clip(target_vol / forecast_vol, 0, max_leverage)

    Rebalanced monthly; a fixed per-trade cost is applied as a bp drag
    proportional to turnover.

    Args:
        spy_returns: Daily log returns for SPY.
        vol_forecasts: Forecast annualized volatility (%) at rebalance dates,
            same units as `target_vol`.
        target_vol: Target annualized portfolio volatility (%).
        max_leverage: Upper bound on `w_SPY` (no leverage cap below 0).
        cost_per_trade: Transaction cost in dollars, converted to bp drag.

    Returns:
        pd.DataFrame with columns ['strategy', 'benchmark', 'weight_spy',
        'cost_drag'], indexed like `spy_returns`.
    """
    weights = (target_vol / vol_forecasts).clip(0, max_leverage)
    weights = weights.resample('ME').last().reindex(spy_returns.index, method='ffill')

    turnover = weights.diff().abs()
    cost_drag = turnover * (cost_per_trade / 10000)

    strategy_returns = weights.shift(1) * spy_returns - cost_drag
    benchmark_returns = spy_returns

    return pd.DataFrame({
        'strategy': strategy_returns,
        'benchmark': benchmark_returns,
        'weight_spy': weights,
        'cost_drag': cost_drag,
    })


def risk_parity_strategy(asset_returns, vol_forecasts_df, cost_per_trade=TRANSACTION_COST):
    """Cross-sectional risk parity across the 15-stock universe.

    Each asset is weighted inversely to its forecast volatility and weights
    are normalized to sum to 1, so every asset contributes approximately
    equal risk to the portfolio rather than equal capital (the equal-weight
    benchmark this strategy is compared against).

    Args:
        asset_returns: DataFrame of daily log returns, assets as columns.
        vol_forecasts_df: DataFrame of forecast volatility, same columns and
            frequency as `asset_returns`.
        cost_per_trade: Transaction cost in dollars, converted to bp drag.

    Returns:
        pd.DataFrame with columns ['strategy', 'equal_weight', 'cost_drag'],
        indexed like `asset_returns`.
    """
    inv_vol = 1.0 / vol_forecasts_df
    weights = inv_vol.div(inv_vol.sum(axis=1), axis=0)
    weights = weights.resample('ME').last().reindex(asset_returns.index, method='ffill')

    turnover = weights.diff().abs().sum(axis=1)
    cost_drag = turnover * (cost_per_trade / 10000)

    strategy_returns = (weights.shift(1) * asset_returns).sum(axis=1) - cost_drag
    equal_weight = asset_returns.mean(axis=1)

    return pd.DataFrame({
        'strategy': strategy_returns,
        'equal_weight': equal_weight,
        'cost_drag': cost_drag,
    })


def regime_strategy(spy_returns, vol_forecasts, bond_returns=None, cost_per_trade=TRANSACTION_COST):
    """Discrete equity/bond allocation keyed to the forecast volatility regime.

    Low (<15%): 100% SPY | Medium (15-25%): 60/40 | High (>25%): 20/80.
    Regime thresholds match the low/medium/high cut points used elsewhere in
    this project (`data.load_and_clean`'s `regime` column). Rebalanced
    monthly with the same fixed-cost convention as the other strategies.

    Args:
        spy_returns: Daily log returns for SPY.
        vol_forecasts: Forecast annualized volatility (%) at rebalance dates.
        bond_returns: Daily log returns for the bond sleeve; defaults to a
            constant ~10%/year proxy if not supplied.
        cost_per_trade: Transaction cost in dollars, converted to bp drag.

    Returns:
        pd.DataFrame with columns ['strategy', 'equity_weight', 'cost_drag'],
        indexed like `spy_returns`.
    """
    if bond_returns is None:
        bond_returns = pd.Series(0.0004, index=spy_returns.index)  # ~10% annual proxy

    equity_weight = pd.cut(
        vol_forecasts,
        bins=[0, 15, 25, np.inf],
        labels=[1.0, 0.6, 0.2]
    ).astype(float)
    equity_weight = equity_weight.resample('ME').last().reindex(spy_returns.index, method='ffill')
    bond_weight = 1.0 - equity_weight

    turnover = equity_weight.diff().abs()
    cost_drag = turnover * (cost_per_trade / 10000)

    strategy_returns = (equity_weight.shift(1) * spy_returns
                        + bond_weight.shift(1) * bond_returns
                        - cost_drag)
    return pd.DataFrame({
        'strategy': strategy_returns,
        'equity_weight': equity_weight,
        'cost_drag': cost_drag,
    })


def vrp_strategy(
    spy_returns: pd.Series,
    model_forecasts: pd.Series,
    iv_series: pd.Series,
    cost_per_trade: float = TRANSACTION_COST,
) -> pd.DataFrame:
    """
    Variance Risk Premium (VRP) strategy: exploit the spread between implied
    volatility and the model's vol forecast.

    Signal logic (proportional, no leverage):
        vrp_spread = IV_t − model_forecast_t
        signal_t   = clip(vrp_spread / IV_t, 0, 1)

    When the model thinks IV is too high (vrp_spread > 0), the strategy goes
    long VRP (i.e. short implied vol).  When the model agrees with or exceeds
    IV, the signal is zero (flat).

    Return proxy — daily P&L of a delta-neutral short straddle:
        daily_iv_decay = IV / 100 / √252   (theta earned per day)
        daily_rv_cost  = |r_t|             (realized move paid)
        vrp_daily_pnl  = daily_iv_decay − daily_rv_cost

    Both terms are in log-return space (daily fraction), consistent with how
    'return' is stored in the project data pipeline.

    Monthly rebalance with $5/trade transaction-cost drag (same convention as
    the three existing strategies). The 'strategy' column is directly consumable
    by performance_metrics().

    Args:
        spy_returns:     Daily log returns for SPY over the test period.
        model_forecasts: Model's annualised vol forecast (%) at each month-end,
                         indexed by month-end dates.
        iv_series:       IV proxy (VIX, %) at the same month-end dates.
        cost_per_trade:  Transaction cost in dollars (converted to bp drag).

    Returns:
        pd.DataFrame with columns ['strategy', 'benchmark', 'vrp_signal',
        'vrp_spread', 'cost_drag'].
    """
    # ── Signal (month-end frequency) ──────────────────────────────────────────
    vrp_spread = iv_series - model_forecasts           # positive → IV > model
    signal = (vrp_spread / iv_series).clip(lower=0, upper=1)

    # Forward-fill to daily frequency (position set at month-end close)
    signal_daily = (
        signal
        .resample('ME').last()
        .reindex(spy_returns.index, method='ffill')
        .fillna(0.0)
    )
    vrp_spread_daily = (
        vrp_spread
        .reindex(spy_returns.index, method='ffill')
        .fillna(0.0)
    )

    # ── Daily P&L proxy for a delta-neutral short straddle ────────────────────
    iv_daily = (
        iv_series
        .reindex(spy_returns.index, method='ffill')
        .bfill()   # fill any leading NaN at the start of the test period
    )
    daily_iv_decay = iv_daily / 100.0 / np.sqrt(252)  # theta earned (daily %)
    daily_rv_cost  = spy_returns.abs()                 # realized move paid
    vrp_daily_pnl  = daily_iv_decay - daily_rv_cost

    # ── Transaction costs ─────────────────────────────────────────────────────
    turnover  = signal_daily.diff().abs()
    cost_drag = turnover * (cost_per_trade / 10_000)

    # ── Strategy returns ──────────────────────────────────────────────────────
    strategy_returns = signal_daily.shift(1) * vrp_daily_pnl - cost_drag

    return pd.DataFrame({
        'strategy':   strategy_returns,
        'benchmark':  spy_returns,
        'vrp_signal': signal_daily,
        'vrp_spread': vrp_spread_daily,
        'cost_drag':  cost_drag,
    })


def performance_metrics(returns, rf_rate=0.04, cost_drag=None):
    """Compute annualized risk/return metrics from a daily log-return series.

    Equity is compounded via `exp(cumsum(returns))`, not `(1+returns).cumprod()`
    — the latter is only valid for simple returns and would silently misstate
    Max Drawdown/Calmar given this project's log-return convention.

    Args:
        returns: Daily log-return series (strategy or benchmark).
        rf_rate: Annualized risk-free rate used in the Sharpe/Sortino excess-
            return numerator.
        cost_drag: Optional per-period transaction-cost series (as returned
            by the strategy functions in this module); if given, its
            annualized mean is reported separately rather than left implicit
            inside `returns`.

    Returns:
        dict with keys 'Ann. Return', 'Ann. Volatility', 'Sharpe', 'Sortino',
        'Max Drawdown', 'Calmar', and (if `cost_drag` given) 'Ann. Cost Drag'.
    """
    ann_ret = returns.mean() * 252
    ann_vol = returns.std() * np.sqrt(252)
    sharpe = (ann_ret - rf_rate) / ann_vol if ann_vol > 0 else np.nan

    downside = returns[returns < 0].std() * np.sqrt(252)
    sortino = (ann_ret - rf_rate) / downside if downside > 0 else np.nan

    cum = np.exp(returns.cumsum())
    drawdown = (cum / cum.cummax() - 1)
    max_dd = drawdown.min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else np.nan

    result = {
        'Ann. Return': ann_ret,
        'Ann. Volatility': ann_vol,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'Max Drawdown': max_dd,
        'Calmar': calmar,
    }
    if cost_drag is not None:
        result['Ann. Cost Drag'] = cost_drag.mean() * 252
    return result
