"""
Feature engineering: ~40-50 predictors for ML models.
All features are constructed using only past information (no look-ahead).
"""
from typing import Optional

import numpy as np
import pandas as pd


def add_historical_vol_features(df, rv_col='rv_21d'):
    """Add lagged realized-volatility, rolling-average, and momentum features.

    Args:
        df: DataFrame containing `rv_col`.
        rv_col: Name of the realized-volatility column to derive lags from.

    Returns:
        `df` with added columns `rv_lag{1,2,3,5,10,20,60}`, `rv_ma{5,20,60}`
        (rolling means of the once-lagged series), and `rv_momentum_5_20`
        (short- minus medium-window rolling mean, a volatility trend signal).
    """
    rv = df[rv_col]
    for lag in [1, 2, 3, 5, 10, 20, 60]:
        df[f'rv_lag{lag}'] = rv.shift(lag)
    for window in [5, 20, 60]:
        df[f'rv_ma{window}'] = rv.shift(1).rolling(window).mean()
    df['rv_momentum_5_20'] = df['rv_ma5'] - df['rv_ma20']
    return df


def add_return_features(df, ret_col='return'):
    """Add lagged-return, distributional-shape, and extreme-move features.

    Args:
        df: DataFrame containing `ret_col`.
        ret_col: Name of the daily log-return column.

    Returns:
        `df` with added columns `ret_lag{1,2,3,5}`, `absret_lag{1,2,3,5}`,
        `ret_skew_20`/`ret_kurt_20` (rolling skewness/kurtosis, capturing
        return-distribution asymmetry and tail risk), `max_abs_ret_20`
        (20-day maximum absolute return), and `ret_sq_lag1` (lagged squared
        return, an ARCH-effect proxy).
    """
    r = df[ret_col]
    for lag in [1, 2, 3, 5]:
        df[f'ret_lag{lag}'] = r.shift(lag)
        df[f'absret_lag{lag}'] = r.abs().shift(lag)
    df['ret_skew_20'] = r.shift(1).rolling(20).skew()
    df['ret_kurt_20'] = r.shift(1).rolling(20).kurt()
    df['max_abs_ret_20'] = r.abs().shift(1).rolling(20).max()
    df['ret_sq_lag1'] = (r ** 2).shift(1)
    return df


def add_microstructure_features(df):
    """Add intraday range, relative-volume, and price-gap features.

    Args:
        df: DataFrame with 'High', 'Low', 'Open', 'Close' columns, and
            optionally 'Volume'.

    Returns:
        `df` with added columns `hl_range` (Parkinson-style log high-low
        range, a proxy for intraday volatility) and its 5-day rolling mean
        `hl_range_ma5`; `volume_ratio` (volume relative to its 20-day mean,
        if 'Volume' is present); and `gap_freq_20` (fraction of the last 20
        days with an overnight open-close gap in the top decile of its
        60-day distribution).
    """
    df['hl_range'] = (np.log(df['High']) - np.log(df['Low'])).shift(1)
    df['hl_range_ma5'] = df['hl_range'].rolling(5).mean()
    if 'Volume' in df.columns:
        vol_ma = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = (df['Volume'] / vol_ma).shift(1)
    open_close_gap = (np.log(df['Open']) - np.log(df['Close'].shift(1))).abs()
    df['gap_freq_20'] = (open_close_gap > open_close_gap.rolling(60).quantile(0.9)).shift(1).rolling(20).mean()
    return df


def add_calendar_features(df):
    """Add day-of-week dummies and a month-of-year column.

    Args:
        df: DataFrame with a DatetimeIndex.

    Returns:
        `df` with added columns `dow_{0..4}` (Monday-Friday indicator dummies)
        and `month` (calendar month, 1-12).
    """
    df['dow'] = df.index.dayofweek
    df['month'] = df.index.month
    for d in range(5):
        df[f'dow_{d}'] = (df['dow'] == d).astype(int)
    return df.drop(columns=['dow'])


def add_market_features(df, vix_df=None, spy_rv=None):
    """Add market-wide (as opposed to asset-specific) predictors.

    Args:
        df: DataFrame with a DatetimeIndex to reindex the market series onto.
        vix_df: DataFrame with a 'Close' column of ^VIX levels, or None to skip.
        spy_rv: Series of SPY realized volatility, or None to skip.

    Returns:
        `df` with added columns `vix_level`/`vix_change` (lagged VIX level and
        percent change, if `vix_df` given) and `spy_rv` (lagged SPY realized
        volatility, if given) -- market-wide risk signals available to every
        individual-stock model, not just the three index tickers.
    """
    if vix_df is not None:
        df['vix_level'] = vix_df['Close'].reindex(df.index).shift(1)
        df['vix_change'] = vix_df['Close'].pct_change().reindex(df.index).shift(1)
    if spy_rv is not None:
        df['spy_rv'] = spy_rv.reindex(df.index).shift(1)
    return df


def engineer_features(
    df,
    vix_df=None,
    spy_rv=None,
    target_col='rv_21d',
    corr_threshold: Optional[float] = 0.95,
    precomputed_drop_cols: Optional[list] = None,
):
    """Run the full feature engineering pipeline and split into (X, y).

    Applies `add_historical_vol_features`, `add_return_features`,
    `add_microstructure_features`, `add_calendar_features`, and
    `add_market_features` in sequence, drops rows with any remaining NaN
    (from rolling-window warmup), then separates predictors from the target.
    Every predictor is constructed from information available at or before
    t-1 relative to the target at t (each feature group applies `.shift(1)`
    internally), so no look-ahead bias is introduced.

    Correlation filtering is applied one of three ways, in priority order:
      - `precomputed_drop_cols` given: drop exactly these columns, no fitting.
        Lets a caller (e.g. `build_features.py`) fit the filter once on a
        pooled, training-only panel across tickers, then apply the same drop
        list everywhere — avoiding both (a) a per-ticker filter, which can
        yield a different surviving-column set per ticker, and (b) fitting
        the filter using any information from the test period.
      - `corr_threshold` is a float and `precomputed_drop_cols` is None: fit
        the filter on whatever `X` is passed to this call.
      - `corr_threshold` is None and `precomputed_drop_cols` is None: skip
        filtering entirely and return the raw (unfiltered) feature set.

    Args:
        df: Cleaned per-asset OHLCV DataFrame (output of `data.load_and_clean`).
        vix_df: Optional DataFrame with 'Close' column of ^VIX levels.
        spy_rv: Optional Series of SPY realized volatility.
        target_col: Column in `df` to use as the forecast target.
        corr_threshold: Absolute pairwise-correlation cutoff for the
            redundancy filter (applied only when `precomputed_drop_cols`
            is None); pass None to skip filtering.
        precomputed_drop_cols: Explicit columns to drop, bypassing filter
            fitting; takes priority over `corr_threshold`.

    Returns:
        Tuple `(X, y)`: `X` is the predictor DataFrame after NaN-dropping and
        correlation filtering; `y` is the aligned target Series.
    """
    df = df.copy()
    df = add_historical_vol_features(df, rv_col=target_col)
    df = add_return_features(df)
    df = add_microstructure_features(df)
    df = add_calendar_features(df)
    df = add_market_features(df, vix_df=vix_df, spy_rv=spy_rv)

    feature_cols = [c for c in df.columns if c not in
                    ['Open', 'High', 'Low', 'Close', 'Volume', 'return',
                     'rv_21d', 'regime', 'month']]

    df = df.dropna(subset=feature_cols + [target_col])
    X = df[feature_cols]
    y = df[target_col]

    if precomputed_drop_cols is not None:
        X = X.drop(columns=[c for c in precomputed_drop_cols if c in X.columns])
    elif corr_threshold is not None:
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        drop_cols = [col for col in upper.columns if any(upper[col] > corr_threshold)]
        X = X.drop(columns=drop_cols)

    return X, y
