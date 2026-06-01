"""
Feature engineering: ~40-50 predictors for ML models.
All features are constructed using only past information (no look-ahead).
"""
import numpy as np
import pandas as pd


def add_historical_vol_features(df, rv_col='rv_21d'):
    """Lagged RV, rolling averages, volatility momentum."""
    rv = df[rv_col]
    for lag in [1, 2, 3, 5, 10, 20, 60]:
        df[f'rv_lag{lag}'] = rv.shift(lag)
    for window in [5, 20, 60]:
        df[f'rv_ma{window}'] = rv.shift(1).rolling(window).mean()
    df['rv_momentum_5_20'] = df['rv_ma5'] - df['rv_ma20']
    return df


def add_return_features(df, ret_col='return'):
    """Lagged returns, abs returns, rolling skew/kurtosis, extreme-value flags."""
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
    """High-low range, volume ratio, gap frequency."""
    df['hl_range'] = (np.log(df['High']) - np.log(df['Low'])).shift(1)
    df['hl_range_ma5'] = df['hl_range'].rolling(5).mean()
    if 'Volume' in df.columns:
        vol_ma = df['Volume'].rolling(20).mean()
        df['volume_ratio'] = (df['Volume'] / vol_ma).shift(1)
    open_close_gap = (np.log(df['Open']) - np.log(df['Close'].shift(1))).abs()
    df['gap_freq_20'] = (open_close_gap > open_close_gap.rolling(60).quantile(0.9)).shift(1).rolling(20).mean()
    return df


def add_calendar_features(df):
    """Day-of-week and month-of-year dummies."""
    df['dow'] = df.index.dayofweek
    df['month'] = df.index.month
    for d in range(5):
        df[f'dow_{d}'] = (df['dow'] == d).astype(int)
    return df.drop(columns=['dow'])


def add_market_features(df, vix_df=None, spy_rv=None):
    """
    VIX level/change and S&P 500 RV for individual-stock models.
    vix_df: DataFrame with 'Close' column for ^VIX.
    spy_rv: Series of SPY realized volatility.
    """
    if vix_df is not None:
        df['vix_level'] = vix_df['Close'].reindex(df.index).shift(1)
        df['vix_change'] = vix_df['Close'].pct_change().reindex(df.index).shift(1)
    if spy_rv is not None:
        df['spy_rv'] = spy_rv.reindex(df.index).shift(1)
    return df


def engineer_features(df, vix_df=None, spy_rv=None, target_col='rv_21d', corr_threshold=0.95):
    """
    Full feature engineering pipeline.
    Returns (X, y) with look-ahead-free features and the target.
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

    # Correlation filtering
    corr_matrix = X.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    drop_cols = [col for col in upper.columns if any(upper[col] > corr_threshold)]
    X = X.drop(columns=drop_cols)

    return X, y
