"""
Build data/processed/features.csv: the feature matrix consumed by the
ML/hybrid walk-forward notebooks (notebooks/03_ml_models.ipynb).

The previous features.csv on disk was an orphaned artifact: no script or
notebook in this repo ever called engineer_features() to produce it, and its
correlation filter (|r|>0.95) had evidently been fit per-ticker rather than
on a single pooled panel — different tickers ended up with different
surviving columns (e.g. `rv_lag5` only survived for AMZN/WMT), which breaks
every downstream consumer that assumes one fixed FEATURE_COLS list (the
walk-forward loops, and especially LSTMModel's fixed input dimensionality).

This script fixes both problems:
  1. Builds each ticker's RAW (unfiltered) features via engineer_features()
     with the same vix_df/spy_rv inputs for all 18 tickers (spy_rv is passed
     even for SPY itself — mildly redundant there, since it's just SPY's own
     past RV, already visible via rv_lag*, but this guarantees every ticker
     produces the identical column set, which matters more than avoiding one
     redundant column for one asset).
  2. Fits the |r|>0.95 correlation filter ONCE, on the pooled rows from all
     18 tickers restricted to dates <= TRAIN_END (2017-12-31) -- using zero
     information from the 2018-2025 test period -- then applies that same
     drop-column list uniformly across the full panel (train+test, all
     tickers). This is a deliberate middle ground: refitting the filter
     inside every one of the 96 monthly x 18 ticker expanding-window slices
     would let the surviving feature set drift over time (breaking the fixed
     dimensionality every consumer assumes) for negligible benefit, since the
     ~50 engineered features are lags/rolling stats of only a handful of
     underlying series whose pairwise correlation structure is not expected
     to cross the 0.95 threshold differently across an 8-year window.

Run directly: `python src/build_features.py` (from the project root, with the
project's .venv activated).
"""
import os

import numpy as np
import pandas as pd

from src.data import ASSETS, TRAIN_END, fetch_vix_daily, load_and_clean
from src.features import engineer_features


def build_features(
    assets=ASSETS,
    raw_dir: str = 'data/raw',
    processed_dir: str = 'data/processed',
    train_end: str = TRAIN_END,
    corr_threshold: float = 0.95,
) -> pd.DataFrame:
    vix_daily = fetch_vix_daily()
    vix_df = vix_daily.rename('Close').to_frame()  # add_market_features expects a 'Close' column

    spy_rv = load_and_clean('SPY', raw_dir=raw_dir)['rv_21d']

    raw_frames = {}
    for ticker in assets:
        df = load_and_clean(ticker, raw_dir=raw_dir)
        X, y = engineer_features(df, vix_df=vix_df, spy_rv=spy_rv, corr_threshold=None)
        panel = X.copy()
        panel['rv_21d'] = y
        panel['ticker'] = ticker
        raw_frames[ticker] = panel
        print(f"  {ticker}: {len(panel):,} rows, {X.shape[1]} raw features")

    full_panel = pd.concat(raw_frames.values())
    feature_cols = [c for c in full_panel.columns if c not in ('ticker', 'rv_21d')]

    # Fit the correlation filter once, on pooled pre-TRAIN_END rows across all tickers.
    train_mask = full_panel.index <= pd.Timestamp(train_end)
    X_train_pooled = full_panel.loc[train_mask, feature_cols]
    corr_matrix = X_train_pooled.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    drop_cols = [col for col in upper.columns if any(upper[col] > corr_threshold)]

    keep_cols = ['ticker', 'rv_21d'] + [c for c in feature_cols if c not in drop_cols]
    full_panel = full_panel[keep_cols]

    os.makedirs(processed_dir, exist_ok=True)
    out_path = os.path.join(processed_dir, 'features.csv')
    full_panel.to_csv(out_path)

    n_features = len(feature_cols) - len(drop_cols)
    print(f"\nBuilt features.csv: {full_panel.shape[0]:,} rows x {full_panel.shape[1]} cols "
          f"({n_features} features + ticker + rv_21d)")
    print(f"Dropped {len(drop_cols)} correlated columns "
          f"(|r|>{corr_threshold}, fit on pooled rows <= {train_end}): {drop_cols}")
    print(f"Saved -> {out_path}")
    return full_panel


if __name__ == '__main__':
    build_features()
