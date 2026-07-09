"""
Multi-seed robustness check for LSTM (and, cheaply, RF/XGBoost) -- addresses
the peer-review finding that LSTM's Diebold-Mariano p-value vs. GARCH(1,1)
(0.0523) sits right at the significance threshold with a single fixed seed
(42) and no variance estimate.

Scoped to SPY only, and a subset of refit dates (every 4th month-end, ~24
dates spanning both the 2018-2021 and 2022-2025 sub-periods already used in
notebooks/04_results.ipynb's stability analysis) x 5 seeds -- full 18-asset,
96-date x 5-seed LSTM reruns would be ~5x the original ~24hr run; this subset
is tractable while still bounding how much of the LSTM-vs-GARCH result is
optimization variance vs. genuine signal.

Run from the project root:
    python run_multiseed_lstm.py

Output: results/lstm_multiseed_spy.csv
    columns: date, seed, model, h1, rv_h1
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import time
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')

SEEDS = [42, 7, 123, 2024, 99]
REFIT_DATES = pd.date_range('2017-12-31', '2025-12-31', freq='ME')
SUBSET_DATES = REFIT_DATES[:-1][::4]  # every 4th month-end, ~24 dates


def _targets(rv, T):
    future = rv.loc[rv.index > T]
    h1 = float(future.iloc[0]) if len(future) >= 1 else np.nan
    return h1


def fit_forecast_one(model_name, ticker, T, seed, X_all, y_all, rv):
    X_train = X_all.loc[X_all.index <= T]
    y_train = y_all.loc[y_all.index <= T]
    if len(X_train) < 90:
        return None

    if model_name == 'LSTM':
        from src.models import LSTMModel
        model = LSTMModel(hidden_size=128, num_layers=2, dropout=0.2,
                          lookback=30, epochs=50, batch_size=32, random_state=seed)
    elif model_name == 'Random Forest':
        from src.models import RandomForestModel
        model = RandomForestModel(n_estimators=500, random_state=seed, n_jobs=1)
    else:  # XGBoost
        from src.models import XGBoostModel
        model = XGBoostModel(n_estimators=1000, learning_rate=0.05,
                             max_depth=4, random_state=seed)

    model.fit(X_train, y_train)
    fc_val = model.forecast(X_train.iloc[[-1]])
    rv_h1 = _targets(rv, T)
    return {'date': T, 'seed': seed, 'model': model_name, 'h1': fc_val, 'rv_h1': rv_h1}


if __name__ == '__main__':
    PROCESSED = 'data/processed'
    RESULTS = 'results'
    os.makedirs(RESULTS, exist_ok=True)

    features_df = pd.read_csv(os.path.join(PROCESSED, 'features.csv'), index_col=0, parse_dates=True)
    FEATURE_COLS = [c for c in features_df.columns if c not in ('ticker', 'rv_21d')]
    rv_all = pd.read_csv(os.path.join(PROCESSED, 'realized_vol.csv'), index_col=0, parse_dates=True)

    df_spy = features_df[features_df['ticker'] == 'SPY']
    X_all = df_spy[FEATURE_COLS]
    y_all = df_spy['rv_21d']
    rv = rv_all['SPY'].dropna()

    print(f"{len(SUBSET_DATES)} refit dates x {len(SEEDS)} seeds x 3 models "
          f"= {len(SUBSET_DATES) * len(SEEDS) * 3} fits (SPY only)")

    jobs = [
        (model_name, T, seed)
        for model_name in ['LSTM', 'Random Forest', 'XGBoost']
        for T in SUBSET_DATES
        for seed in SEEDS
    ]

    t0 = time.time()
    # LSTM fits are the expensive ones; RF/XGBoost are fast regardless of n_jobs.
    # Use a modest n_jobs so LSTM's own single-threaded torch ops aren't starved.
    results = Parallel(n_jobs=4, verbose=5)(
        delayed(fit_forecast_one)(model_name, 'SPY', T, seed, X_all, y_all, rv)
        for model_name, T, seed in jobs
    )
    records = [r for r in results if r is not None]
    df = pd.DataFrame(records)
    out_path = os.path.join(RESULTS, 'lstm_multiseed_spy.csv')
    df.to_csv(out_path, index=False)
    print(f"Done in {time.time()-t0:.0f}s | {len(df):,} rows -> {out_path}")

    for model_name in ['LSTM', 'Random Forest', 'XGBoost']:
        sub = df[df.model == model_name]
        spread = sub.groupby('seed')['h1'].mean()
        print(f"  {model_name}: per-seed mean h1 forecast range "
              f"{spread.min():.2f}-{spread.max():.2f} (seed sensitivity check)")
