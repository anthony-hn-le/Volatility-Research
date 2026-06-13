"""
Walk-forward runner for LSTM, GARCH-RF, and GARCH-XGB.

Run from the project root:
    python run_walk_forward.py

Outputs (skipped if already present):
    data/processed/forecasts_lstm.csv
    data/processed/forecasts_garch_rf.csv
    data/processed/forecasts_garch_xgb.csv
"""
import os
# Must precede any C-extension import (xgboost / torch / OpenBLAS).
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import time
import warnings
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

warnings.filterwarnings('ignore')


# ---------------------------------------------------------------------------
# Worker functions (defined at module level for pickling; data passed explicitly)
# ---------------------------------------------------------------------------

def _build_record(T, ticker, model_name, fc_val, rv):
    future = rv.loc[rv.index > T]
    return {
        'date':   T,
        'ticker': ticker,
        'model':  model_name,
        'h1':     fc_val,
        'h5':     fc_val,
        'h20':    fc_val,
        'rv_h1':  float(future.iloc[0])  if len(future) >= 1  else np.nan,
        'rv_h5':  float(future.iloc[4])  if len(future) >= 5  else np.nan,
        'rv_h20': float(future.iloc[19]) if len(future) >= 20 else np.nan,
    }


def walk_forward_lstm(ticker, features_df, feature_cols, rv_all, refit_dates):
    # Import here so each worker sets env-vars before loading C extensions.
    from src.models import LSTMModel

    df_t   = features_df[features_df['ticker'] == ticker]
    X_all  = df_t[feature_cols].dropna(axis=1, how='all')
    y_all  = df_t['rv_21d']
    rv     = rv_all[ticker].dropna()
    records = []

    for T in refit_dates[:-1]:
        X_train = X_all.loc[X_all.index <= T]
        y_train = y_all.loc[y_all.index <= T]
        if len(X_train) < 90:
            continue

        model = LSTMModel(hidden_size=128, num_layers=2, dropout=0.2,
                          lookback=30, epochs=50, batch_size=32, random_state=42)
        model.fit(X_train, y_train)
        fc_val = model.forecast(X_train.iloc[[-1]])
        records.append(_build_record(T, ticker, 'LSTM', fc_val, rv))

    return pd.DataFrame(records)


def walk_forward_hybrid(ticker, ml_type, features_df, feature_cols, rv_all,
                        refit_dates, raw_dir):
    from src.models import GARCHHybridModel
    from src.data import load_and_clean

    df_t      = features_df[features_df['ticker'] == ticker]
    X_all     = df_t[feature_cols].dropna(axis=1, how='all')
    y_all     = df_t['rv_21d']
    rv        = rv_all[ticker].dropna()
    raw_df    = load_and_clean(ticker, raw_dir=raw_dir)
    ret_all   = raw_df['return'].dropna()
    records   = []

    for T in refit_dates[:-1]:
        X_train   = X_all.loc[X_all.index <= T]
        y_train   = y_all.loc[y_all.index <= T]
        ret_train = ret_all.loc[ret_all.index <= T]
        if len(X_train) < 60 or len(ret_train) < 60:
            continue

        model = GARCHHybridModel(ml_model_type=ml_type)
        model.fit(X_train, y_train, ret_train)
        fc_val = model.forecast(X_train.iloc[[-1]])
        records.append(_build_record(T, ticker, f'GARCH-{ml_type}', fc_val, rv))

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    from src.data import ASSETS

    ROOT      = os.path.dirname(os.path.abspath(__file__))
    PROCESSED = os.path.join(ROOT, 'data', 'processed')
    RAW       = os.path.join(ROOT, 'data', 'raw')
    REFIT_DATES = pd.date_range('2017-12-31', '2025-12-31', freq='ME')

    print('Loading features and realized vol...', flush=True)
    features_df = pd.read_csv(
        os.path.join(PROCESSED, 'features.csv'),
        index_col=0, parse_dates=True,
    )
    FEATURE_COLS = [c for c in features_df.columns if c not in ('ticker', 'rv_21d')]

    rv_all = pd.read_csv(
        os.path.join(PROCESSED, 'realized_vol.csv'),
        index_col=0, parse_dates=True,
    )
    print(f'  {len(FEATURE_COLS)} feature cols, {len(ASSETS)} assets, '
          f'{len(REFIT_DATES)-1} refit dates\n', flush=True)

    # --- LSTM (parallel across tickers via separate processes) ---------------
    # Each loky worker is a separate process with its own PyTorch instance;
    # OMP_NUM_THREADS=1 is set per-process in src/models.py at import time.
    # n_jobs=4 leaves headroom on Apple Silicon's efficiency cores.
    OUT_LSTM = os.path.join(PROCESSED, 'forecasts_lstm.csv')
    if os.path.exists(OUT_LSTM):
        print('forecasts_lstm.csv already exists — skipping LSTM.', flush=True)
    else:
        print('>>> LSTM walk-forward (parallel, n_jobs=4) ...', flush=True)
        t0 = time.time()
        lstm_results = Parallel(n_jobs=4, verbose=5)(
            delayed(walk_forward_lstm)(
                ticker, features_df, FEATURE_COLS, rv_all, REFIT_DATES
            ) for ticker in ASSETS
        )
        df_lstm = pd.concat(lstm_results, ignore_index=True)
        df_lstm.to_csv(OUT_LSTM, index=False)
        print(f'Done in {time.time()-t0:.0f}s  |  {len(df_lstm):,} rows  →  {OUT_LSTM}\n',
              flush=True)

    # --- GARCH-RF (parallel) ------------------------------------------------
    OUT_GRF = os.path.join(PROCESSED, 'forecasts_garch_rf.csv')
    if os.path.exists(OUT_GRF):
        print('forecasts_garch_rf.csv already exists — skipping GARCH-RF.', flush=True)
    else:
        print('>>> GARCH-RF walk-forward (parallel) ...', flush=True)
        t0 = time.time()
        grf_results = Parallel(n_jobs=-1, verbose=5)(
            delayed(walk_forward_hybrid)(
                ticker, 'RF', features_df, FEATURE_COLS, rv_all, REFIT_DATES, RAW
            ) for ticker in ASSETS
        )
        df_garch_rf = pd.concat(grf_results, ignore_index=True)
        df_garch_rf.to_csv(OUT_GRF, index=False)
        print(f'Done in {time.time()-t0:.0f}s  |  {len(df_garch_rf):,} rows  →  {OUT_GRF}\n',
              flush=True)

    # --- GARCH-XGB (parallel) -----------------------------------------------
    OUT_GXGB = os.path.join(PROCESSED, 'forecasts_garch_xgb.csv')
    if os.path.exists(OUT_GXGB):
        print('forecasts_garch_xgb.csv already exists — skipping GARCH-XGB.', flush=True)
    else:
        print('>>> GARCH-XGB walk-forward (parallel) ...', flush=True)
        t0 = time.time()
        gxgb_results = Parallel(n_jobs=-1, verbose=5)(
            delayed(walk_forward_hybrid)(
                ticker, 'XGB', features_df, FEATURE_COLS, rv_all, REFIT_DATES, RAW
            ) for ticker in ASSETS
        )
        df_garch_xgb = pd.concat(gxgb_results, ignore_index=True)
        df_garch_xgb.to_csv(OUT_GXGB, index=False)
        print(f'Done in {time.time()-t0:.0f}s  |  {len(df_garch_xgb):,} rows  →  {OUT_GXGB}\n',
              flush=True)

    # --- Sanity check -------------------------------------------------------
    print('=== Output summary ===', flush=True)
    for path, label in [
        (os.path.join(PROCESSED, 'forecasts_lstm.csv'),      'LSTM'),
        (os.path.join(PROCESSED, 'forecasts_garch_rf.csv'),  'GARCH-RF'),
        (os.path.join(PROCESSED, 'forecasts_garch_xgb.csv'), 'GARCH-XGB'),
    ]:
        if os.path.exists(path):
            df = pd.read_csv(path)
            spy = df[df['ticker'] == 'SPY']
            print(f'  {label:12s}  {len(df):,} rows  '
                  f'SPY h1: {spy["h1"].min():.1f}%–{spy["h1"].max():.1f}%')
        else:
            print(f'  {label:12s}  MISSING')
