"""
Hyperparameter sensitivity check for XGBoost and Random Forest -- addresses
the peer-review finding that both models' configs were fixed a priori with no
grid/random search, so "was XGBoost just undertuned relative to RF" couldn't
be ruled out.

Scoped to SPY only, same refit-date subset as run_multiseed_lstm.py (every
4th month-end, ~24 dates) to keep this a directional sensitivity read rather
than a full re-run.

Grid: XGBoost max_depth in {3,4,6} (n_estimators=1000, learning_rate=0.05
fixed); RF n_estimators in {300,500,1000} (max_features='sqrt' fixed).

Run from the project root:
    python run_hyperparameter_sensitivity.py

Output: results/hyperparameter_sensitivity.csv
    columns: model, param, value, RMSE_h1_SPY
"""
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

REFIT_DATES = pd.date_range('2017-12-31', '2025-12-31', freq='ME')
SUBSET_DATES = REFIT_DATES[:-1][::4]


def _rv_h1(rv, T):
    future = rv.loc[rv.index > T]
    return float(future.iloc[0]) if len(future) >= 1 else np.nan


def run_grid_point(model_name, param_value, X_all, y_all, rv):
    from src.evaluation import rmse

    preds, actuals = [], []
    for T in SUBSET_DATES:
        X_train = X_all.loc[X_all.index <= T]
        y_train = y_all.loc[y_all.index <= T]
        if len(X_train) < 60:
            continue

        if model_name == 'XGBoost':
            from src.models import XGBoostModel
            model = XGBoostModel(n_estimators=1000, learning_rate=0.05,
                                 max_depth=param_value, random_state=42)
        else:  # Random Forest
            from src.models import RandomForestModel
            model = RandomForestModel(n_estimators=param_value, random_state=42, n_jobs=1)

        model.fit(X_train, y_train)
        fc_val = model.forecast(X_train.iloc[[-1]])
        rv_h1 = _rv_h1(rv, T)
        if not np.isnan(rv_h1):
            preds.append(fc_val)
            actuals.append(rv_h1)

    return float(rmse(np.array(actuals), np.array(preds)))


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

    grid = (
        [('XGBoost', 'max_depth', d) for d in [3, 4, 6]] +
        [('Random Forest', 'n_estimators', n) for n in [300, 500, 1000]]
    )

    rows = []
    for model_name, param_name, value in grid:
        print(f">>> {model_name} {param_name}={value} ...", flush=True)
        r = run_grid_point(model_name, value, X_all, y_all, rv)
        rows.append({'model': model_name, 'param': param_name, 'value': value, 'RMSE_h1_SPY': round(r, 4)})
        print(f"    RMSE_h1_SPY = {r:.4f}")

    out_df = pd.DataFrame(rows)
    out_path = os.path.join(RESULTS, 'hyperparameter_sensitivity.csv')
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")
    print(out_df.to_string(index=False))
