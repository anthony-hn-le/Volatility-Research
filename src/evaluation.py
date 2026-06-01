"""
Statistical loss functions and evaluation metrics.
"""
import numpy as np
import pandas as pd


def rmse(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))


def qlike(y_true, y_pred):
    """Quasi-likelihood loss: robust to imperfect volatility proxies."""
    ratio = y_true / (y_pred ** 2)
    return np.mean(ratio - np.log(ratio) - 1)


def directional_accuracy(y_true, y_pred):
    """Fraction of periods where sign of change is correctly predicted."""
    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    return np.mean(true_dir == pred_dir)


def evaluate_forecasts(y_true, forecasts_dict):
    """
    Compute all metrics for a dict of {model_name: y_pred}.
    Returns a DataFrame with models as rows.
    """
    records = []
    for name, y_pred in forecasts_dict.items():
        records.append({
            'model': name,
            'RMSE': rmse(y_true, y_pred),
            'MAE': mae(y_true, y_pred),
            'QLIKE': qlike(y_true, y_pred),
            'DirAcc': directional_accuracy(y_true.values, y_pred),
        })
    return pd.DataFrame(records).set_index('model').sort_values('RMSE')


def regime_evaluation(y_true, forecasts_dict, regimes):
    """
    Evaluate forecasts separately for low/medium/high volatility regimes.
    regimes: Series with values 'low', 'medium', 'high', aligned to y_true.
    """
    results = {}
    for regime in ['low', 'medium', 'high']:
        mask = regimes == regime
        if mask.sum() < 10:
            continue
        subset_true = y_true[mask]
        subset_preds = {k: v[mask.values] for k, v in forecasts_dict.items()}
        results[regime] = evaluate_forecasts(subset_true, subset_preds)
    return results


def vrp_series(iv_t: pd.Series, rv_realized_t21: pd.Series) -> pd.Series:
    """
    Compute the ex-post Variance Risk Premium time series.

        VRP_t = IV_t  −  RV_{t+21}

    Both inputs must be pre-aligned on the same index (month-end dates) and
    in the same units (% annualised, matching rv_21d scale).

    A positive VRP means the market overstated future vol — option sellers
    collected premium. The empirical average VRP is positive over long horizons.

    Args:
        iv_t:            Implied volatility at each month-end (e.g. VIX values).
        rv_realized_t21: Realized volatility over the 21 trading days *after*
                         each month-end (forward-looking, constructed by caller).
    Returns:
        pd.Series named 'VRP', same index as inputs.
    """
    vrp = (iv_t - rv_realized_t21).rename('VRP')
    return vrp


def forecast_vs_iv_accuracy(
    model_forecasts: dict,
    iv_series: pd.Series,
    rv_realized: pd.Series,
    iv_label: str = 'Implied Vol (VIX)',
) -> pd.DataFrame:
    """
    Compare model vol forecasts against implied volatility as competing
    predictors of future realized volatility.

    Adds the IV series as one more entry in the forecast dict, then delegates
    to evaluate_forecasts() — all existing loss functions (RMSE, MAE, QLIKE,
    DirAcc) are reused without modification.

    Appends a 'vs_IV_RMSE_diff' column: negative values mean a model beats
    the market-implied-vol benchmark on RMSE.

    Args:
        model_forecasts: dict of {model_name: pd.Series} on month-end dates.
        iv_series:       IV at each month-end (same index as model_forecasts).
        rv_realized:     Forward-looking RV over the next 21 days (same index).
        iv_label:        Row label for the IV benchmark in the output table.

    Returns:
        pd.DataFrame sorted by RMSE, with an extra 'vs_IV_RMSE_diff' column.
    """
    combined = {**model_forecasts, iv_label: iv_series}

    # Align all series to the common index (drop any dates missing in rv_realized)
    idx = rv_realized.dropna().index
    combined_aligned = {k: v.reindex(idx) for k, v in combined.items()}
    rv_aligned = rv_realized.reindex(idx)

    result = evaluate_forecasts(rv_aligned, combined_aligned)

    iv_rmse = result.loc[iv_label, 'RMSE']
    result['vs_IV_RMSE_diff'] = result['RMSE'] - iv_rmse
    return result
