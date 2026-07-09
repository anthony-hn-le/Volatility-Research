"""
Statistical loss functions and evaluation metrics.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm


def rmse(y_true, y_pred):
    """Root mean squared error between realized and forecast volatility."""
    return np.sqrt(np.mean((y_true - y_pred) ** 2))


def mae(y_true, y_pred):
    """Mean absolute error between realized and forecast volatility."""
    return np.mean(np.abs(y_true - y_pred))


def qlike_loss(y_true, y_pred):
    """Per-observation QLIKE loss (un-averaged) -- needed by DM/MCS, which
    require a loss *series*, not the aggregate scalar `qlike()` returns."""
    y_pred = np.clip(y_pred, 1e-6, None)   # guard against zero/negative forecasts
    ratio = y_true / (y_pred ** 2)
    return ratio - np.log(ratio) - 1


def qlike(y_true, y_pred):
    """Quasi-likelihood loss: robust to imperfect volatility proxies."""
    return np.mean(qlike_loss(y_true, y_pred))


def directional_accuracy(y_true, y_pred):
    """Fraction of periods where sign of change is correctly predicted."""
    true_dir = np.sign(np.diff(y_true))
    pred_dir = np.sign(np.diff(y_pred))
    return np.mean(true_dir == pred_dir)


def evaluate_forecasts(y_true, forecasts_dict):
    """Compute out-of-sample (OOS) accuracy metrics for multiple models at once.

    Args:
        y_true: Realized volatility over the OOS evaluation period.
        forecasts_dict: {model_name: y_pred} of aligned forecast series.

    Returns:
        pd.DataFrame indexed by model name with columns ['RMSE', 'MAE',
        'QLIKE', 'DirAcc'], sorted ascending by RMSE.
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
    """Evaluate forecasts separately within each volatility regime.

    Args:
        y_true: Realized volatility over the OOS evaluation period.
        forecasts_dict: {model_name: y_pred} of aligned forecast series.
        regimes: Categorical series with values 'low'/'medium'/'high',
            aligned to `y_true` (see `data.load_and_clean`'s `regime` column).

    Returns:
        dict {regime_name: DataFrame}, one `evaluate_forecasts` result per
        regime; regimes with fewer than 10 observations are omitted.
    """
    results = {}
    regimes = pd.array(regimes)  # ensure consistent comparison (handles Categorical too)
    for regime in ['low', 'medium', 'high']:
        mask = (regimes == regime) & pd.notna(regimes)
        if mask.sum() < 10:
            continue
        mask_np = np.asarray(mask)
        subset_true = y_true[mask_np]
        subset_preds = {k: v[mask_np] for k, v in forecasts_dict.items()}
        results[regime] = evaluate_forecasts(subset_true, subset_preds)
    return results


def diebold_mariano(loss_a: np.ndarray, loss_b: np.ndarray, h: int = 1) -> tuple[float, float]:
    """
    Diebold-Mariano test for equal predictive accuracy between two models' losses.

    d_t = loss_a_t - loss_b_t. Under H0 (equal accuracy), E[d_t] = 0. The mean's
    variance is estimated with a Newey-West HAC estimator using h-1 lags, which
    accounts for the MA(h-1) autocorrelation induced by h-step-ahead forecast
    errors (h=1 reduces to the simple sample variance).

    Args:
        loss_a, loss_b: per-observation losses (e.g. squared error, QLIKE) for
            the two models being compared, same length, aligned by time.
        h: forecast horizon in steps (controls the number of HAC lags, h-1).

    Returns:
        (dm_stat, p_value). Negative dm_stat => model A has lower average loss
        (more accurate) than model B. p_value is two-sided (vs. the normal CDF).
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    d = d[~np.isnan(d)]
    T = len(d)
    d_bar = d.mean()

    gamma0 = np.var(d, ddof=0)
    var_d = gamma0
    for lag in range(1, h):
        cov = np.cov(d[lag:], d[:-lag], ddof=0)[0, 1] if T > lag else 0.0
        var_d += 2 * (1 - lag / h) * cov
    se = np.sqrt(var_d / T)

    dm_stat = d_bar / se if se > 0 else np.nan
    p_value = 2 * (1 - norm.cdf(abs(dm_stat)))
    return float(dm_stat), float(p_value)


def model_confidence_set(losses_df: pd.DataFrame, size: float = 0.10,
                          reps: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    Hansen, Lunde & Nason (2011) Model Confidence Set, via arch.bootstrap.MCS.

    Args:
        losses_df: T x k DataFrame of per-observation losses (e.g. squared
            error or QLIKE), one column per model, aligned by row.
        size: test size (e.g. 0.10 for a 90% confidence set).
        reps: bootstrap replications.

    Returns:
        DataFrame indexed by model name with columns ['Pvalue', 'in_mcs'],
        sorted by p-value descending (models most clearly in the MCS first).
    """
    from arch.bootstrap import MCS

    mcs = MCS(losses_df.dropna(), size=size, reps=reps, seed=seed)
    mcs.compute()

    result = mcs.pvalues.copy()
    result['in_mcs'] = ~result.index.isin(mcs.excluded)
    return result.sort_values('Pvalue', ascending=False)


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
