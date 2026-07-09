"""
Model wrappers for walk-forward forecasting.
Each model implements fit(returns_or_X, y) and forecast(X_new) -> float.
"""
from __future__ import annotations

# Must be set before any C-extension that links OpenMP/MKL is imported.
# Without this, xgboost and PyTorch deadlock on macOS Apple Silicon.
import os
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

from typing import Literal, Optional, cast

# Matches the Literal accepted by arch_model's vol parameter type stub
_ArchVolLiteral = Literal['GARCH', 'ARCH', 'EGARCH', 'FIGARCH', 'APARCH', 'HARCH']

import numpy as np
import pandas as pd
import xgboost as xgb
from arch import arch_model
from arch.univariate.base import ARCHModelResult
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset




class GARCHModel:
    """Wrapper around `arch.arch_model` for the GARCH family of conditional-
    variance models (Engle 1982; Bollerslev 1986).

    vol='Garch', o=0  → GARCH(p,q)
    vol='Garch', o=1  → GJR-GARCH(p,o,q)  (asymmetric; o is the leverage term,
                         capturing the stylized fact that negative returns
                         raise future volatility more than positive returns
                         of equal magnitude; Glosten, Jagannathan & Runkle 1993)
    vol='EGARCH'      → EGARCH(p,q)       (Nelson 1991; models log-variance,
                         so no non-negativity constraint on parameters)
    """

    def __init__(self, vol: str = 'Garch', p: int = 1, o: int = 0, q: int = 1) -> None:
        """
        Args:
            vol: Conditional-variance specification passed to `arch_model`
                ('Garch' or 'EGARCH'; see class docstring for the o=0/o=1
                distinction under 'Garch').
            p: ARCH order (lagged squared innovations).
            o: Asymmetric/leverage order (0 = symmetric GARCH, 1 = GJR-GARCH).
            q: GARCH order (lagged conditional variance).
        """
        self.vol = vol
        self.p = p
        self.o = o   # asymmetric/leverage order (0 = symmetric GARCH, 1 = GJR-GARCH)
        self.q = q
        self.result: Optional[ARCHModelResult] = None

    def fit(self, returns: pd.Series, y=None) -> "GARCHModel":
        """Fit the conditional-variance model via maximum likelihood.

        Args:
            returns: Daily log returns. Internally rescaled by 100 (i.e. to
                percent units), which is `arch`'s recommended scale for
                numerical stability of the MLE optimizer.
            y: Unused; present only so this wrapper matches the `fit(X, y)`
                interface shared by the ML model wrappers.

        Returns:
            self, with `self.result` set to the fitted `ARCHModelResult`.
        """
        model = arch_model(
            returns * 100,
            vol=cast(_ArchVolLiteral, self.vol),
            p=self.p, o=self.o, q=self.q,
        )
        self.result = model.fit(disp='off')
        return self

    def forecast(self, horizon: int = 1) -> np.ndarray:
        """Forecast conditional volatility 1..horizon steps ahead.

        EGARCH does not support analytic multi-step forecasts (an `arch`
        library limitation); this method falls back to simulation-based
        forecasting (200 paths) automatically when `horizon > 1` and the
        analytic method fails.

        Args:
            horizon: Number of steps ahead to forecast.

        Returns:
            np.ndarray of shape (horizon,): annualized volatility forecasts
            in percent, on the same `rv_21d` scale as the realized-vol target.
        """
        assert self.result is not None, "Call fit() before forecast()"
        try:
            fc = self.result.forecast(horizon=horizon, method='analytic')
        except ValueError:
            # EGARCH: analytic forecasts only available for horizon=1
            fc = self.result.forecast(horizon=horizon,
                                      method='simulation', simulations=200)
        var_pct_sq = fc.variance.values[-1, :]
        # var_pct_sq is daily variance in (% return)^2; ×√252 to match rv_21d scale
        return np.sqrt(var_pct_sq) * np.sqrt(252)  # annualized vol in %


class HARRVModel:
    """Heterogeneous Autoregressive model of Realized Volatility (Corsi 2009).

    OLS regression of RV_t on its own daily (t-1), weekly (5-day mean), and
    monthly (22-day mean) lags -- a parsimonious way to capture the
    long-memory / multi-horizon persistence typically observed in realized
    volatility without a full ARFIMA specification.
    """

    def __init__(self) -> None:
        from statsmodels.regression.linear_model import OLS
        from statsmodels.tools import add_constant
        self._OLS = OLS
        self._add_constant = add_constant
        self.result = None
        self._last_rv: Optional[pd.Series] = None

    def _build_X(self, rv_series: pd.Series) -> pd.DataFrame:
        rv = rv_series.values
        X = pd.DataFrame({
            'rv_d': rv[:-1],
            'rv_w': pd.Series(rv).shift(1).rolling(5).mean().values[:-1],
            'rv_m': pd.Series(rv).shift(1).rolling(22).mean().values[:-1],
        })
        return X.iloc[21:]  # drop NaN warmup

    def fit(self, rv_series: pd.Series, y=None) -> "HARRVModel":
        """Fit the daily/weekly/monthly RV-lag regression via OLS.

        Args:
            rv_series: Realized-volatility series (e.g. `rv_21d`), in the
                same units as the target this model forecasts.
            y: Unused; present only so this wrapper matches the `fit(X, y)`
                interface shared by the ML model wrappers.

        Returns:
            self, with `self.result` set to the fitted `RegressionResults`.
        """
        X = self._build_X(rv_series)
        target = rv_series.values[22:]
        X_const = self._add_constant(X.dropna())
        self.result = self._OLS(target[-len(X_const):], X_const).fit()
        self._last_rv = rv_series
        return self

    def forecast(self, rv_series: Optional[pd.Series] = None) -> float:
        """One-step-ahead RV forecast from the fitted HAR-RV regression.

        Args:
            rv_series: Realized-volatility series to forecast from; defaults
                to the series passed to `fit()`.

        Returns:
            Scalar one-step-ahead volatility forecast, same units as the
            fitted target.
        """
        assert self.result is not None, "Call fit() before forecast()"
        if rv_series is None:
            rv_series = self._last_rv
        rv: np.ndarray = rv_series.to_numpy(dtype=float)  # type: ignore[union-attr]
        # Build x_new as a named-column row and use result.predict() (which aligns
        # by column name against self.result.params) rather than a manual
        # `params @ x_new` dot product. add_constant()'s default prepends 'const'
        # as the FIRST column (['const','rv_d','rv_w','rv_m']), so a manual
        # positional array ending in the constant would silently misalign every
        # coefficient by one slot -- has_constant='add' forces the constant
        # column even though a single-row frame would otherwise look "constant"
        # in every column and confuse add_constant's default column-detection.
        x_new = pd.DataFrame([{
            'rv_d': rv[-1],
            'rv_w': np.mean(rv[-5:]),
            'rv_m': np.mean(rv[-22:]),
        }])
        x_new = self._add_constant(x_new, has_constant='add')
        return float(self.result.predict(x_new).iloc[0])


class HARXModel(HARRVModel):
    """
    HAR-RV augmented with lagged VIX as an exogenous regressor (HAR-X).

    Same OLS architecture as HARRVModel plus one added `vix_lag1` regressor --
    isolating whether VIX's information content (not any ML architecture)
    explains part of the gap between econometric and ML forecast accuracy.
    """

    def _build_X(self, rv_series: pd.Series, vix_series: pd.Series) -> pd.DataFrame:  # type: ignore[override]
        X = super()._build_X(rv_series).copy()
        # Base class's X row j corresponds to rv_series.iloc[21 + j] (verified:
        # X built from rv[:-1] then .iloc[21:] keeps positions 21..N-2 of the
        # original series) -- so vix_lag1 must use the identical positional
        # slice to stay aligned with rv_d's "as of the same date" convention.
        n = len(rv_series)
        vix_full = vix_series.reindex(rv_series.index).to_numpy(dtype=float)
        X['vix_lag1'] = vix_full[21:n - 1]
        return X

    def fit(self, rv_series: pd.Series, vix_series: pd.Series, y=None) -> "HARXModel":  # type: ignore[override]
        X = self._build_X(rv_series, vix_series)
        target = rv_series.values[22:]
        X_const = self._add_constant(X.dropna())
        self.result = self._OLS(target[-len(X_const):], X_const).fit()
        self._last_rv = rv_series
        self._last_vix = vix_series
        return self

    def forecast(self, rv_series: Optional[pd.Series] = None,  # type: ignore[override]
                 vix_series: Optional[pd.Series] = None) -> float:
        assert self.result is not None, "Call fit() before forecast()"
        if rv_series is None:
            rv_series = self._last_rv
        if vix_series is None:
            vix_series = self._last_vix
        rv: np.ndarray = rv_series.to_numpy(dtype=float)  # type: ignore[union-attr]
        vix_last = float(vix_series.reindex(rv_series.index).ffill().iloc[-1])  # type: ignore[union-attr]
        x_new = pd.DataFrame([{
            'rv_d': rv[-1],
            'rv_w': np.mean(rv[-5:]),
            'rv_m': np.mean(rv[-22:]),
            'vix_lag1': vix_last,
        }])
        x_new = self._add_constant(x_new, has_constant='add')
        return float(self.result.predict(x_new).iloc[0])


class RandomForestModel:
    """Random Forest regressor with an internal StandardScaler.

    Ensemble of bagged regression trees (Breiman 2001); feature scaling is
    not statistically required for tree-based models but is applied here for
    a consistent preprocessing interface across the ML wrappers in this
    module (some of which, like LSTMModel, do require it).
    """

    def __init__(
        self,
        n_estimators: int = 500,
        max_features: Literal['sqrt', 'log2'] = 'sqrt',
        random_state: int = 42,
        n_jobs: int = -1,
    ) -> None:
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_features=max_features,
            n_jobs=n_jobs,          # pass 1 when nesting inside an outer Parallel loop
            random_state=random_state,
        )
        self.scaler = StandardScaler()

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "RandomForestModel":
        """Fit the scaler and the forest on the training window.

        Args:
            X: Feature matrix (see `features.engineer_features`).
            y: Target series (realized volatility), aligned to `X`.

        Returns:
            self.
        """
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def forecast(self, X_new: pd.DataFrame) -> float:
        """Predict the target for a single new feature row.

        Args:
            X_new: One-row feature DataFrame, same columns as the training `X`.

        Returns:
            Scalar volatility forecast.
        """
        X_scaled = self.scaler.transform(X_new)
        return float(self.model.predict(X_scaled)[0])

    def feature_importance(self, feature_names) -> pd.Series:
        """Mean decrease in impurity per feature, sorted descending.

        Args:
            feature_names: Column labels matching the order used in `fit()`.

        Returns:
            pd.Series indexed by feature name.
        """
        return pd.Series(
            self.model.feature_importances_, index=feature_names
        ).sort_values(ascending=False)


class XGBoostModel:
    """Gradient-boosted trees (Chen & Guestrin 2016) with early stopping.

    Early stopping uses a chronological hold-out validation split (the most
    recent `val_size` fraction of the training window, not a random split,
    to respect the time-series ordering) to select the boosting round with
    the lowest validation loss, guarding against overfitting the training
    window.
    """

    def __init__(
        self,
        n_estimators: int = 1000,
        learning_rate: float = 0.05,
        max_depth: int = 4,
        random_state: int = 42,
    ) -> None:
        self.params = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            early_stopping_rounds=50,
        )
        self.model: Optional[xgb.XGBRegressor] = None
        self.scaler = StandardScaler()

    def fit(self, X: pd.DataFrame, y: pd.Series, val_size: float = 0.15) -> "XGBoostModel":
        """Fit the scaler and the boosted-tree ensemble with early stopping.

        Args:
            X: Feature matrix, chronologically ordered.
            y: Target series (realized volatility), aligned to `X`.
            val_size: Fraction of the most recent rows held out as the
                early-stopping validation set (chronological, not random).

        Returns:
            self.
        """
        n_val = max(1, int(len(X) * val_size))
        X_tr, X_val = X.iloc[:-n_val], X.iloc[-n_val:]
        y_tr, y_val = y.iloc[:-n_val], y.iloc[-n_val:]
        X_tr_s = self.scaler.fit_transform(X_tr)
        X_val_s = self.scaler.transform(X_val)
        self.model = xgb.XGBRegressor(**self.params, verbosity=0)
        self.model.fit(X_tr_s, y_tr, eval_set=[(X_val_s, y_val)], verbose=False)
        return self

    def forecast(self, X_new: pd.DataFrame) -> float:
        """Predict the target for a single new feature row.

        Args:
            X_new: One-row feature DataFrame, same columns as the training `X`.

        Returns:
            Scalar volatility forecast.
        """
        assert self.model is not None, "Call fit() before forecast()"
        X_scaled = self.scaler.transform(X_new)
        return float(self.model.predict(X_scaled)[0])


# ---------------------------------------------------------------------------
# PyTorch LSTM internals
# ---------------------------------------------------------------------------

class _LSTMNet(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int,
                 dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def _train_lstm_net(
    net: "_LSTMNet",
    X_tr_t: "torch.Tensor",
    y_tr_t: "torch.Tensor",
    X_val_t: "torch.Tensor",
    y_val_t: "torch.Tensor",
    lr: float,
    epochs: int,
    batch_size: int,
) -> None:
    """Train `net` in-place with early stopping."""
    loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t),
        batch_size=batch_size, shuffle=False,
    )
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    best_val, patience, no_improve = float('inf'), 10, 0
    best_state: Optional[dict] = None

    net.train()
    for _ in range(epochs):
        for xb, yb in loader:
            opt.zero_grad()
            loss_fn(net(xb), yb).backward()
            opt.step()

        net.eval()
        with torch.no_grad():
            val_loss = loss_fn(net(X_val_t), y_val_t).item()
        net.train()

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        net.load_state_dict(best_state)
    net.eval()


class LSTMModel:
    """
    PyTorch LSTM for volatility forecasting.

    fit(X, y) builds sliding windows of shape (lookback, n_features), trains
    the net, and stores the last window for forecast().

    forecast(X_new) accepts a single-row DataFrame (the walk-forward loop
    pattern) or a full lookback-length DataFrame.  When given one row it
    slides it into the stored window from fit().
    """

    def __init__(
        self,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        lookback: int = 30,
        lr: float = 1e-3,
        epochs: int = 50,
        batch_size: int = 32,
        random_state: int = 42,
    ) -> None:
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.lookback = lookback
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self._net: Optional[_LSTMNet] = None
        self._x_scaler = StandardScaler()
        self._y_scaler = StandardScaler()
        self._last_window: Optional[np.ndarray] = None  # shape (lookback, features)

    def _build_sequences(self, X_scaled: np.ndarray, y_scaled: np.ndarray):
        seqs, targets = [], []
        for i in range(self.lookback, len(X_scaled) + 1):
            seqs.append(X_scaled[i - self.lookback: i])
            targets.append(y_scaled[i - 1])
        return np.array(seqs, dtype=np.float32), np.array(targets, dtype=np.float32)

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LSTMModel":
        """Build sliding windows, scale, and train the LSTM with early stopping.

        Args:
            X: Feature matrix, chronologically ordered; must have at least
                `lookback + 1` rows.
            y: Target series (realized volatility), aligned to `X`.

        Returns:
            self.

        Raises:
            ValueError: If fewer than 2 sliding windows can be built from `X`.
        """
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        X_scaled = self._x_scaler.fit_transform(X).astype(np.float32)
        y_scaled = self._y_scaler.fit_transform(
            y.values.reshape(-1, 1)
        ).ravel().astype(np.float32)

        seqs, targets = self._build_sequences(X_scaled, y_scaled)
        if len(seqs) < 2:
            raise ValueError(f"Not enough data for lookback={self.lookback}")

        # Chronological 90/10 train/val split
        # Use torch.tensor() (copies) to avoid memory aliasing with numpy views
        n_val = max(1, int(len(seqs) * 0.10))
        X_tr_t = torch.tensor(seqs[:-n_val])
        y_tr_t = torch.tensor(targets[:-n_val])
        X_val_t = torch.tensor(seqs[-n_val:])
        y_val_t = torch.tensor(targets[-n_val:])

        self._net = _LSTMNet(X_scaled.shape[1], self.hidden_size,
                             self.num_layers, self.dropout)

        _train_lstm_net(self._net, X_tr_t, y_tr_t, X_val_t, y_val_t,
                        self.lr, self.epochs, self.batch_size)

        # Store last lookback rows for walk-forward forecast() calls
        self._last_window = X_scaled[-self.lookback:]
        return self

    def forecast(self, X_new: pd.DataFrame) -> float:
        """Forecast the target for the next step.

        Args:
            X_new: A single new feature row (slides into the window stored
                from `fit()`), or a full `lookback`-row feature window.

        Returns:
            Scalar volatility forecast, inverse-transformed back to the
            target's original units.

        Raises:
            ValueError: If `X_new` has neither 1 nor `lookback` rows.
        """
        assert self._net is not None, "Call fit() before forecast()"

        X_new_scaled = self._x_scaler.transform(X_new).astype(np.float32)

        if len(X_new_scaled) == self.lookback:
            window = X_new_scaled
        elif len(X_new_scaled) == 1:
            # Slide stored window: drop oldest, append new row
            window = np.vstack([self._last_window[1:], X_new_scaled])
        else:
            raise ValueError(
                f"X_new must have 1 or {self.lookback} rows, got {len(X_new_scaled)}"
            )

        seq = torch.from_numpy(window[np.newaxis])  # (1, lookback, features)
        with torch.no_grad():
            pred_scaled = self._net(seq).item()

        return float(
            self._y_scaler.inverse_transform([[pred_scaled]])[0, 0]
        )


# ---------------------------------------------------------------------------
# Hybrid GARCH + ML
# ---------------------------------------------------------------------------

class GARCHHybridModel:
    """
    Hybrid model: GARCH fitted values and standardized residuals are appended
    to the engineer_features matrix, then a RandomForest or XGBoost is trained
    on the augmented feature set.

    Usage
    -----
    model = GARCHHybridModel(ml_model_type='RF')
    model.fit(X_train, y_train, returns_train)
    fc = model.forecast(X_new)   # returns_series not needed at forecast time
    """

    def __init__(
        self,
        ml_model_type: Literal['RF', 'XGB'] = 'RF',
        garch_vol: str = 'Garch',
        garch_p: int = 1,
        garch_o: int = 0,
        garch_q: int = 1,
    ) -> None:
        self.ml_model_type = ml_model_type
        self._garch = GARCHModel(vol=garch_vol, p=garch_p, o=garch_o, q=garch_q)
        self._ml: Optional[RandomForestModel | XGBoostModel] = None
        self._garch_fc_vol: Optional[float] = None
        self._garch_last_resid: Optional[float] = None

    def _make_ml_model(self):
        if self.ml_model_type == 'RF':
            return RandomForestModel(n_jobs=1)
        return XGBoostModel()

    def _augment(self, X: pd.DataFrame, fitted_vol: pd.Series,
                 std_resid: pd.Series) -> pd.DataFrame:
        aug = X.copy()
        aug['garch_fitted_vol'] = fitted_vol.reindex(X.index)
        aug['garch_std_resid'] = std_resid.reindex(X.index)
        return aug.dropna(subset=['garch_fitted_vol', 'garch_std_resid'])

    def fit(self, X: pd.DataFrame, y: pd.Series,
            returns: pd.Series) -> "GARCHHybridModel":
        """Fit the GARCH stage, augment features with its output, then fit the ML stage.

        Args:
            X: Feature matrix from `features.engineer_features`.
            y: Target series (realized volatility), aligned to `X`.
            returns: Daily log-return series used to fit the internal GARCHModel.

        Returns:
            self.
        """
        # Fit GARCH on the full returns series available at time T
        self._garch.fit(returns)
        result = self._garch.result

        # Two conventions must match between the training features built here and the
        # forecast-time features built in forecast(), or the ML model learns a mapping
        # that is systematically off at inference time:
        #
        # (1) SCALE: result.conditional_volatility is on the daily, fit-time scale
        # (the model was fit on returns*100), while GARCHModel.forecast() -- used at
        # forecast() time for this same feature slot -- reports annualized vol via
        # `* sqrt(252)`. Annualizing cond_vol here keeps "garch_fitted_vol" on the same
        # scale in training and inference.
        #
        # (2) LOOK-AHEAD: cond_vol[t] is already a one-step-ahead quantity (equivalent to
        # result.forecast(start=t-1, horizon=1)), so it aligns with X[t]'s "information
        # through t-1" convention with no further shift needed. std_resid[t], however,
        # equals resid_t / cond_vol[t], where resid_t is the demeaned return realized ON
        # day t -- same-day information, unlike every other feature in this codebase
        # (all `.shift(1)`'d). It is shifted by one day below so the ML model is trained
        # on the same genuinely-lagged quantity (`std_resid.iloc[-1]` as of T) that
        # forecast() supplies at inference time.
        cond_vol = result.conditional_volatility * np.sqrt(252)  # annualize (% of return -> % p.a.)
        std_resid = result.resid / result.conditional_volatility

        fitted_vol = pd.Series(cond_vol.values, index=returns.index).reindex(X.index)
        std_resid_s = pd.Series(std_resid.values, index=returns.index).shift(1).reindex(X.index)

        X_aug = self._augment(X, fitted_vol, std_resid_s)
        y_aligned = y.reindex(X_aug.index)

        self._ml = self._make_ml_model()
        self._ml.fit(X_aug, y_aligned)

        # Cache one-step-ahead GARCH values for forecast()
        garch_1step = self._garch.forecast(horizon=1)
        self._garch_fc_vol = float(garch_1step[0])
        self._garch_last_resid = float(std_resid.iloc[-1])
        return self

    def forecast(self, X_new: pd.DataFrame) -> float:
        """Predict the target using the cached GARCH output plus new features.

        Args:
            X_new: One-row feature DataFrame, same columns as the training `X`.

        Returns:
            Scalar volatility forecast.
        """
        assert self._ml is not None, "Call fit() before forecast()"
        X_aug = X_new.copy()
        X_aug['garch_fitted_vol'] = self._garch_fc_vol
        X_aug['garch_std_resid'] = self._garch_last_resid
        return self._ml.forecast(X_aug)
