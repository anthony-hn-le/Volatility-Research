"""
Model wrappers for walk-forward forecasting.
Each model implements fit(returns_or_X, y) and forecast(X_new) -> float.
"""
from __future__ import annotations

from typing import Literal, Optional, cast

# Matches the Literal accepted by arch_model's vol parameter type stub
_ArchVolLiteral = Literal['GARCH', 'ARCH', 'EGARCH', 'FIGARCH', 'APARCH', 'HARCH']

import numpy as np
import pandas as pd
from arch import arch_model
from arch.univariate.base import ARCHModelResult
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import xgboost as xgb


class GARCHModel:
    """
    GARCH-family wrapper.

    vol='Garch', o=0  → GARCH(p,q)
    vol='Garch', o=1  → GJR-GARCH(p,o,q)  (asymmetric; o is the leverage term)
    vol='EGARCH'      → EGARCH(p,q)
    """

    def __init__(self, vol: str = 'Garch', p: int = 1, o: int = 0, q: int = 1) -> None:
        self.vol = vol
        self.p = p
        self.o = o   # asymmetric/leverage order (0 = symmetric GARCH, 1 = GJR-GARCH)
        self.q = q
        self.result: Optional[ARCHModelResult] = None

    def fit(self, returns: pd.Series, y=None) -> "GARCHModel":
        model = arch_model(
            returns * 100,
            vol=cast(_ArchVolLiteral, self.vol),
            p=self.p, o=self.o, q=self.q,
        )
        self.result = model.fit(disp='off')
        return self

    def forecast(self, horizon: int = 1) -> np.ndarray:
        """
        Return h-step-ahead annualized vol forecasts (% annual), shape (horizon,).

        EGARCH does not support analytic multi-step forecasts (arch library
        limitation); the method falls back to simulation-based forecasting
        (200 paths) automatically when `horizon > 1` and analytic fails.
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
    """HAR-RV: OLS regression on daily, weekly, monthly RV lags (Corsi 2009)."""

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
        X = self._build_X(rv_series)
        target = rv_series.values[22:]
        X_const = self._add_constant(X.dropna())
        self.result = self._OLS(target[-len(X_const):], X_const).fit()
        self._last_rv = rv_series
        return self

    def forecast(self, rv_series: Optional[pd.Series] = None) -> float:
        assert self.result is not None, "Call fit() before forecast()"
        if rv_series is None:
            rv_series = self._last_rv
        rv: np.ndarray = rv_series.to_numpy(dtype=float)  # type: ignore[union-attr]
        x_new = np.array([rv[-1], np.mean(rv[-5:]), np.mean(rv[-22:]), 1.0])
        return float(self.result.params @ x_new)


class RandomForestModel:
    """Random Forest with StandardScaler."""

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
        X_scaled = self.scaler.fit_transform(X)
        self.model.fit(X_scaled, y)
        return self

    def forecast(self, X_new: pd.DataFrame) -> float:
        X_scaled = self.scaler.transform(X_new)
        return float(self.model.predict(X_scaled)[0])

    def feature_importance(self, feature_names) -> pd.Series:
        return pd.Series(
            self.model.feature_importances_, index=feature_names
        ).sort_values(ascending=False)


class XGBoostModel:
    """XGBoost with early stopping."""

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
        n_val = max(1, int(len(X) * val_size))
        X_tr, X_val = X.iloc[:-n_val], X.iloc[-n_val:]
        y_tr, y_val = y.iloc[:-n_val], y.iloc[-n_val:]
        X_tr_s = self.scaler.fit_transform(X_tr)
        X_val_s = self.scaler.transform(X_val)
        self.model = xgb.XGBRegressor(**self.params, verbosity=0)
        self.model.fit(X_tr_s, y_tr, eval_set=[(X_val_s, y_val)], verbose=False)
        return self

    def forecast(self, X_new: pd.DataFrame) -> float:
        assert self.model is not None, "Call fit() before forecast()"
        X_scaled = self.scaler.transform(X_new)
        return float(self.model.predict(X_scaled)[0])
