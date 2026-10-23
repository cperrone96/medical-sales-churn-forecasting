"""Monthly revenue forecasting with rolling-origin model selection.

Revenue grows multiplicatively (a percentage per month) with a seasonal
swing proportional to its level. On the log scale trend and seasonality
become additive, so the structured models here work on log revenue (additive
Holt-Winters on the raw series does not converge on this data).
Forecasts are back-transformed with ``exp``, which gives the median forecast.

Candidates:

* ``seasonal_naive``: same month last year.
* ``drift``: last value plus the average historical monthly change.
* ``ets_log``: ETS(A,A,A) on log revenue (additive error, trend and season).
* ``ets_damped_log``: the same with a damped trend.
* ``sarima_log``: SARIMA(0,1,1)(0,1,1)12, the "airline" model, on log revenue.
* ``trend_fourier_log``: OLS on log revenue with a linear trend and two
  Fourier harmonic pairs for annual seasonality.

All intervals are 95% prediction intervals for each horizon.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Callable, Sequence

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.statespace.sarimax import SARIMAX

SEASON = 12
Z95 = 1.959963984540054


@dataclass
class Forecast:
    mean: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    converged: bool = True
    notes: list[str] = field(default_factory=list)


def _seasonal_naive(history: pd.Series, horizon: int) -> Forecast:
    values = history.to_numpy()
    mean = np.resize(values[-SEASON:], horizon)
    sigma = np.std(values[SEASON:] - values[:-SEASON], ddof=1)
    k = np.arange(horizon) // SEASON  # completed seasons before each step
    half = Z95 * sigma * np.sqrt(k + 1)
    return Forecast(mean, mean - half, mean + half)


def _drift(history: pd.Series, horizon: int) -> Forecast:
    values = history.to_numpy()
    n = len(values)
    slope = (values[-1] - values[0]) / (n - 1)
    h = np.arange(1, horizon + 1)
    mean = values[-1] + slope * h
    sigma = np.std(np.diff(values) - slope, ddof=1)
    half = Z95 * sigma * np.sqrt(h * (1 + h / (n - 1)))
    return Forecast(mean, mean - half, mean + half)


def _ets_log(history: pd.Series, horizon: int, damped: bool = False) -> Forecast:
    model = ETSModel(
        np.log(history), error="add", trend="add", damped_trend=damped, seasonal="add", seasonal_periods=SEASON
    )
    fit = model.fit(disp=False)
    frame = fit.get_prediction(start=len(history), end=len(history) + horizon - 1).summary_frame(alpha=0.05)
    return Forecast(
        np.exp(frame["mean"].to_numpy()),
        np.exp(frame["pi_lower"].to_numpy()),
        np.exp(frame["pi_upper"].to_numpy()),
        bool(fit.mle_retvals.get("converged", False)),
    )


def _sarima_log(history: pd.Series, horizon: int) -> Forecast:
    fit = SARIMAX(np.log(history), order=(0, 1, 1), seasonal_order=(0, 1, 1, SEASON)).fit(disp=False)
    pred = fit.get_forecast(horizon)
    ci = pred.conf_int(alpha=0.05).to_numpy()
    return Forecast(
        np.exp(pred.predicted_mean.to_numpy()),
        np.exp(ci[:, 0]),
        np.exp(ci[:, 1]),
        bool(fit.mle_retvals.get("converged", False)),
    )


def _fourier_design(t: np.ndarray, months: np.ndarray, harmonics: int = 2) -> np.ndarray:
    columns = [np.ones_like(t, dtype=float), t.astype(float)]
    for k in range(1, harmonics + 1):
        columns += [np.sin(2 * np.pi * k * months / SEASON), np.cos(2 * np.pi * k * months / SEASON)]
    return np.column_stack(columns)


def _trend_fourier_log(history: pd.Series, horizon: int) -> Forecast:
    n = len(history)
    months = pd.date_range(history.index[0], periods=n + horizon, freq="MS").month.to_numpy()
    X = _fourier_design(np.arange(n + horizon), months)
    fit = sm.OLS(np.log(history.to_numpy()), X[:n]).fit()
    frame = fit.get_prediction(X[n:]).summary_frame(alpha=0.05)
    return Forecast(
        np.exp(frame["mean"].to_numpy()),
        np.exp(frame["obs_ci_lower"].to_numpy()),
        np.exp(frame["obs_ci_upper"].to_numpy()),
    )


METHODS: dict[str, Callable[[pd.Series, int], Forecast]] = {
    "seasonal_naive": _seasonal_naive,
    "drift": _drift,
    "ets_log": _ets_log,
    "ets_damped_log": lambda history, horizon: _ets_log(history, horizon, damped=True),
    "sarima_log": _sarima_log,
    "trend_fourier_log": _trend_fourier_log,
}
BASELINES = ("seasonal_naive", "drift")


def forecast(history: pd.Series, horizon: int, method: str) -> Forecast:
    """Fit ``method`` on ``history`` and forecast ``horizon`` steps.

    Optimizer warnings are captured into ``notes`` instead of printed; the
    ``converged`` flag comes from the optimizer itself.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = METHODS[method](history, horizon)
    result.notes = sorted({str(w.message).split(".")[0] for w in caught})
    return result


def rolling_origin(
    series: pd.Series, origins: Sequence[int], horizon: int, methods: Sequence[str] = tuple(METHODS)
) -> pd.DataFrame:
    """Forecasts from each origin (number of months of history), one row per step."""
    rows = []
    for origin in origins:
        history, actual = series.iloc[:origin], series.iloc[origin : origin + horizon]
        if len(actual) < horizon:
            raise ValueError(f"origin {origin} leaves fewer than {horizon} months to score")
        for method in methods:
            fc = forecast(history, horizon, method)
            for step in range(horizon):
                rows.append(
                    {
                        "origin": history.index[-1].strftime("%Y-%m"),
                        "method": method,
                        "step": step + 1,
                        "month": actual.index[step].strftime("%Y-%m"),
                        "actual": float(actual.iloc[step]),
                        "forecast": float(fc.mean[step]),
                        "lower_95": float(fc.lower[step]),
                        "upper_95": float(fc.upper[step]),
                        "converged": fc.converged,
                        "notes": "; ".join(fc.notes),
                    }
                )
    return pd.DataFrame(rows)


def score(predictions: pd.DataFrame) -> pd.DataFrame:
    """MAPE, RMSE and interval coverage per method; a method counts as valid only if every fit converged."""
    frame = predictions.assign(
        ape=lambda d: (d.actual - d.forecast).abs() / d.actual,
        sq=lambda d: (d.actual - d.forecast) ** 2,
        covered=lambda d: (d.actual >= d.lower_95) & (d.actual <= d.upper_95),
    )
    out = frame.groupby("method", sort=False).agg(
        mape=("ape", "mean"),
        rmse=("sq", lambda s: float(np.sqrt(s.mean()))),
        coverage_95=("covered", "mean"),
        all_fits_converged=("converged", "all"),
        n_forecasts=("ape", "size"),
    )
    return out.reset_index()


def run_revenue(revenue: pd.Series, holdout: int = 6, horizon: int = 6, first_origin: int = 24) -> dict:
    """Select by rolling-origin backtest, confirm on an untouched holdout, then forecast forward."""
    revenue = revenue.asfreq("MS")
    development = revenue.iloc[:-holdout]
    origins = list(range(first_origin, len(development) - horizon + 1))
    backtest = rolling_origin(development, origins, horizon)
    backtest_scores = score(backtest)
    eligible = backtest_scores.loc[backtest_scores.all_fits_converged]
    selected = str(eligible.sort_values("mape").iloc[0]["method"])

    holdout_predictions = rolling_origin(revenue, [len(development)], holdout)
    holdout_scores = score(holdout_predictions)

    future = forecast(revenue, horizon, selected)
    future_index = pd.date_range(revenue.index[-1] + pd.offsets.MonthBegin(1), periods=horizon, freq="MS")
    forward = pd.DataFrame(
        {
            "month": future_index.strftime("%Y-%m"),
            "forecast": future.mean,
            "lower_95": future.lower,
            "upper_95": future.upper,
        }
    )

    best_baseline = holdout_scores.loc[holdout_scores.method.isin(BASELINES)].sort_values("mape").iloc[0]
    sel_holdout = holdout_scores.set_index("method").loc[selected]
    summary = {
        "series": {
            "months": len(revenue),
            "first_month": revenue.index[0].strftime("%Y-%m"),
            "last_month": revenue.index[-1].strftime("%Y-%m"),
        },
        "design": {
            "holdout_months": holdout,
            "backtest_window": [development.index[0].strftime("%Y-%m"), development.index[-1].strftime("%Y-%m")],
            "backtest_origins": len(origins),
            "horizon_months": horizon,
            "selection_metric": "mean MAPE across all backtest forecasts, among methods whose fits all converged",
        },
        "backtest": backtest_scores.to_dict(orient="records"),
        "selected_method": selected,
        "holdout": holdout_scores.to_dict(orient="records"),
        "holdout_selected_vs_best_baseline": {
            "selected_mape": float(sel_holdout["mape"]),
            "best_baseline": str(best_baseline["method"]),
            "best_baseline_mape": float(best_baseline["mape"]),
        },
        "forward_forecast_converged": future.converged,
    }
    return {"summary": summary, "backtest": backtest, "holdout": holdout_predictions, "forward": forward}
