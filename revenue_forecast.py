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
* ``ensemble_log``: equal-weight average of the four log-scale models on the
  log scale (geometric mean). Its interval averages the members' bounds the
  same way, which is a pragmatic approximation rather than an exact interval.

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
# Equal-weight ensemble of every log-scale model. Its members are fixed in advance,
# not chosen from backtest results, so it competes on the same footing as the others.
ENSEMBLE = "ensemble_log"
ENSEMBLE_MEMBERS = ("ets_log", "ets_damped_log", "sarima_log", "trend_fourier_log")
CANDIDATES = (*METHODS, ENSEMBLE)


def combine(forecasts: Sequence[Forecast]) -> Forecast:
    """Equal-weight combination on the log scale (geometric mean of forecasts and of interval bounds)."""

    def geo(values: list[np.ndarray]) -> np.ndarray:
        return np.exp(np.mean(np.log(np.vstack(values)), axis=0))

    return Forecast(
        geo([f.mean for f in forecasts]),
        geo([f.lower for f in forecasts]),
        geo([f.upper for f in forecasts]),
        all(f.converged for f in forecasts),
        sorted({n for f in forecasts for n in f.notes}),
    )


def forecast(history: pd.Series, horizon: int, method: str) -> Forecast:
    """Fit ``method`` on ``history`` and forecast ``horizon`` steps.

    Optimizer warnings are captured into ``notes`` instead of printed; the
    ``converged`` flag comes from the optimizer itself.
    """
    if method == ENSEMBLE:
        return combine([forecast(history, horizon, m) for m in ENSEMBLE_MEMBERS])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        result = METHODS[method](history, horizon)
    result.notes = sorted({str(w.message).split(".")[0] for w in caught})
    return result


def per_origin_mape(predictions: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute percentage error for each (origin, method): one score per backtest origin."""
    ape = (predictions.actual - predictions.forecast).abs() / predictions.actual
    return predictions.assign(ape=ape).pivot_table(index="origin", columns="method", values="ape", aggfunc="mean")


def select_method(backtest: pd.DataFrame) -> dict:
    """Pre-specified selection rule: the converged candidate with the lowest backtest MAPE.

    MAPE is averaged over every origin and horizon step (each origin contributes
    the same number of forecasts, so this equals the mean of per-origin MAPEs).
    For context, each candidate also gets its spread across origins and a paired
    comparison with the selected method: the mean per-origin difference and its
    standard error. A candidate within one paired standard error of the selected
    one is flagged as not distinguishable from it on the backtest.
    """
    converged = backtest.groupby("method").converged.all()
    scores = per_origin_mape(backtest)[[m for m in converged.index if converged[m]]]
    selected = str(scores.mean().idxmin())
    n = len(scores)
    comparisons = {}
    for m in scores.columns:
        diff = scores[m] - scores[selected]
        se = float(diff.std(ddof=1) / np.sqrt(n)) if m != selected else 0.0
        comparisons[m] = {
            "mean_mape": float(scores[m].mean()),
            "sd_mape_across_origins": float(scores[m].std(ddof=1)),
            "min_origin_mape": float(scores[m].min()),
            "max_origin_mape": float(scores[m].max()),
            "paired_difference_vs_selected": float(diff.mean()),
            "paired_difference_se": se,
            "within_one_paired_se": bool(m == selected or diff.mean() <= se),
        }
    return {"selected": selected, "n_origins": n, "comparisons": comparisons, "per_origin_mape": scores}


def rolling_origin(
    series: pd.Series, origins: Sequence[int], horizon: int, methods: Sequence[str] = CANDIDATES
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
    selection = select_method(backtest)
    selected = selection["selected"]
    backtest_scores = score(backtest)

    # The holdout is scored once, for every candidate, after selection is fixed.
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
            "selection_rule": "lowest MAPE averaged over all backtest origins and horizon steps, among candidates whose fits all converged; the holdout is not used for selection",
        },
        "backtest": backtest_scores.to_dict(orient="records"),
        "selection": {k: v for k, v in selection.items() if k != "per_origin_mape"},
        "selected_method": selected,
        "holdout": holdout_scores.to_dict(orient="records"),
        "holdout_selected_vs_best_baseline": {
            "selected_mape": float(sel_holdout["mape"]),
            "best_baseline": str(best_baseline["method"]),
            "best_baseline_mape": float(best_baseline["mape"]),
        },
        "forward_forecast_converged": future.converged,
    }
    by_origin = selection["per_origin_mape"].reset_index()
    return {
        "summary": summary,
        "backtest": backtest,
        "backtest_by_origin": by_origin,
        "holdout": holdout_predictions,
        "forward": forward,
    }
