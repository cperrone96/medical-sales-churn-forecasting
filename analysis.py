"""Deterministic temporal churn and revenue-forecasting analysis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 42


def temporal_split(df: pd.DataFrame):
    dates = pd.to_datetime(df["snapshot_date"])
    q1, q2 = dates.quantile([.60, .80])
    return df.loc[dates <= q1].copy(), df.loc[(dates > q1) & (dates <= q2)].copy(), df.loc[dates > q2].copy()


def churn_pipeline(X: pd.DataFrame):
    cat = [c for c in X if X[c].dtype == "object"]
    num = [c for c in X if c not in cat]
    prep = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), num),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), cat),
    ])
    return Pipeline([("preprocess", prep), ("model", LogisticRegression(max_iter=500, class_weight="balanced", random_state=RANDOM_STATE))])


def classification_metrics(y, p):
    return {"prevalence": float(np.mean(y)), "roc_auc": float(roc_auc_score(y, p)), "pr_auc": float(average_precision_score(y, p)), "brier": float(brier_score_loss(y, p))}


def bootstrap_ci(y, p, n=200):
    rng = np.random.default_rng(RANDOM_STATE)
    vals = []
    for _ in range(n):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) > 1:
            vals.append(classification_metrics(y[idx], p[idx]))
    return {k: {"low_2.5": float(np.quantile([v[k] for v in vals], .025)), "high_97.5": float(np.quantile([v[k] for v in vals], .975))} for k in ("roc_auc", "pr_auc", "brier")}


def _forecast_predictions(history: pd.Series, horizon: int, method: str):
    if method == "seasonal_naive":
        if len(history) < 12:
            pred = np.repeat(history.iloc[-1], horizon)
        else:
            pred = np.resize(history.iloc[-12:].to_numpy(), horizon)
        residual = history.iloc[12:].to_numpy() - history.iloc[:-12].to_numpy() if len(history) > 12 else np.diff(history.to_numpy())
    elif method == "drift":
        slope = (history.iloc[-1] - history.iloc[0]) / max(len(history) - 1, 1)
        pred = history.iloc[-1] + slope * np.arange(1, horizon + 1)
        residual = np.diff(history.to_numpy())
    else:
        from statsmodels.tsa.holtwinters import ExponentialSmoothing
        seasonal = 12 if len(history) >= 24 else None
        fit = ExponentialSmoothing(history, trend="add", seasonal="add" if seasonal else None, seasonal_periods=seasonal, initialization_method="estimated").fit(optimized=True)
        pred = fit.forecast(horizon).to_numpy()
        residual = np.asarray(fit.resid)
    spread = float(np.std(residual, ddof=1)) if len(residual) > 1 else 0.0
    return np.asarray(pred), np.asarray(pred) - 1.96 * spread, np.asarray(pred) + 1.96 * spread


def rolling_origin(revenue: pd.Series, origins=(24, 30, 36), horizon=3):
    rows = []
    for origin in origins:
        if origin + horizon > len(revenue):
            continue
        history, actual = revenue.iloc[:origin], revenue.iloc[origin:origin + horizon]
        for method in ("seasonal_naive", "drift", "holt_winters"):
            pred, low, high = _forecast_predictions(history, horizon, method)
            rows.append({"origin": str(revenue.index[origin - 1]), "method": method, "horizon": horizon, "mae": float(mean_absolute_error(actual, pred)), "rmse": float(mean_squared_error(actual, pred) ** .5), "coverage_95": float(np.mean((actual.to_numpy() >= low) & (actual.to_numpy() <= high)))})
    return pd.DataFrame(rows)


def run_analysis(data_dir="data", output_dir="metrics"):
    data_dir, out = Path(data_dir), Path(output_dir); out.mkdir(exist_ok=True)
    accounts = pd.read_csv(data_dir / "accounts.csv")
    required = {"account_id", "snapshot_date", "churn_horizon_days", "churned"}
    missing = required - set(accounts.columns)
    if missing:
        raise ValueError(f"regenerate synthetic data; missing snapshot contract fields: {sorted(missing)}")
    train, val, test = temporal_split(accounts)
    excluded = ["account_id", "snapshot_date", "churn_horizon_days", "churned"]
    features = [c for c in accounts if c not in excluded]
    model = churn_pipeline(train[features]); model.fit(train[features], train["churned"])
    p_val, p_test = model.predict_proba(val[features])[:, 1], model.predict_proba(test[features])[:, 1]
    baseline_val = np.repeat(train.churned.mean(), len(val)); baseline_test = np.repeat(train.churned.mean(), len(test))
    classification = {"baseline_prevalence": classification_metrics(test.churned.to_numpy(), baseline_test), "logistic_regression": classification_metrics(test.churned.to_numpy(), p_test), "validation": {"baseline_prevalence": classification_metrics(val.churned.to_numpy(), baseline_val), "logistic_regression": classification_metrics(val.churned.to_numpy(), p_val)}}
    calibration_p, calibration_y = calibration_curve(test.churned, p_test, n_bins=10, strategy="quantile")
    pd.DataFrame({"predicted_probability": calibration_p, "observed_fraction": calibration_y}).to_csv(out / "churn_calibration.csv", index=False)
    # A top-decile policy is reported descriptively, not as a guaranteed lift.
    n = max(1, len(test) // 10); top = np.argsort(-p_test)[:n]
    ranking = {"top_decile_precision": float(test.churned.iloc[top].mean()), "base_rate": float(test.churned.mean()), "top_decile_capture": float(test.churned.iloc[top].sum() / max(test.churned.sum(), 1))}
    coef = np.abs(model.named_steps["model"].coef_[0]); names = model.named_steps["preprocess"].get_feature_names_out()
    pd.DataFrame({"encoded_feature": names, "coefficient_abs": coef}).sort_values("coefficient_abs", ascending=False).to_csv(out / "churn_logistic_coefficients.csv", index=False)
    rev = pd.read_csv(data_dir / "monthly_revenue.csv", parse_dates=["month"]).set_index("month")["revenue"].sort_index()
    holdout = 6; train_rev, test_rev = rev.iloc[:-holdout], rev.iloc[-holdout:]
    forecasts = []
    for method in ("seasonal_naive", "drift", "holt_winters"):
        pred, low, high = _forecast_predictions(train_rev, holdout, method)
        forecasts.append({"method": method, "horizon": holdout, "mae": float(mean_absolute_error(test_rev, pred)), "rmse": float(mean_squared_error(test_rev, pred) ** .5), "mape": float(np.mean(np.abs((test_rev.to_numpy() - pred) / test_rev.to_numpy()))), "coverage_95": float(np.mean((test_rev.to_numpy() >= low) & (test_rev.to_numpy() <= high)))})
    rolling = rolling_origin(rev)
    rolling.to_csv(out / "rolling_origin_forecasts.csv", index=False)
    summary = {"data_contract": {"snapshot_date": "account state date", "observation_window": "features describe behavior available by snapshot_date", "future_horizon_days": 90, "synthetic": True}, "temporal_split": {"train_end": str(train.snapshot_date.max()), "validation_end": str(val.snapshot_date.max()), "test_start": str(test.snapshot_date.min()), "rows": [len(train), len(val), len(test)]}, "classification": classification, "uncertainty_95pct": bootstrap_ci(test.churned.to_numpy(), p_test), "ranking": ranking, "forecast_holdout": forecasts, "forecast_rolling_origin": rolling.to_dict(orient="records"), "interpretation": "selected logistic coefficients are predictive associations; synthetic recovery does not establish real-account validity or causal retention impact"}
    (out / "sales_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    run_analysis()
