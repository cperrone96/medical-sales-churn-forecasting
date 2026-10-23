import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from churn_model import feature_columns, temporal_split, top_decile
from generate_data import SEED, generate_accounts, generate_revenue, true_churn_probability
from revenue_forecast import forecast, rolling_origin, score

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def accounts():
    return pd.read_csv(ROOT / "data" / "accounts.csv", parse_dates=["snapshot_date"])


def test_committed_data_matches_the_generator(accounts):
    rng = np.random.default_rng(SEED)
    regenerated = generate_accounts(rng)
    revenue = generate_revenue(rng)
    pd.testing.assert_frame_equal(regenerated, accounts, check_dtype=False)
    pd.testing.assert_frame_equal(revenue, pd.read_csv(ROOT / "data" / "monthly_revenue.csv"), check_dtype=False)


def test_temporal_split_is_ordered_and_embargoed(accounts):
    train, val, test = temporal_split(accounts)
    assert train.snapshot_date.max() + pd.Timedelta(days=90) < val.snapshot_date.min()
    assert val.snapshot_date.max() < test.snapshot_date.min()
    assert set(train.account_id).isdisjoint(val.account_id) and set(val.account_id).isdisjoint(test.account_id)


def test_features_exclude_identifiers_dates_and_outcome(accounts):
    assert not {"account_id", "snapshot_date", "churned", "churn_horizon_days"} & set(feature_columns(accounts))


def test_true_probability_is_consistent_with_observed_churn(accounts):
    p = true_churn_probability(accounts)
    assert abs(p.mean() - accounts.churned.mean()) < 0.03


def test_top_decile_lift():
    y = np.array([1] * 10 + [0] * 90)
    p = np.r_[np.ones(10), np.zeros(90)]
    assert top_decile(y, p) == {"precision": 1.0, "capture": 1.0, "lift": 10.0}


def _monthly(values):
    return pd.Series(values, index=pd.date_range("2020-01-01", periods=len(values), freq="MS"))


def test_baselines_are_exact_on_their_own_structure():
    seasonal = _monthly(np.tile(np.arange(12.0) + 1, 3) + np.r_[np.zeros(35), 0.1])
    assert np.allclose(forecast(seasonal.iloc[:24], 6, "seasonal_naive").mean, np.arange(6.0) + 1)
    linear = _monthly(10 + 2 * np.arange(30.0) + np.tile([0.1, -0.1], 15))
    drift = forecast(linear, 3, "drift")
    assert np.allclose(drift.mean, linear.iloc[-1] + 2 * np.arange(1, 4), atol=0.05)
    widths = drift.upper - drift.lower
    assert np.all(np.diff(widths) > 0)  # intervals widen with the horizon


def test_log_models_recover_multiplicative_growth_and_converge():
    rng = np.random.default_rng(0)
    t = np.arange(48)
    series = _monthly(1e5 * 1.03**t * (1 + 0.1 * np.sin(2 * np.pi * t / 12)) * rng.normal(1, 0.01, 48))
    truth = 1e5 * 1.03 ** np.arange(42, 48) * (1 + 0.1 * np.sin(2 * np.pi * np.arange(42, 48) / 12))
    for method in ("ets_log", "trend_fourier_log", "sarima_log"):
        fc = forecast(series.iloc[:42], 6, method)
        assert fc.converged, method
        assert np.mean(np.abs(fc.mean - truth) / truth) < 0.03, method
        assert np.all(fc.lower < fc.mean) and np.all(fc.mean < fc.upper)


def test_rolling_origin_never_uses_future_values():
    rng = np.random.default_rng(1)
    series = _monthly(1e5 * 1.02 ** np.arange(36) * rng.normal(1, 0.02, 36))
    tampered = series.copy()
    tampered.iloc[30:] *= 10
    a = rolling_origin(series, [30], 6, methods=("drift", "ets_log"))
    b = rolling_origin(tampered, [30], 6, methods=("drift", "ets_log"))
    assert np.allclose(a.forecast, b.forecast)
    assert (score(a).n_forecasts == 6).all()


def test_committed_results_follow_the_selection_protocol():
    summary = json.loads((ROOT / "metrics" / "sales_metrics.json").read_text())
    churn = summary["churn"]
    assert churn["selected_model"] == min(churn["validation"], key=lambda m: churn["validation"][m]["log_loss"])
    backtest = pd.DataFrame(summary["revenue"]["backtest"])
    eligible = backtest.loc[backtest.all_fits_converged]
    assert summary["revenue"]["selected_method"] == eligible.sort_values("mape").iloc[0]["method"]
