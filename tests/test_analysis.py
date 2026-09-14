import numpy as np
import pandas as pd

from analysis import rolling_origin, temporal_split


def test_temporal_split_has_no_date_overlap():
    d = pd.DataFrame({"snapshot_date": pd.date_range("2023-01-01", periods=20), "churned": [0, 1] * 10})
    a, b, c = temporal_split(d)
    assert a.snapshot_date.max() < b.snapshot_date.min() < c.snapshot_date.min()


def test_rolling_origin_reports_baselines_and_intervals():
    idx = pd.date_range("2022-01-01", periods=42, freq="MS")
    y = pd.Series(np.arange(42, dtype=float), index=idx)
    out = rolling_origin(y, origins=(24, 30), horizon=3)
    assert set(out.method) == {"seasonal_naive", "drift", "holt_winters"}
    assert out.coverage_95.between(0, 1).all()


def test_snapshot_features_are_explicitly_pre_target():
    d = pd.read_csv("data/accounts.csv")
    assert pd.to_datetime(d.snapshot_date).is_monotonic_increasing
    assert d.churn_horizon_days.eq(90).all()
    assert "churned" in d and "snapshot_date" in d
