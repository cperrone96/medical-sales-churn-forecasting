"""Generate the synthetic B2B medical-device sales data used in this project.

Two tables are produced with a fixed seed, so every run writes identical files:

* ``data/accounts.csv``: 1,400 account snapshots. Features describe the
  account on ``snapshot_date``; ``churned`` records whether the account was
  lost in the following 90 days. Churn is drawn from a known logistic model
  (``CHURN_INTERCEPT`` and ``CHURN_COEFFICIENTS``), so the analysis can compare
  fitted models against the true data-generating probability.
* ``data/monthly_revenue.csv``: 42 months of company revenue with compounding
  growth, annual seasonality (peak in June, trough in December) and
  multiplicative noise.

Run:  python generate_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_ACCOUNTS = 1400
FIRST_SNAPSHOT = "2021-01-01"
CHURN_HORIZON_DAYS = 90

REGIONS = ["Northeast", "Southeast", "Midwest", "West", "Southwest"]
SPECIALTIES = ["Cardiology", "Orthopedics", "Oncology", "Neurology", "Dermatology", "General Surgery"]
ACCOUNT_TYPES = ["Hospital", "Surgery Center", "Clinic", "Private Practice"]
ACCOUNT_TYPE_WEIGHTS = [0.22, 0.28, 0.30, 0.20]

# True churn model: logit(p) = intercept + sum(coef * (feature - center)).
CHURN_INTERCEPT = -1.4
CHURN_COEFFICIENTS: dict[str, tuple[float, float]] = {
    # feature: (coefficient, center)
    "days_since_last_order": (0.020, 60),
    "engagement_score": (-0.030, 58),
    "support_tickets_qtr": (0.180, 2),
    "rep_tenure_months": (-0.045, 22),
    "avg_discount_pct": (2.10, 0.12),
    "order_frequency_yr": (-0.055, 9),
    "onboarding_score": (-0.28, 3.6),
}


def true_churn_probability(accounts: pd.DataFrame) -> np.ndarray:
    """Probability of churn under the generator's logistic model."""
    z = CHURN_INTERCEPT + sum(coef * (accounts[col].to_numpy() - center) for col, (coef, center) in CHURN_COEFFICIENTS.items())
    return 1 / (1 + np.exp(-z))


def generate_accounts(rng: np.random.Generator, n: int = N_ACCOUNTS) -> pd.DataFrame:
    """Account snapshots, one per day, with a 90-day churn outcome."""
    region = rng.choice(REGIONS, n)
    specialty = rng.choice(SPECIALTIES, n)
    account_type = rng.choice(ACCOUNT_TYPES, n, p=ACCOUNT_TYPE_WEIGHTS)
    rep_tenure = np.clip(rng.normal(22, 12, n), 1, 60).round()  # months the rep has covered the account
    onboarding = np.clip(rng.normal(3.6, 0.9, n), 1, 5).round(1)  # onboarding quality, 1 to 5
    product_lines = rng.integers(1, 6, n)
    avg_discount = np.clip(rng.normal(0.12, 0.06, n), 0, 0.4).round(3)
    support_tickets = rng.poisson(2.2, n)
    engagement = np.clip(rng.normal(58, 18, n) + rep_tenure * 0.3, 0, 100).round()  # meetings and email opens, 0 to 100
    order_frequency = np.clip(rng.normal(9, 4, n) + product_lines * 1.2, 1, None).round()
    avg_order_value = np.clip(rng.normal(4200, 1500, n) + product_lines * 500, 500, None).round(-1)
    recency_days = np.clip(rng.exponential(45, n) + (100 - engagement) * 0.6, 1, 400).round()

    accounts = pd.DataFrame(
        {
            "account_id": [f"ACCT{1000 + i}" for i in range(n)],
            "snapshot_date": pd.date_range(FIRST_SNAPSHOT, periods=n, freq="D"),
            "region": region,
            "specialty": specialty,
            "account_type": account_type,
            "rep_tenure_months": rep_tenure,
            "onboarding_score": onboarding,
            "product_lines": product_lines,
            "avg_discount_pct": avg_discount,
            "support_tickets_qtr": support_tickets,
            "engagement_score": engagement,
            "order_frequency_yr": order_frequency,
            "avg_order_value": avg_order_value,
            "annual_revenue": (order_frequency * avg_order_value).round(-1),
            "days_since_last_order": recency_days,
            "churn_horizon_days": CHURN_HORIZON_DAYS,
        }
    )
    accounts["churned"] = (rng.random(n) < true_churn_probability(accounts)).astype(int)
    return accounts


def generate_revenue(rng: np.random.Generator) -> pd.DataFrame:
    """Monthly revenue: 4.5% compounding growth x seasonality x 5% noise."""
    months = pd.date_range("2022-01-01", "2025-06-01", freq="MS")
    t = np.arange(len(months))
    trend = 150_000 * 1.045**t
    season = 1 + 0.14 * np.sin(2 * np.pi * (months.month - 3) / 12)
    noise = rng.normal(1, 0.05, len(months))
    return pd.DataFrame({"month": months.strftime("%Y-%m-01"), "revenue": (trend * season * noise).round(-2)})


def main(out_dir: Path = Path(__file__).resolve().parent / "data") -> None:
    rng = np.random.default_rng(SEED)
    out_dir.mkdir(parents=True, exist_ok=True)
    accounts = generate_accounts(rng)
    revenue = generate_revenue(rng)
    accounts.to_csv(out_dir / "accounts.csv", index=False)
    revenue.to_csv(out_dir / "monthly_revenue.csv", index=False)
    print(f"accounts.csv: {len(accounts)} rows, churn rate {accounts['churned'].mean():.1%}")
    print(f"monthly_revenue.csv: {len(revenue)} months, ${revenue['revenue'].iloc[0]:,.0f} to ${revenue['revenue'].iloc[-1]:,.0f}")


if __name__ == "__main__":
    main()
