"""
Generate SYNTHETIC medical-device sales data for this portfolio project.

None of this is real company data. It is fabricated with a fixed random seed so
the analysis is fully reproducible, while realistically mirroring a B2B medical-
device sales operation (accounts across regions/specialties, ordering behavior,
support load, and account churn; plus a company-level monthly revenue series).

Run:  python generate_data.py
Writes: data/accounts.csv  and  data/monthly_revenue.csv
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 1400

REGIONS = ["Northeast", "Southeast", "Midwest", "West", "Southwest"]
SPECIALTIES = ["Cardiology", "Orthopedics", "Oncology", "Neurology",
               "Dermatology", "General Surgery"]
ACCT_TYPES = ["Hospital", "Surgery Center", "Clinic", "Private Practice"]
TYPE_W = [0.22, 0.28, 0.30, 0.20]

# ---------------------------------------------------------------------------
# Account-level table (features + churn label)
# ---------------------------------------------------------------------------
region = RNG.choice(REGIONS, N)
specialty = RNG.choice(SPECIALTIES, N)
acct_type = RNG.choice(ACCT_TYPES, N, p=TYPE_W)

# tenure of the assigned sales rep on the account (months)
rep_tenure = np.clip(RNG.normal(22, 12, N), 1, 60).round()
# onboarding quality score 1-5
onboarding = np.clip(RNG.normal(3.6, 0.9, N), 1, 5).round(1)
# product lines purchased (1-5)
product_lines = RNG.integers(1, 6, N)
# average discount granted (price sensitivity signal)
avg_discount = np.clip(RNG.normal(0.12, 0.06, N), 0, 0.4).round(3)
# support tickets per quarter (friction signal)
support_tickets = RNG.poisson(2.2, N)
# engagement score: meetings + email opens, 0-100
engagement = np.clip(RNG.normal(58, 18, N) + rep_tenure * 0.3, 0, 100).round()

# ordering behavior (RFM)
base_orders = np.clip(RNG.normal(9, 4, N) + product_lines * 1.2, 1, None)
order_frequency = base_orders.round()                       # orders in last year
avg_order_value = np.clip(RNG.normal(4200, 1500, N)
                          + product_lines * 500, 500, None).round(-1)
monetary = (order_frequency * avg_order_value).round(-1)     # annual revenue
recency_days = np.clip(RNG.exponential(45, N)
                       + (100 - engagement) * 0.6, 1, 400).round()

# ---------------------------------------------------------------------------
# Churn: latent logistic risk driven by realistic drivers, then sampled
# ---------------------------------------------------------------------------
z = (
    -1.4
    + 0.020 * (recency_days - 60)          # longer since last order -> churn
    - 0.030 * (engagement - 58)            # low engagement -> churn
    + 0.180 * (support_tickets - 2)        # friction -> churn
    - 0.045 * (rep_tenure - 22)            # new rep relationship -> churn
    + 2.10 * (avg_discount - 0.12)         # discount-dependent -> churn
    - 0.055 * (order_frequency - 9)        # infrequent buyers -> churn
    - 0.28 * (onboarding - 3.6)            # poor onboarding -> churn
)
churn_prob = 1 / (1 + np.exp(-z))
churned = (RNG.random(N) < churn_prob).astype(int)

accounts = pd.DataFrame({
    "account_id": [f"ACCT{1000+i}" for i in range(N)],
    # Each row is a synthetic account snapshot.  Features describe the state
    # at snapshot_date; churned is the outcome in the following 90 days.
    "snapshot_date": pd.date_range("2023-01-01", periods=N, freq="D"),
    "region": region,
    "specialty": specialty,
    "account_type": acct_type,
    "rep_tenure_months": rep_tenure,
    "onboarding_score": onboarding,
    "product_lines": product_lines,
    "avg_discount_pct": avg_discount,
    "support_tickets_qtr": support_tickets,
    "engagement_score": engagement,
    "order_frequency_yr": order_frequency,
    "avg_order_value": avg_order_value,
    "annual_revenue": monetary,
    "days_since_last_order": recency_days,
    "churn_horizon_days": 90,
    "churned": churned,
})
accounts.to_csv("data/accounts.csv", index=False)
print(f"accounts.csv: {len(accounts)} rows | churn rate {churned.mean():.1%}")

# ---------------------------------------------------------------------------
# Company monthly revenue series (trend + seasonality + noise), 2022-01..2025-06
# mirrors a fast-growth arc similar to a real scale-up
# ---------------------------------------------------------------------------
months = pd.date_range("2022-01-01", "2025-06-01", freq="MS")
t = np.arange(len(months))
trend = 150_000 * (1.045 ** t)                     # ~4.5% compounding monthly
season = 1 + 0.14 * np.sin(2 * np.pi * (months.month - 3) / 12)  # Q4/spring lift
noise = RNG.normal(1, 0.05, len(months))
revenue = (trend * season * noise).round(-2)

rev = pd.DataFrame({"month": months.strftime("%Y-%m-01"), "revenue": revenue})
rev.to_csv("data/monthly_revenue.csv", index=False)
print(f"monthly_revenue.csv: {len(rev)} months | "
      f"${revenue[0]:,.0f} -> ${revenue[-1]:,.0f}")
