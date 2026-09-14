# Medical-device sales: churn prediction and revenue forecasting

This repository is a reproducible exercise on **synthetic** medical-device
commercial data. It demonstrates two bounded questions: (1) can account
snapshots be ranked by whether they churn in the following 90 days, and (2) can
monthly company revenue be forecast for planning? It is not evidence about any
real account, retention program, or causal intervention.

## Data and validation

`generate_data.py` uses NumPy seed 42. Each account row is a snapshot: all
features describe behavior available on `snapshot_date`, and `churned` is the
label in the next 90 days. The classification model uses chronological
train/validation/test partitions, a prevalence baseline, balanced logistic
regression, Brier score, quantile calibration, bootstrap intervals, top-decile
ranking diagnostics, and standardized logistic coefficients. No feature is
allowed to use the future label.

Revenue evaluation uses a six-month untouched holdout plus rolling-origin
backtesting. Seasonal-naive and drift are reported alongside Holt–Winters;
each forecast includes a simple residual-based 95% interval and horizon error.
Intervals are descriptive under this synthetic generator, not guarantees.

## Run

```bash
python -m pip install -r requirements.txt
python generate_data.py
python analysis.py
python -m pytest -q
```

Outputs are written to `metrics/`, including `sales_metrics.json`, calibration,
coefficient, and rolling-origin tables. Synthetic recovery does not establish
real-account validity, transportability, or causal retention impact. A real
deployment would need account-level snapshot lineage, leakage review, temporal
drift monitoring, calibration, and business/legal governance.
