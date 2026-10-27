# Medical-Device Sales: Churn Prediction and Revenue Forecasting

Two planning questions for a B2B medical-device business: which accounts are most likely to be lost in the next 90 days, so account managers know where to spend their time, and what monthly revenue to plan for over the next six months. Both are answered on synthetic data with time-ordered validation that mirrors how the models would be used.

## Key results

**Churn** (test period: 280 account snapshots, churn rate 22.5%; 95% bootstrap intervals, 2,000 resamples)

| Model | ROC-AUC (95% CI) | PR-AUC | Brier score | Top-decile precision |
|---|---|---|---|---|
| Prevalence baseline | 0.500 | 0.225 | 0.174 | n/a |
| **Logistic regression (selected)** | **0.790 (0.723 to 0.851)** | **0.577** | **0.136** | **64% (2.9x base rate)** |
| Gradient boosting | 0.730 (0.657 to 0.799) | 0.483 | 0.156 | 50% |
| True generator probability (ceiling) | 0.792 | 0.579 | 0.136 | 64% |

- Logistic regression was selected on validation log loss. On test it essentially matches the true data-generating probability and beats gradient boosting by 0.060 ROC-AUC (paired 95% CI 0.026 to 0.096).
- All seven true churn drivers are recovered with the correct sign. ROC curves and the coefficient comparison are in [figures/roc_curves.png](figures/roc_curves.png).

**Retention value by outreach capacity** (test period, model's top-k list)

> **Assumptions, stated rather than estimated:** outreach retains **30%** of accounts that would otherwise churn and costs **$2,500 per account contacted**. Revenue protected = save rate x annual revenue (from the data) of the churners on the list, so it's one year of revenue with no margin applied. A random list of the same size is the comparison.

| Accounts contacted | Churners reached | Revenue protected (95% CI) | Random list of same size | Outreach cost | Revenue per outreach $ |
|---|---|---|---|---|---|
| 14 (top 5%) | 11 | $220k ($132k to $332k) | $71k | $35k | 6.3 |
| **28 (top 10%)** | **18** | **$378k ($252k to $548k)** | **$143k** | **$70k** | **5.4** |
| 56 (top 20%) | 31 | $639k ($462k to $875k) | $285k | $140k | 4.6 |
| 84 (top 30%) | 39 | $878k ($646k to $1.15M) | $428k | $210k | 4.2 |
| 140 (top 50%) | 50 | $1.10M ($834k to $1.43M) | $713k | $350k | 3.2 |

- The ranked list protects 1.5 to 3.1 times what a random list of the same size would. Returns fall as capacity grows: going from 84 to 140 accounts adds $227k of revenue for $140k of outreach.
- The model's own expected value for each list (sum of predicted churn probability x revenue x save rate) lands within 5% of the realized figure at every capacity, which is what good calibration buys.
- A 3 x 3 grid of save rates (15%, 30%, 45%) and costs ($1,500, $2,500, $4,000) is in [metrics/retention_sensitivity.csv](metrics/retention_sensitivity.csv). The top 30% list stays net positive in every cell; the top 50% list goes slightly negative at a 15% save rate and $4,000 cost.

**Revenue** (backtest: 7 origins x 6-month horizon inside January 2022 to December 2024; holdout: January to June 2025, scored once after selection)

| Method | Backtest MAPE, mean | Spread across origins (SD; min to max) | Paired gap vs selected (± 1 SE) | Holdout MAPE | Holdout 95% PI coverage |
|---|---|---|---|---|---|
| **Trend + Fourier regression on log (selected)** | **6.5%** | **2.0 pts; 3.8% to 8.7%** | **n/a** | **6.2%** | **5 of 6** |
| Equal-weight ensemble of the 4 log models | 6.7% | 1.3 pts; 5.2% to 8.1% | +0.2 ± 0.4 pts | 5.4% | 5 of 6 |
| SARIMA(0,1,1)(0,1,1)12 on log | 6.7% | 0.7 pts; 6.0% to 8.0% | +0.2 ± 0.9 pts | 5.9% | 6 of 6 |
| ETS(A,A,A) on log | 6.8% | 1.2 pts; 5.3% to 8.3% | +0.3 ± 0.4 pts | 5.5% | 5 of 6 |
| Damped-trend ETS on log | 7.6% | 2.0 pts; 5.4% to 11.0% | +1.1 ± 0.5 pts | 5.2% | 5 of 6 |
| Drift (baseline) | 18.8% | 7.7 pts; 10.2% to 29.4% | +12.3 ± 2.5 pts | 22.6% | 2 of 6 |
| Seasonal naive (baseline) | 42.5% | 0.6 pts; 41.4% to 43.2% | +36.0 ± 0.7 pts | 44.3% | 0 of 6 |

**How the forecast was chosen, and why the holdout doesn't overturn it.** The selection rule was fixed before the holdout was touched: lowest MAPE averaged over all 42 backtest forecasts (7 origins x 6 steps), among candidates whose fits all converged. Trend + Fourier wins that comparison. On the six holdout months it has the highest error of the five log-scale candidates, and damped ETS, last on the backtest, has the lowest. **Six holdout months can't distinguish models whose backtest errors overlap.** The holdout gap between the best and worst log-scale model is 1.0 point, while the selected model's own error moves across a 4.9-point range from one backtest origin to the next, and ETS, SARIMA and the ensemble are each within one paired standard error of it on the backtest. Switching to whichever model happened to win the holdout would be selecting on the test set. What the holdout does confirm is the large gap: every log-scale model cuts error by roughly three quarters against the best baseline (22.6% MAPE for drift).

![Revenue holdout and forward forecast, and backtest spread vs holdout](figures/revenue_forecast.png)

Left: holdout forecast from the selected model and the forward forecast from a refit on all 42 months. Right: each gray dot is one backtest origin's MAPE, the bar is the backtest mean used for selection, and the diamond is the single holdout score.

## Recommendation

- **Churn outreach:** give account managers the model's ranked list each period and size the effort to capacity. At the stated assumptions the top 10% returns $5.40 of protected annual revenue per outreach dollar; returns fall to about $1.60 per dollar on the block between the top 30% and top 50%, which only pays if gross margin on retained revenue exceeds roughly 62%. The save rate is the assumption everything hinges on, so measure it with a randomized holdout group in the first outreach cycle before scaling up.
- **Revenue planning:** plan July to December 2025 on the trend + Fourier forecast, $1.04M to $1.12M per month, with its 95% interval (about $0.92M to $1.27M across the six months). Use the equal-weight ensemble as a cross-check. Rerun the backtest as each month closes; if the ensemble or another model pulls clearly ahead on the rolling backtest, switch on that evidence, not on one six-month window.

## Approach

- **Chronological split with a label embargo.** Account snapshots are split 60/20/20 by date. Because churn is observed 90 days after a snapshot, training rows whose outcome window would still be open at the start of validation are dropped.
- **Logistic regression vs. histogram gradient boosting**, each tuned on the validation period and compared on **log loss**, a proper scoring rule that rewards calibrated probabilities as well as ranking. The test period is scored once.
- **Known-truth benchmark.** Since the generator's churn model is known, its true probabilities give an upper bound on achievable performance, and fitted coefficients can be checked against the true ones.
- **Retention decision analysis.** The selected model's test-period scores rank accounts; for capacities from 5% to 50% of accounts the top-k list is valued with the stated save rate and outreach cost, with bootstrap intervals and a sensitivity grid over both assumptions.
- **Log-scale forecasting.** Revenue compounds by a percentage each month and its seasonal swing scales with its level, so ETS, SARIMA and a trend + Fourier harmonic regression are fit to log revenue. An equal-weight ensemble of the four log-scale models, with members fixed in advance, competes as a fifth candidate.
- **Rolling-origin backtest for model selection.** Seven origins, each forecasting six months ahead within the first 36 months. The lowest mean MAPE among candidates whose fits all converged is selected; per-origin errors and paired comparisons are saved to show how separable the candidates are. The final six months are then scored once.
- **Uncertainty quantification.** Horizon-specific 95% prediction intervals for every method (analytic intervals for the naive baselines) with empirical coverage, and bootstrap intervals for the churn model comparison and the retention values.

## Data

The data are **synthetic**, produced by `generate_data.py` with a fixed seed. No real company, account or customer is represented.

- `data/accounts.csv`: 1,400 account snapshots, one per day from January 2021 to October 2024, with region, specialty, account type, rep tenure, onboarding score, product lines, discounting, support tickets, engagement, annual revenue and ordering behavior (frequency, order value, recency). The 90-day churn outcome is drawn from a logistic model in seven of these features. Overall churn rate: 22.6%.
- `data/monthly_revenue.csv`: 42 months (January 2022 to June 2025) of revenue with 4.5% compounding monthly growth, a 14% seasonal swing peaking in June and 5% multiplicative noise.

## Repository structure

```
├── generate_data.py                    # synthetic data generator (true churn model lives here)
├── churn_model.py                      # temporal split, models, bootstrap, coefficient recovery, retention value
├── revenue_forecast.py                 # forecasting methods, ensemble, rolling-origin backtest, holdout
├── analysis.py                         # runs both analyses, writes metrics/ and figures/
├── sales_churn_and_forecasting.ipynb   # walkthrough with outputs
├── data/                               # generated CSVs
├── metrics/                            # every reported number (JSON and CSV)
├── figures/                            # churn ROC + coefficient recovery, revenue forecast + backtest spread
└── tests/
```

Every number above comes from `metrics/`: `sales_metrics.json` (churn metrics, revenue selection and holdout), `retention_capacity.csv`, `retention_sensitivity.csv`, `rolling_origin_mape_by_origin.csv` and `revenue_forward_forecast.csv`.

## Reproduce

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python generate_data.py     # rewrites data/ identically
python analysis.py          # about 30 seconds; rewrites metrics/ and figures/
pytest -q
jupyter nbconvert --to notebook --execute --inplace sales_churn_and_forecasting.ipynb
```

## Limitations

- The data are synthetic, and the churn outcome is linear in the log-odds by construction, which favors logistic regression. On real accounts with interactions or thresholds, gradient boosting may win; the same validation protocol would decide.
- The retention save rate and outreach cost are assumptions, not estimates. Revenue protected counts one year of revenue with no margin, ignores the chance that saved accounts churn later, and assumes outreach doesn't change behavior for accounts that weren't going to churn. Only a randomized pilot can estimate the save rate.
- The revenue generator is a log-linear trend times a sinusoid, so the trend + Fourier model is close to correctly specified, which is likely why its backtest error falls as history grows (8.7% at the earliest origin, 3.8% at the latest). Real revenue with trend changes or shocks would need the backtest rerun, and damped or local-trend models may do better there.
- Seven backtest origins overlap heavily, so the paired standard errors are rough. The ensemble's interval averages its members' bounds on the log scale, which is an approximation rather than an exact prediction interval.
- The churn test period has 280 snapshots (63 churners), so the confidence intervals are wide. The revenue holdout is six points.
- `annual_revenue` is order frequency times order value, so those coefficients split weight between collinear features and shouldn't be read individually.
- Coefficients describe association in the generator, not the effect of an intervention such as a retention discount.
