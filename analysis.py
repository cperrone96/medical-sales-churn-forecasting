"""Run the churn and revenue analyses and write every reported number and figure.

Usage:  python analysis.py
Reads ``data/`` (written by generate_data.py); writes ``metrics/`` and ``figures/``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve

from churn_model import run_churn
from revenue_forecast import BASELINES, run_revenue

ROOT = Path(__file__).resolve().parent
LABELS = {
    "logistic_regression": "Logistic regression",
    "gradient_boosting": "Gradient boosting",
    "oracle_true_probability": "True generator probability",
    "seasonal_naive": "Seasonal naive",
    "drift": "Drift",
    "ets_log": "ETS(A,A,A) on log",
    "ets_damped_log": "Damped ETS on log",
    "sarima_log": "SARIMA airline on log",
    "trend_fourier_log": "Trend + Fourier on log",
}


def load_revenue(path: Path) -> pd.Series:
    return pd.read_csv(path, parse_dates=["month"]).set_index("month")["revenue"].sort_index()


def plot_churn(predictions: pd.DataFrame, coefficients: pd.DataFrame, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax_roc, ax_coef) = plt.subplots(1, 2, figsize=(11, 4.6), gridspec_kw={"width_ratios": [1, 1.15]})
    styles = {
        "logistic_regression": ("#2a6fb0", "-"),
        "gradient_boosting": ("#d1495b", "-"),
        "oracle_true_probability": ("#555555", ":"),
    }
    y = predictions["churned"].to_numpy()
    for name, (color, ls) in styles.items():
        fpr, tpr, _ = roc_curve(y, predictions[name])
        ax_roc.plot(
            fpr, tpr, ls, color=color, lw=2, label=f"{LABELS[name]} ({roc_auc_score(y, predictions[name]):.3f})"
        )
    ax_roc.plot([0, 1], [0, 1], "--", color="#aaaaaa", lw=1, label="Prevalence baseline (0.500)")
    ax_roc.set(xlabel="False positive rate", ylabel="True positive rate", title="Churn ROC, test period (AUC)")
    ax_roc.legend(loc="lower right", frameon=False, fontsize=9)

    numeric = coefficients.loc[coefficients.feature_type == "numeric"].sort_values(
        ["true_coefficient_per_sd", "fitted_coefficient_per_sd"]
    )
    pos = np.arange(len(numeric))
    ax_coef.barh(pos - 0.2, numeric.true_coefficient_per_sd, height=0.4, color="#bbbbbb", label="True (generator)")
    ax_coef.barh(
        pos + 0.2, numeric.fitted_coefficient_per_sd, height=0.4, color="#2a6fb0", label="Fitted (logistic regression)"
    )
    ax_coef.set_yticks(pos, numeric.encoded_feature)
    ax_coef.axvline(0, color="#333333", lw=0.8)
    ax_coef.set(xlabel="Log-odds change per 1 SD", title="Coefficient recovery, numeric features")
    ax_coef.legend(frameon=False, fontsize=9, loc="lower right")
    for ax in (ax_roc, ax_coef):
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_revenue(revenue: pd.Series, holdout: pd.DataFrame, forward: pd.DataFrame, selected: str, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.plot(revenue.index, revenue / 1e6, color="#222222", lw=2, label="Actual revenue")
    months = pd.to_datetime(holdout.loc[holdout.method == selected, "month"])
    sel = holdout.loc[holdout.method == selected]
    ax.fill_between(months, sel.lower_95 / 1e6, sel.upper_95 / 1e6, color="#d1495b", alpha=0.18, lw=0)
    ax.plot(
        months,
        sel.forecast / 1e6,
        "o-",
        color="#d1495b",
        lw=2,
        label=f"{LABELS[selected]} (selected), holdout forecast + 95% PI",
    )
    for name, color in zip(BASELINES, ("#7a9e3a", "#e09f3e")):
        part = holdout.loc[holdout.method == name]
        ax.plot(
            months, part.forecast / 1e6, "s--", color=color, lw=1.5, ms=4, label=f"{LABELS[name]}, holdout forecast"
        )
    fmonths = pd.to_datetime(forward.month)
    ax.fill_between(fmonths, forward.lower_95 / 1e6, forward.upper_95 / 1e6, color="#2a6fb0", alpha=0.18, lw=0)
    ax.plot(
        fmonths, forward.forecast / 1e6, "o-", color="#2a6fb0", lw=2, label="Next 6 months, refit on all data + 95% PI"
    )
    ax.axvline(months.iloc[0] - pd.Timedelta(days=15), color="#999999", lw=1, ls=":")
    ax.set(ylabel="Monthly revenue ($M)", title="Revenue forecast: 6-month holdout and forward forecast")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_analysis(
    data_dir: Path = ROOT / "data", output_dir: Path = ROOT / "metrics", figure_dir: Path | None = ROOT / "figures"
) -> dict:
    data_dir, output_dir = Path(data_dir), Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    accounts = pd.read_csv(data_dir / "accounts.csv", parse_dates=["snapshot_date"])
    revenue = load_revenue(data_dir / "monthly_revenue.csv")

    churn = run_churn(accounts)
    churn["model_selection"].to_csv(output_dir / "churn_model_selection.csv", index=False)
    churn["calibration"].to_csv(output_dir / "churn_calibration.csv", index=False)
    churn["coefficients"].to_csv(output_dir / "churn_logistic_coefficients.csv", index=False)

    rev = run_revenue(revenue)
    rev["backtest"].to_csv(output_dir / "rolling_origin_forecasts.csv", index=False)
    rev["holdout"].to_csv(output_dir / "revenue_holdout_forecasts.csv", index=False)
    rev["forward"].to_csv(output_dir / "revenue_forward_forecast.csv", index=False)

    summary = {
        "data": {"synthetic": True, "accounts": len(accounts), "churn_rate": float(accounts["churned"].mean())},
        "churn": churn["summary"],
        "revenue": rev["summary"],
    }
    (output_dir / "sales_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")
    if figure_dir is not None:
        figure_dir = Path(figure_dir)
        figure_dir.mkdir(parents=True, exist_ok=True)
        plot_churn(churn["test_predictions"], churn["coefficients"], figure_dir / "roc_curves.png")
        plot_revenue(
            revenue,
            rev["holdout"],
            rev["forward"],
            rev["summary"]["selected_method"],
            figure_dir / "revenue_forecast.png",
        )
    return summary


if __name__ == "__main__":
    run_analysis()
