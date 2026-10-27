"""90-day account churn: temporal validation of logistic regression vs gradient boosting."""

from __future__ import annotations

import json
from typing import Iterable, Sequence

import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from generate_data import CHURN_COEFFICIENTS, true_churn_probability

RANDOM_STATE = 42
N_BOOT = 2000
TARGET = "churned"
NON_FEATURES = ("account_id", "snapshot_date", "churn_horizon_days", TARGET)
CATEGORICAL = ["region", "specialty", "account_type"]
LOGISTIC_GRID: tuple[dict, ...] = tuple({"C": c} for c in (0.01, 0.1, 1.0, 10.0))
BOOSTING_GRID: tuple[dict, ...] = (
    {"learning_rate": 0.05, "max_iter": 100, "max_leaf_nodes": 7},
    {"learning_rate": 0.05, "max_iter": 200, "max_leaf_nodes": 7},
    {"learning_rate": 0.03, "max_iter": 300, "max_leaf_nodes": 4},
    {"learning_rate": 0.1, "max_iter": 100, "max_leaf_nodes": 15},
)


def feature_columns(accounts: pd.DataFrame) -> list[str]:
    """Every column except identifiers, the snapshot date and the outcome."""
    return [c for c in accounts.columns if c not in NON_FEATURES]


def temporal_split(
    accounts: pd.DataFrame, embargo_days: int | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological 60/20/20 split by snapshot date with a label embargo.

    A snapshot's label is only known ``churn_horizon_days`` after the
    snapshot. Training rows whose outcome window would still be open when
    validation starts are dropped, so the model only learns from labels that
    would actually have been observed by then.
    """
    dates = pd.to_datetime(accounts["snapshot_date"])
    if embargo_days is None:
        embargo_days = int(accounts["churn_horizon_days"].max())
    q60, q80 = dates.quantile([0.60, 0.80])
    train = accounts.loc[dates <= q60 - pd.Timedelta(days=embargo_days)].copy()
    val = accounts.loc[(dates > q60) & (dates <= q80)].copy()
    test = accounts.loc[dates > q80].copy()
    return train, val, test


def build_logistic(features: Sequence[str], C: float) -> Pipeline:
    numeric = [c for c in features if c not in CATEGORICAL]
    prep = ColumnTransformer(
        [("num", StandardScaler(), numeric), ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)]
    )
    return Pipeline([("preprocess", prep), ("model", LogisticRegression(C=C, max_iter=1000))])


def build_boosting(features: Sequence[str], learning_rate: float, max_iter: int, max_leaf_nodes: int) -> Pipeline:
    numeric = [c for c in features if c not in CATEGORICAL]
    prep = ColumnTransformer(
        [
            ("num", "passthrough", numeric),
            ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan), CATEGORICAL),
        ]
    )
    model = HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_iter=max_iter,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=20,
        categorical_features=[False] * len(numeric) + [True] * len(CATEGORICAL),
        early_stopping=False,
        random_state=RANDOM_STATE,
    )
    return Pipeline([("preprocess", prep), ("model", model)])


def classification_metrics(y: Iterable[int], p: Iterable[float]) -> dict[str, float]:
    y, p = np.asarray(y), np.asarray(p)
    return {
        "roc_auc": float(roc_auc_score(y, p)),
        "pr_auc": float(average_precision_score(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "log_loss": float(log_loss(y, np.clip(p, 1e-6, 1 - 1e-6))),
    }


def top_decile(y: Iterable[int], p: Iterable[float]) -> dict[str, float]:
    """Precision and capture when the highest-risk 10% of accounts are flagged."""
    y, p = np.asarray(y), np.asarray(p)
    flagged = np.argsort(-p, kind="stable")[: max(1, len(y) // 10)]
    precision = float(y[flagged].mean())
    return {
        "precision": precision,
        "capture": float(y[flagged].sum() / max(y.sum(), 1)),
        "lift": precision / float(y.mean()),
    }


def calibration_table(y: Iterable[int], p: Iterable[float], n_bins: int = 5) -> pd.DataFrame:
    """Mean predicted risk vs observed churn in equal-count bins."""
    observed, predicted = calibration_curve(np.asarray(y), np.asarray(p), n_bins=n_bins, strategy="quantile")
    return pd.DataFrame(
        {"bin": np.arange(1, len(predicted) + 1), "mean_predicted": predicted, "observed_rate": observed}
    )


def bootstrap(
    y: Iterable[int],
    predictions: dict[str, np.ndarray],
    paired: tuple[str, str],
    n_boot: int = N_BOOT,
    seed: int = RANDOM_STATE,
) -> dict:
    """Account-level percentile bootstrap; the AUC difference is paired."""
    y = np.asarray(y)
    rng = np.random.default_rng(seed)
    draws: dict[str, dict[str, list[float]]] = {
        name: {"roc_auc": [], "pr_auc": [], "brier": [], "top_decile_precision": []} for name in predictions
    }
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if len(np.unique(y[idx])) < 2:
            continue
        aucs = {}
        for name, p in predictions.items():
            m = classification_metrics(y[idx], p[idx])
            aucs[name] = m["roc_auc"]
            for key in ("roc_auc", "pr_auc", "brier"):
                draws[name][key].append(m[key])
            draws[name]["top_decile_precision"].append(top_decile(y[idx], p[idx])["precision"])
        diffs.append(aucs[paired[1]] - aucs[paired[0]])

    def interval(values: Sequence[float]) -> dict[str, float]:
        return {"low_2.5": float(np.quantile(values, 0.025)), "high_97.5": float(np.quantile(values, 0.975))}

    out: dict = {name: {k: interval(v) for k, v in metrics.items()} for name, metrics in draws.items()}
    out[f"roc_auc_difference_{paired[1]}_minus_{paired[0]}"] = interval(diffs)
    out["n_resamples"] = len(diffs)
    return out


def coefficient_recovery(pipe: Pipeline, train: pd.DataFrame) -> pd.DataFrame:
    """Fitted standardized logistic coefficients next to the generator's true ones.

    True coefficients are converted to the same per-standard-deviation scale
    using the training data; features the generator does not use have a true
    coefficient of zero.
    """
    names = pipe.named_steps["preprocess"].get_feature_names_out()
    fitted = pipe.named_steps["model"].coef_[0]
    rows = []
    for name, coef in zip(names, fitted):
        kind, feature = name.split("__", 1)
        true = 0.0
        if kind == "num" and feature in CHURN_COEFFICIENTS:
            true = CHURN_COEFFICIENTS[feature][0] * float(train[feature].std(ddof=0))
        rows.append(
            {
                "encoded_feature": feature,
                "feature_type": "numeric" if kind == "num" else "categorical",
                "fitted_coefficient_per_sd": float(coef),
                "true_coefficient_per_sd": true,
                "is_true_driver": kind == "num" and feature in CHURN_COEFFICIENTS,
            }
        )
    table = pd.DataFrame(rows)
    return table.reindex(table["fitted_coefficient_per_sd"].abs().sort_values(ascending=False).index).reset_index(
        drop=True
    )


# Retention planning assumptions. These are stated, illustrative inputs, NOT estimates from the data.
# Only account revenue (``annual_revenue``) and the churn outcomes come from the data.
SAVE_RATE = 0.30  # share of would-be churners that outreach retains
OUTREACH_COST = 2_500.0  # rep time plus a retention offer, per account contacted (USD)
CAPACITY_SHARES = (0.05, 0.10, 0.20, 0.30, 0.50)  # share of test-period accounts the team can contact
SENSITIVITY_SAVE_RATES = (0.15, 0.30, 0.45)
SENSITIVITY_COSTS = (1_500.0, 2_500.0, 4_000.0)


def top_k_mask(p: np.ndarray, k: int) -> np.ndarray:
    """Boolean mask of the ``k`` highest-risk rows (stable ordering for ties)."""
    mask = np.zeros(len(p), dtype=bool)
    mask[np.argsort(-p, kind="stable")[:k]] = True
    return mask


def revenue_protected(y: np.ndarray, revenue: np.ndarray, contact: np.ndarray, save_rate: float) -> float:
    """Annual revenue kept: revenue of contacted accounts that would have churned, times the save rate."""
    return float(save_rate * np.sum(contact * y * revenue))


def retention_capacity(y: np.ndarray, revenue: np.ndarray, p: np.ndarray, n_boot: int) -> dict:
    """Revenue protected vs outreach effort when the team works the model's top-k list on the test period."""
    n = len(y)
    churned_revenue = float(np.sum(y * revenue))
    rng = np.random.default_rng(RANDOM_STATE)
    boot_idx = [rng.integers(0, n, n) for _ in range(n_boot)]
    rows, sensitivity = [], []
    for share in CAPACITY_SHARES:
        k = int(round(share * n))
        contact = top_k_mask(p, k)
        protected = revenue_protected(y, revenue, contact, SAVE_RATE)
        cost = OUTREACH_COST * k
        boot = [revenue_protected(y[i], revenue[i], top_k_mask(p[i], k), SAVE_RATE) for i in boot_idx]
        rows.append(
            {
                "capacity_share": share,
                "accounts_contacted": k,
                "churners_reached": int(np.sum(contact & (y == 1))),
                "share_of_churned_revenue_reached": float(np.sum(contact * y * revenue) / churned_revenue),
                "revenue_protected": protected,
                "revenue_protected_ci_low": float(np.quantile(boot, 0.025)),
                "revenue_protected_ci_high": float(np.quantile(boot, 0.975)),
                "model_expected_revenue_protected": float(SAVE_RATE * np.sum(contact * p * revenue)),
                "random_list_revenue_protected": SAVE_RATE * k / n * churned_revenue,
                "outreach_cost": cost,
                "net_of_outreach_cost": protected - cost,
                "revenue_protected_per_outreach_dollar": protected / cost,
            }
        )
        for save_rate in SENSITIVITY_SAVE_RATES:
            for unit_cost in SENSITIVITY_COSTS:
                sensitivity.append(
                    {
                        "accounts_contacted": k,
                        "save_rate": save_rate,
                        "outreach_cost_per_account": unit_cost,
                        "net_of_outreach_cost": revenue_protected(y, revenue, contact, save_rate) - unit_cost * k,
                    }
                )
    return {
        "assumptions": {
            "save_rate": SAVE_RATE,
            "outreach_cost_per_account": OUTREACH_COST,
            "revenue_basis": "annual_revenue of each test-period account, from the data",
            "value_definition": "save_rate x annual revenue of contacted accounts that churned; no margin applied",
        },
        "test_accounts": n,
        "test_churned_revenue": churned_revenue,
        "capacity": pd.DataFrame(rows),
        "sensitivity": pd.DataFrame(sensitivity),
    }


def run_churn(accounts: pd.DataFrame, n_boot: int = N_BOOT) -> dict:
    """Select on validation log loss, score test once, and return tables plus a summary."""
    features = feature_columns(accounts)
    train, val, test = temporal_split(accounts)
    candidates = [("logistic_regression", p, build_logistic(features, **p)) for p in LOGISTIC_GRID]
    candidates += [("gradient_boosting", p, build_boosting(features, **p)) for p in BOOSTING_GRID]

    selection_rows, best = [], {}
    for family, params, pipe in candidates:
        pipe.fit(train[features], train[TARGET])
        scores = classification_metrics(val[TARGET], pipe.predict_proba(val[features])[:, 1])
        selection_rows.append({"model": family, "params": json.dumps(params), **scores})
        if family not in best or scores["log_loss"] < best[family][1]["log_loss"]:
            best[family] = (pipe, scores, params)
    selected = min(best, key=lambda name: best[name][1]["log_loss"])
    other = "gradient_boosting" if selected == "logistic_regression" else "logistic_regression"

    y_test = test[TARGET].to_numpy()
    test_pred = {name: best[name][0].predict_proba(test[features])[:, 1] for name in best}
    oracle = true_churn_probability(test)
    base_rate = float(train[TARGET].mean())
    test_metrics = {"prevalence_baseline": classification_metrics(y_test, np.full(len(test), base_rate))}
    test_metrics.update({name: classification_metrics(y_test, p) for name, p in test_pred.items()})
    test_metrics["oracle_true_probability"] = classification_metrics(y_test, oracle)
    ranking = {name: top_decile(y_test, p) for name, p in {**test_pred, "oracle_true_probability": oracle}.items()}
    uncertainty = bootstrap(y_test, test_pred, paired=(other, selected), n_boot=n_boot)
    calibration = pd.concat([calibration_table(y_test, p).assign(model=name) for name, p in test_pred.items()])
    recovery = coefficient_recovery(best["logistic_regression"][0], train)
    drivers = recovery.loc[recovery["is_true_driver"]]
    retention = retention_capacity(y_test, test["annual_revenue"].to_numpy(), test_pred[selected], n_boot)

    summary = {
        "split": {
            "train_rows": len(train),
            "validation_rows": len(val),
            "test_rows": len(test),
            "train_snapshots": [str(train.snapshot_date.min()), str(train.snapshot_date.max())],
            "validation_snapshots": [str(val.snapshot_date.min()), str(val.snapshot_date.max())],
            "test_snapshots": [str(test.snapshot_date.min()), str(test.snapshot_date.max())],
            "label_embargo_days": int(accounts["churn_horizon_days"].max()),
            "test_churn_rate": float(y_test.mean()),
        },
        "selection_metric": "validation log loss (a proper scoring rule, so it rewards calibrated probabilities as well as ranking)",
        "validation": {name: best[name][1] for name in best},
        "selected_hyperparameters": {name: best[name][2] for name in best},
        "selected_model": selected,
        "test": test_metrics,
        "top_decile": ranking,
        "uncertainty_95pct": uncertainty,
        "retention": {
            "assumptions": retention["assumptions"],
            "test_accounts": retention["test_accounts"],
            "test_churned_revenue": retention["test_churned_revenue"],
            "capacity": retention["capacity"].to_dict(orient="records"),
        },
        "coefficient_recovery": {
            "true_drivers": int(len(drivers)),
            "true_drivers_with_correct_sign": int(
                (np.sign(drivers.fitted_coefficient_per_sd) == np.sign(drivers.true_coefficient_per_sd)).sum()
            ),
            "correlation_fitted_vs_true": float(
                np.corrcoef(recovery.fitted_coefficient_per_sd, recovery.true_coefficient_per_sd)[0, 1]
            ),
        },
    }
    return {
        "summary": summary,
        "retention_capacity": retention["capacity"],
        "retention_sensitivity": retention["sensitivity"],
        "model_selection": pd.DataFrame(selection_rows),
        "calibration": calibration,
        "coefficients": recovery,
        "test_predictions": pd.DataFrame(
            {
                "account_id": test.account_id.to_numpy(),
                "churned": y_test,
                **test_pred,
                "oracle_true_probability": oracle,
            }
        ),
    }
