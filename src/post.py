
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"


# ============================================================
# RESEARCH / ROUTING SETTINGS
# ============================================================
#
# These are the operating thresholds already selected by the
# validation-based routing experiment.
#
# 0.20  -> lower probability boundary
# 0.55  -> higher probability / risk boundary
# 0.003127... -> uncertainty boundary selected on validation
#
# We do NOT optimize these on the test set.
# ============================================================

LOW_PROB_THRESHOLD = 0.20
HIGH_PROB_THRESHOLD = 0.55
OPERATING_THRESHOLD = 0.20

VARIANCE_THRESHOLD = 0.00312741430937158

# Normalized research costs used by the existing routing experiment.
COST_FALSE_SAFE = 5.0
COST_MANUAL = 1.0
COST_FALSE_ESCALATE = 2.0


# ============================================================
# BASIC METRICS
# ============================================================

def ece_score(y, p, n_bins=10):
    y = np.asarray(y).astype(int)
    p = np.asarray(p, dtype=float)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (p >= edges[i]) & (p <= edges[i + 1])
        else:
            mask = (p >= edges[i]) & (p < edges[i + 1])

        if mask.sum() == 0:
            continue

        ece += (
            mask.sum() / len(y)
        ) * abs(
            y[mask].mean() - p[mask].mean()
        )

    return float(ece)


def metric_dict(
    y,
    p,
    threshold=0.50
):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)

    pred = (
        p >= threshold
    ).astype(int)

    return {
        "AUC": float(
            roc_auc_score(y, p)
        ),
        "PR_AUC": float(
            average_precision_score(y, p)
        ),
        "Brier": float(
            brier_score_loss(y, p)
        ),
        "Accuracy": float(
            accuracy_score(y, pred)
        ),
        "Precision": float(
            precision_score(
                y,
                pred,
                zero_division=0
            )
        ),
        "Recall": float(
            recall_score(
                y,
                pred,
                zero_division=0
            )
        ),
        "F1": float(
            f1_score(
                y,
                pred,
                zero_division=0
            )
        ),
        "ECE": float(
            ece_score(y, p)
        ),
    }


# ============================================================
# THRESHOLD ANALYSIS
# ============================================================

def threshold_analysis(df):
    y = df[
        "actual_label"
    ].to_numpy()

    p = df[
        "ensemble_probability"
    ].to_numpy()

    rows = []

    for threshold in [
        0.20,
        0.50,
        0.55,
    ]:

        pred = (
            p >= threshold
        ).astype(int)

        rows.append({
            "threshold": threshold,
            "samples_predicted_default": int(
                pred.sum()
            ),
            "default_prediction_pct": float(
                pred.mean() * 100
            ),
            "precision": float(
                precision_score(
                    y,
                    pred,
                    zero_division=0
                )
            ),
            "recall": float(
                recall_score(
                    y,
                    pred,
                    zero_division=0
                )
            ),
            "f1": float(
                f1_score(
                    y,
                    pred,
                    zero_division=0
                )
            ),
            "accuracy": float(
                accuracy_score(y, pred)
            ),
        })

    out = pd.DataFrame(rows)

    out.to_csv(
        OUTPUT_DIR
        / "threshold_performance.csv",
        index=False
    )

    print(
        "\nTHRESHOLD_ANALYSIS"
    )
    print(
        out.to_string(index=False)
    )

    return out


# ============================================================
# CALIBRATION
# ============================================================

def calibration_table(
    y,
    p,
    n_bins=10
):
    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)

    edges = np.linspace(
        0.0,
        1.0,
        n_bins + 1
    )

    rows = []

    for i in range(n_bins):

        if i == n_bins - 1:
            mask = (
                (p >= edges[i])
                & (p <= edges[i + 1])
            )
        else:
            mask = (
                (p >= edges[i])
                & (p < edges[i + 1])
            )

        if mask.sum() == 0:
            continue

        rows.append({
            "bin": i + 1,
            "lower": edges[i],
            "upper": edges[i + 1],
            "samples": int(mask.sum()),
            "mean_predicted": float(
                p[mask].mean()
            ),
            "observed_default_rate": float(
                y[mask].mean()
            ),
        })

    return pd.DataFrame(rows)


def calibration_analysis(df):
    y = df[
        "actual_label"
    ].to_numpy()

    names = [
        "logit",
        "rf",
        "extra_trees",
        "hgb",
        "xgb",
        "catboost",
        "ensemble_probability",
    ]

    rows = []
    curves = {}

    for name in names:

        if name not in df.columns:
            continue

        p = df[name].to_numpy()

        rows.append({
            "model": name,
            **metric_dict(
                y,
                p,
                threshold=0.50
            ),
        })

        curves[name] = calibration_table(
            y,
            p
        )

    out = pd.DataFrame(rows)

    out.to_csv(
        OUTPUT_DIR
        / "calibration_metrics.csv",
        index=False
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    for name, tab in curves.items():

        ax.plot(
            tab["mean_predicted"],
            tab["observed_default_rate"],
            marker="o",
            label=name,
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        label="perfect calibration",
    )

    ax.set_xlabel(
        "Mean predicted probability"
    )

    ax.set_ylabel(
        "Observed default rate"
    )

    ax.set_title(
        "Calibration / Reliability Curves"
    )

    ax.legend(
        fontsize=8
    )

    fig.tight_layout()

    fig.savefig(
        OUTPUT_DIR
        / "calibration_reliability.png",
        dpi=200
    )

    plt.close(fig)

    if (
        "ensemble_probability"
        in curves
    ):
        curves[
            "ensemble_probability"
        ].to_csv(
            OUTPUT_DIR
            / "ensemble_calibration_table.csv",
            index=False
        )

    print(
        "\nCALIBRATION_ANALYSIS"
    )

    print(
        out.to_string(index=False)
    )

    return out


# ============================================================
# BOOTSTRAP STATISTICAL VALIDATION
# ============================================================

def bootstrap_metrics(
    y,
    p,
    n_boot=1000,
    seed=42
):
    rng = np.random.default_rng(
        seed
    )

    y = np.asarray(y).astype(int)
    p = np.asarray(p).astype(float)

    values = []

    for _ in range(n_boot):

        idx = rng.integers(
            0,
            len(y),
            len(y)
        )

        yb = y[idx]
        pb = p[idx]

        if len(
            np.unique(yb)
        ) < 2:
            continue

        values.append([
            roc_auc_score(
                yb,
                pb
            ),
            average_precision_score(
                yb,
                pb
            ),
            brier_score_loss(
                yb,
                pb
            ),
        ])

    values = np.asarray(
        values
    )

    names = [
        "AUC",
        "PR_AUC",
        "Brier"
    ]

    base = metric_dict(
        y,
        p
    )

    result = {}

    for j, name in enumerate(names):

        result[name] = {
            "estimate": float(
                base[name]
            ),
            "lower_95": float(
                np.percentile(
                    values[:, j],
                    2.5
                )
            ),
            "upper_95": float(
                np.percentile(
                    values[:, j],
                    97.5
                )
            ),
        }

    return result


def statistical_validation(df):
    y = df[
        "actual_label"
    ].to_numpy()

    rows = []

    for label, column in [
        ("xgb", "xgb"),
        (
            "ensemble",
            "ensemble_probability"
        ),
    ]:

        if column not in df.columns:
            continue

        ci = bootstrap_metrics(
            y,
            df[column].to_numpy(),
            n_boot=1000,
            seed=42
        )

        rows.append({
            "model": label,
            "AUC": ci["AUC"]["estimate"],
            "AUC_95_lower": ci["AUC"]["lower_95"],
            "AUC_95_upper": ci["AUC"]["upper_95"],
            "PR_AUC": ci["PR_AUC"]["estimate"],
            "PR_AUC_95_lower": ci["PR_AUC"]["lower_95"],
            "PR_AUC_95_upper": ci["PR_AUC"]["upper_95"],
            "Brier": ci["Brier"]["estimate"],
            "Brier_95_lower": ci["Brier"]["lower_95"],
            "Brier_95_upper": ci["Brier"]["upper_95"],
        })

    out = pd.DataFrame(
        rows
    )

    out.to_csv(
        OUTPUT_DIR
        / "bootstrap_95ci.csv",
        index=False
    )

    print(
        "\nSTATISTICAL_VALIDATION_BOOTSTRAP_95CI"
    )

    print(
        out.to_string(index=False)
    )

    return out


# ============================================================
# PAIRED COST COMPARISON
# ============================================================

def apply_probability_policy(
    p
):
    decisions = np.full(
        len(p),
        "manual_review",
        dtype=object
    )

    decisions[
        p < LOW_PROB_THRESHOLD
    ] = "auto_process"

    decisions[
        p >= HIGH_PROB_THRESHOLD
    ] = "escalate_for_review"

    return decisions


def apply_uncertainty_policy(
    p,
    variance
):
    decisions = np.full(
        len(p),
        "manual_review",
        dtype=object
    )

    confident = (
        variance
        < VARIANCE_THRESHOLD
    )

    decisions[
        confident
        & (p < LOW_PROB_THRESHOLD)
    ] = "auto_process"

    decisions[
        confident
        & (p >= HIGH_PROB_THRESHOLD)
    ] = "escalate_for_review"

    return decisions


def decision_cost(
    decisions,
    y
):
    decisions = np.asarray(
        decisions
    )

    y = np.asarray(y).astype(int)

    costs = np.zeros(
        len(y)
    )

    auto = (
        decisions
        == "auto_process"
    )

    manual = (
        decisions
        == "manual_review"
    )

    escalate = (
        decisions
        == "escalate_for_review"
    )

    costs[
        auto & (y == 1)
    ] = COST_FALSE_SAFE

    costs[
        manual
    ] = COST_MANUAL

    costs[
        escalate & (y == 0)
    ] = COST_FALSE_ESCALATE

    return costs


def paired_cost_bootstrap(
    df,
    n_boot=5000,
    seed=42
):
    rng = np.random.default_rng(
        seed
    )

    y = df[
        "actual_label"
    ].to_numpy()

    p = df[
        "ensemble_probability"
    ].to_numpy()

    variance = df[
        "prediction_variance"
    ].to_numpy()

    probability_decisions = (
        apply_probability_policy(p)
    )

    uncertainty_decisions = (
        apply_uncertainty_policy(
            p,
            variance
        )
    )

    probability_cost = (
        decision_cost(
            probability_decisions,
            y
        )
    )

    uncertainty_cost = (
        decision_cost(
            uncertainty_decisions,
            y
        )
    )

    # Positive difference means uncertainty-aware
    # routing costs more.
    observed_difference = (
        uncertainty_cost.mean()
        - probability_cost.mean()
    )

    diffs = []

    for _ in range(n_boot):

        idx = rng.integers(
            0,
            len(y),
            len(y)
        )

        diffs.append(
            (
                uncertainty_cost[idx].mean()
                - probability_cost[idx].mean()
            )
        )

    diffs = np.asarray(
        diffs
    )

    lower = float(
        np.percentile(
            diffs,
            2.5
        )
    )

    upper = float(
        np.percentile(
            diffs,
            97.5
        )
    )

    result = pd.DataFrame({
        "observed_cost_difference": [
            observed_difference
        ],
        "bootstrap_mean_difference": [
            float(diffs.mean())
        ],
        "lower_95": [lower],
        "upper_95": [upper],
        "n_boot": [n_boot],
    })

    result.to_csv(
        OUTPUT_DIR
        / "paired_cost_bootstrap.csv",
        index=False
    )

    print(
        "\nPAIRED_COST_BOOTSTRAP"
    )

    print(
        "Probability-only mean cost:",
        float(
            probability_cost.mean()
        )
    )

    print(
        "Uncertainty-aware mean cost:",
        float(
            uncertainty_cost.mean()
        )
    )

    print(
        "Observed cost difference:",
        float(
            observed_difference
        )
    )

    print(
        "95% CI:",
        lower,
        upper
    )

    if (
        lower <= 0
        <= upper
    ):
        print(
            "INTERPRETATION: "
            "difference is not clearly distinguishable "
            "from zero at the 95% bootstrap level."
        )
    else:
        print(
            "INTERPRETATION: "
            "95% bootstrap interval excludes zero."
        )

    # Save row-level paired results.
    paired = pd.DataFrame({
        "actual_label": y,
        "ensemble_probability": p,
        "prediction_variance": variance,
        "probability_only_decision": (
            probability_decisions
        ),
        "uncertainty_aware_decision": (
            uncertainty_decisions
        ),
        "probability_only_cost": (
            probability_cost
        ),
        "uncertainty_aware_cost": (
            uncertainty_cost
        ),
        "cost_difference": (
            uncertainty_cost
            - probability_cost
        ),
    })

    paired.to_csv(
        OUTPUT_DIR
        / "paired_routing_costs.csv",
        index=False
    )

    return result


# ============================================================
# ROUTING SUMMARY
# ============================================================

def routing_summary(
    df
):
    y = df[
        "actual_label"
    ].to_numpy()

    p = df[
        "ensemble_probability"
    ].to_numpy()

    variance = df[
        "prediction_variance"
    ].to_numpy()

    policies = {
        "probability_only": (
            apply_probability_policy(p)
        ),
        "uncertainty_aware": (
            apply_uncertainty_policy(
                p,
                variance
            )
        ),
    }

    rows = []

    for policy, decisions in policies.items():

        costs = decision_cost(
            decisions,
            y
        )

        for decision in [
            "auto_process",
            "manual_review",
            "escalate_for_review",
        ]:

            mask = (
                decisions
                == decision
            )

            rows.append({
                "policy": policy,
                "decision": decision,
                "samples": int(
                    mask.sum()
                ),
                "percentage": float(
                    mask.mean() * 100
                ),
                "mean_cost": float(
                    costs.mean()
                ),
            })

    out = pd.DataFrame(
        rows
    )

    out.to_csv(
        OUTPUT_DIR
        / "post_routing_summary.csv",
        index=False
    )

    print(
        "\nPOST_ROUTING_SUMMARY"
    )

    print(
        out.to_string(index=False)
    )

    return out


# ============================================================
# ERROR ANALYSIS
# ============================================================

def error_analysis(
    df
):
    out = df.copy()

    y = out[
        "actual_label"
    ].astype(int)

    p = out[
        "ensemble_probability"
    ].astype(float)

    # IMPORTANT:
    # This is the proposed operating threshold,
    # not the arbitrary default 0.50.
    out[
        "ensemble_prediction"
    ] = (
        p >= OPERATING_THRESHOLD
    ).astype(int)

    out[
        "error"
    ] = (
        out[
            "ensemble_prediction"
        ] != y
    ).astype(int)

    out[
        "error_type"
    ] = "correct"

    out.loc[
        (
            out["ensemble_prediction"] == 0
        )
        & (y == 1),
        "error_type",
    ] = "false_negative"

    out.loc[
        (
            out["ensemble_prediction"] == 1
        )
        & (y == 0),
        "error_type",
    ] = "false_positive"

    # --------------------------------------------------------
    # OPERATING THRESHOLD METRICS
    # --------------------------------------------------------

    precision = precision_score(
        y,
        out["ensemble_prediction"],
        zero_division=0
    )

    recall = recall_score(
        y,
        out["ensemble_prediction"],
        zero_division=0
    )

    f1 = f1_score(
        y,
        out["ensemble_prediction"],
        zero_division=0
    )

    print(
        "\nOPERATING_THRESHOLD_ANALYSIS"
    )

    print(
        "Operating threshold:",
        OPERATING_THRESHOLD
    )

    print(
        "Precision:",
        precision
    )

    print(
        "Recall:",
        recall
    )

    print(
        "F1:",
        f1
    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm = confusion_matrix(
        y,
        out[
            "ensemble_prediction"
        ]
    )

    cm_df = pd.DataFrame(
        cm,
        index=[
            "actual_0",
            "actual_1"
        ],
        columns=[
            "predicted_0",
            "predicted_1"
        ],
    )

    cm_df.to_csv(
        OUTPUT_DIR
        / "ensemble_confusion_matrix_operating_threshold.csv"
    )

    print(
        "\nCONFUSION_MATRIX_AT_OPERATING_THRESHOLD"
    )

    print(
        cm_df
    )

    # --------------------------------------------------------
    # PROBABILITY BINS
    # --------------------------------------------------------

    out[
        "probability_bin"
    ] = pd.cut(
        p,
        bins=np.linspace(
            0,
            1,
            11
        ),
        include_lowest=True
    )

    probability_error = (
        out.groupby(
            "probability_bin",
            observed=True
        )
        .agg(
            samples=("error", "size"),
            error_rate=("error", "mean"),
            mean_probability=(
                "ensemble_probability",
                "mean"
            ),
        )
        .reset_index()
    )

    probability_error.to_csv(
        OUTPUT_DIR
        / "error_analysis_probability_bins.csv",
        index=False
    )

    # --------------------------------------------------------
    # UNCERTAINTY QUARTILES
    # --------------------------------------------------------

    out[
        "uncertainty_quartile"
    ] = pd.qcut(
        out[
            "prediction_variance"
        ],
        q=4,
        labels=[
            "Q1_low",
            "Q2",
            "Q3",
            "Q4_high",
        ],
        duplicates="drop"
    )

    uncertainty_error = (
        out.groupby(
            "uncertainty_quartile",
            observed=True
        )
        .agg(
            samples=("error", "size"),
            error_rate=("error", "mean"),
            mean_variance=(
                "prediction_variance",
                "mean"
            ),
            mean_probability=(
                "ensemble_probability",
                "mean"
            ),
        )
        .reset_index()
    )

    uncertainty_error.to_csv(
        OUTPUT_DIR
        / "error_analysis_uncertainty.csv",
        index=False
    )

    # --------------------------------------------------------
    # FALSE POSITIVES / FALSE NEGATIVES
    # --------------------------------------------------------

    fn = out.loc[
        out["error_type"]
        == "false_negative"
    ]

    fp = out.loc[
        out["error_type"]
        == "false_positive"
    ]

    fn.to_csv(
        OUTPUT_DIR
        / "ensemble_false_negatives_operating_threshold.csv",
        index=False
    )

    fp.to_csv(
        OUTPUT_DIR
        / "ensemble_false_positives_operating_threshold.csv",
        index=False
    )

    high_uncertainty_errors = (
        out.loc[
            out["error"] == 1
        ]
        .sort_values(
            "prediction_variance",
            ascending=False
        )
        .head(100)
    )

    high_uncertainty_errors.to_csv(
        OUTPUT_DIR
        / "high_uncertainty_errors_top100.csv",
        index=False
    )

    print(
        "\nERROR_ANALYSIS"
    )

    print(
        "\nProbability-bin analysis:"
    )

    print(
        probability_error.to_string(
            index=False
        )
    )

    print(
        "\nUncertainty-bin analysis:"
    )

    print(
        uncertainty_error.to_string(
            index=False
        )
    )

    print(
        "\nFalse negatives:",
        len(fn),
        "| False positives:",
        len(fp)
    )

    return {
        "probability_error": probability_error,
        "uncertainty_error": uncertainty_error,
        "confusion_matrix": cm_df,
    }


# ============================================================
# ABLATION STUDY
# ============================================================

def ablation_study(
    df
):
    y = df[
        "actual_label"
    ].to_numpy()

    rows = []

    # Individual model baselines.
    for name in [
        "logit",
        "rf",
        "extra_trees",
        "hgb",
        "xgb",
        "catboost",
    ]:

        if name not in df.columns:
            continue

        scores = metric_dict(
            y,
            df[name].to_numpy(),
            threshold=0.50
        )

        rows.append({
            "stage": name,
            **scores,
            "Q1_error_rate": np.nan,
            "Q4_error_rate": np.nan,
            "Q4_minus_Q1_error": np.nan,
            "auto_process_pct": np.nan,
            "manual_review_pct": np.nan,
            "escalate_pct": np.nan,
            "mean_cost": np.nan,
        })

    # Ensemble.
    ep = df[
        "ensemble_probability"
    ].to_numpy()

    ensemble_scores = metric_dict(
        y,
        ep,
        threshold=0.50
    )

    rows.append({
        "stage": "six_model_ensemble",
        **ensemble_scores,
        "Q1_error_rate": np.nan,
        "Q4_error_rate": np.nan,
        "Q4_minus_Q1_error": np.nan,
        "auto_process_pct": np.nan,
        "manual_review_pct": np.nan,
        "escalate_pct": np.nan,
        "mean_cost": np.nan,
    })

    # Ensemble + uncertainty.
    uq = pd.qcut(
        df[
            "prediction_variance"
        ],
        q=4,
        labels=[
            "Q1_low",
            "Q2",
            "Q3",
            "Q4_high",
        ],
        duplicates="drop"
    )

    tmp = pd.DataFrame({
        "uq": uq,
        "correct": (
            (
                (ep >= 0.50)
                .astype(int)
                == y
            )
        ).astype(int),
    })

    g = (
        tmp.groupby(
            "uq",
            observed=True
        )["correct"]
        .mean()
    )

    q1 = (
        float(
            1 - g["Q1_low"]
        )
        if "Q1_low" in g.index
        else np.nan
    )

    q4 = (
        float(
            1 - g["Q4_high"]
        )
        if "Q4_high" in g.index
        else np.nan
    )

    rows.append({
        "stage": "ensemble_plus_uncertainty",
        **ensemble_scores,
        "Q1_error_rate": q1,
        "Q4_error_rate": q4,
        "Q4_minus_Q1_error": (
            q4 - q1
        ),
        "auto_process_pct": np.nan,
        "manual_review_pct": np.nan,
        "escalate_pct": np.nan,
        "mean_cost": np.nan,
    })

    # Routing rows.
    routing_file = (
        OUTPUT_DIR
        / "post_routing_summary.csv"
    )

    if routing_file.exists():

        routing = pd.read_csv(
            routing_file
        )

        for policy in [
            "probability_only",
            "uncertainty_aware",
        ]:

            part = routing[
                routing["policy"]
                == policy
            ]

            def pct(decision):
                z = part.loc[
                    part["decision"]
                    == decision,
                    "percentage"
                ]

                return (
                    float(z.iloc[0])
                    if len(z)
                    else 0.0
                )

            cost = float(
                part[
                    "mean_cost"
                ].iloc[0]
            ) if len(part) else np.nan

            rows.append({
                "stage": (
                    f"{policy}_routing"
                ),
                "AUC": np.nan,
                "PR_AUC": np.nan,
                "Brier": np.nan,
                "Accuracy": np.nan,
                "Precision": np.nan,
                "Recall": np.nan,
                "F1": np.nan,
                "ECE": np.nan,
                "Q1_error_rate": np.nan,
                "Q4_error_rate": np.nan,
                "Q4_minus_Q1_error": np.nan,
                "auto_process_pct": pct(
                    "auto_process"
                ),
                "manual_review_pct": pct(
                    "manual_review"
                ),
                "escalate_pct": pct(
                    "escalate_for_review"
                ),
                "mean_cost": cost,
            })

    out = pd.DataFrame(
        rows
    )

    out.to_csv(
        OUTPUT_DIR
        / "ablation_study.csv",
        index=False
    )

    print(
        "\nABLATION_STUDY"
    )

    print(
        out.to_string(index=False)
    )

    return out


# ============================================================
# MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        exist_ok=True
    )

    pred_file = (
        OUTPUT_DIR
        / "model_predictions.csv"
    )

    if not pred_file.exists():
        raise FileNotFoundError(
            "outputs/model_predictions.csv not found. "
            "Run the main training pipeline first."
        )

    df = pd.read_csv(
        pred_file
    )

    required = [
        "actual_label",
        "ensemble_probability",
        "prediction_variance",
    ]

    missing = [
        c for c in required
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )

    print(
        "POST_ANALYSIS_START"
        f" | rows = {len(df)}"
    )

    # 1. Threshold metrics.
    threshold_analysis(
        df
    )

    # 2. Ablation.
    ablation_study(
        df
    )

    # 3. Calibration.
    calibration_analysis(
        df
    )

    # 4. Bootstrap confidence intervals.
    statistical_validation(
        df
    )

    # 5. Routing summary using paired test rows.
    routing_summary(
        df
    )

    # 6. Paired bootstrap cost comparison.
    paired_cost_bootstrap(
        df,
        n_boot=5000,
        seed=42
    )

    # 7. Error analysis at the proposed
    # operating threshold.
    error_analysis(
        df
    )

    print(
        "\nPOST_ANALYSIS_COMPLETE"
    )

    print(
        "FILES_SAVED",
        str(OUTPUT_DIR)
    )


if __name__ == "__main__":
    main()
