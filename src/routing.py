from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"

LOW_PROB_THRESHOLD = 0.20
HIGH_PROB_THRESHOLD = 0.575
VARIANCE_THRESHOLD = 0.00312741430937158


def make_probability_tiers(p):
    d = np.full(len(p), "manual_review", dtype=object)
    d[p < LOW_PROB_THRESHOLD] = "auto_process"
    d[p >= HIGH_PROB_THRESHOLD] = "escalate_for_review"
    return d


def make_uncertainty_tiers(p, variance):
    d = np.full(len(p), "manual_review", dtype=object)
    confident = variance < VARIANCE_THRESHOLD
    d[confident & (p < LOW_PROB_THRESHOLD)] = "auto_process"
    d[confident & (p >= HIGH_PROB_THRESHOLD)] = "escalate_for_review"
    return d


def summarize_policy(policy_name, decisions, y):
    y = np.asarray(y).astype(int)
    decisions = np.asarray(decisions)
    total = len(y)
    total_defaults = int(y.sum())
    rows = []

    for decision in ["auto_process", "manual_review", "escalate_for_review"]:
        mask = decisions == decision
        n = int(mask.sum())
        defaults = int(y[mask].sum())
        rows.append({
            "policy": policy_name,
            "decision": decision,
            "samples": n,
            "percentage": n / total * 100,
            "defaults_in_tier": defaults,
            "default_rate_in_tier": float(y[mask].mean()) if n else 0.0,
            "share_of_all_defaults": defaults / total_defaults * 100 if total_defaults else 0.0,
        })

    return pd.DataFrame(rows)


def main():
    pred_file = OUTPUT_DIR / "model_predictions.csv"

    if not pred_file.exists():
        raise FileNotFoundError(
            "outputs/model_predictions.csv was not found."
        )

    df = pd.read_csv(pred_file)

    required = [
        "actual_label",
        "ensemble_probability",
        "prediction_variance",
    ]

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(f"Missing columns: {missing}")

    y = df["actual_label"].to_numpy()
    p = df["ensemble_probability"].to_numpy()
    variance = df["prediction_variance"].to_numpy()

    probability_decisions = make_probability_tiers(p)
    uncertainty_decisions = make_uncertainty_tiers(p, variance)

    probability_summary = summarize_policy(
        "probability_only",
        probability_decisions,
        y,
    )

    uncertainty_summary = summarize_policy(
        "uncertainty_aware",
        uncertainty_decisions,
        y,
    )

    summary = pd.concat(
        [probability_summary, uncertainty_summary],
        ignore_index=True,
    )

    summary.to_csv(
        OUTPUT_DIR / "three_tier_routing_analysis.csv",
        index=False,
    )

    detail = pd.DataFrame({
        "actual_label": y,
        "ensemble_probability": p,
        "prediction_variance": variance,
        "probability_only_decision": probability_decisions,
        "uncertainty_aware_decision": uncertainty_decisions,
    })

    detail.to_csv(
        OUTPUT_DIR / "three_tier_routing_details.csv",
        index=False,
    )

    comparison = (
        probability_summary[
            [
                "decision",
                "samples",
                "percentage",
                "defaults_in_tier",
                "default_rate_in_tier",
                "share_of_all_defaults",
            ]
        ]
        .rename(columns={
            "samples": "prob_samples",
            "percentage": "prob_percentage",
            "defaults_in_tier": "prob_defaults",
            "default_rate_in_tier": "prob_default_rate",
            "share_of_all_defaults": "prob_default_capture",
        })
        .merge(
            uncertainty_summary[
                [
                    "decision",
                    "samples",
                    "percentage",
                    "defaults_in_tier",
                    "default_rate_in_tier",
                    "share_of_all_defaults",
                ]
            ].rename(columns={
                "samples": "unc_samples",
                "percentage": "unc_percentage",
                "defaults_in_tier": "unc_defaults",
                "default_rate_in_tier": "unc_default_rate",
                "share_of_all_defaults": "unc_default_capture",
            }),
            on="decision",
            how="outer",
        )
    )

    comparison.to_csv(
        OUTPUT_DIR / "three_tier_routing_comparison.csv",
        index=False,
    )

    print("\nTHREE_TIER_ROUTING_ANALYSIS")
    print("\nThresholds:")
    print("LOW_PROB_THRESHOLD =", LOW_PROB_THRESHOLD)
    print("HIGH_PROB_THRESHOLD =", HIGH_PROB_THRESHOLD)
    print("VARIANCE_THRESHOLD =", VARIANCE_THRESHOLD)

    print("\nProbability-only routing:")
    print(probability_summary.to_string(index=False))

    print("\nUncertainty-aware routing:")
    print(uncertainty_summary.to_string(index=False))

    print("\nTHREE_TIER_ROUTING_COMPARISON")
    print(comparison.to_string(index=False))

    print("\nTOTAL_TEST_SAMPLES =", len(y))
    print("TOTAL_ACTUAL_DEFAULTS =", int(y.sum()))
    print("\nFILES_SAVED =", str(OUTPUT_DIR))
    print("\nROUTING_ANALYSIS_COMPLETE")


if __name__ == "__main__":
    main()
