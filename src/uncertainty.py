from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score


# ============================================================
# SETTINGS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "outputs"

PRED_FILE = OUTPUT_DIR / "model_predictions.csv"

OPERATING_THRESHOLD = 0.20
N_BOOT = 5000
RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(PRED_FILE)

required = [
    "ensemble_probability",
    "prediction_variance",
    "actual_label",
]

for c in required:
    if c not in df.columns:
        raise ValueError(f"Missing column: {c}")

p = df["ensemble_probability"].to_numpy()
v = df["prediction_variance"].to_numpy()
y = df["actual_label"].to_numpy()

# Binary prediction at the meaningful screening threshold
pred = (p >= OPERATING_THRESHOLD).astype(int)

# 1 = incorrect prediction
error = (pred != y).astype(int)

df["predicted_label"] = pred
df["prediction_error"] = error


# ============================================================
# 1. PROBABILITY BINS
# ============================================================

bins = [0.0, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 1.0]

labels = [
    "0.00-0.10",
    "0.10-0.20",
    "0.20-0.30",
    "0.30-0.40",
    "0.40-0.50",
    "0.50-0.60",
    "0.60-1.00",
]

df["probability_bin"] = pd.cut(
    df["ensemble_probability"],
    bins=bins,
    labels=labels,
    include_lowest=True
)


# ============================================================
# 2. WITHIN-PROBABILITY-BIN UNCERTAINTY ANALYSIS
# ============================================================

rows = []

for prob_bin, g in df.groupby("probability_bin", observed=True):

    if len(g) < 30:
        continue

    # Median split INSIDE each probability bin.
    # This prevents high uncertainty simply meaning
    # "higher predicted probability".
    med = g["prediction_variance"].median()

    low = g[g["prediction_variance"] <= med]
    high = g[g["prediction_variance"] > med]

    low_error = low["prediction_error"].mean()
    high_error = high["prediction_error"].mean()

    rows.append({
        "probability_bin": str(prob_bin),
        "samples": len(g),

        "low_uncertainty_samples": len(low),
        "high_uncertainty_samples": len(high),

        "low_uncertainty_error": low_error,
        "high_uncertainty_error": high_error,

        "error_difference_high_minus_low":
            high_error - low_error,

        "low_mean_variance":
            low["prediction_variance"].mean(),

        "high_mean_variance":
            high["prediction_variance"].mean(),

        "mean_probability":
            g["ensemble_probability"].mean(),
    })


within_bin = pd.DataFrame(rows)

within_bin.to_csv(
    OUTPUT_DIR / "uncertainty_within_probability_bins.csv",
    index=False
)


# ============================================================
# 3. OVERALL LOW/HIGH UNCERTAINTY ANALYSIS
# ============================================================

variance_median = np.median(v)

low_all = df[df["prediction_variance"] <= variance_median]
high_all = df[df["prediction_variance"] > variance_median]

overall = pd.DataFrame([
    {
        "group": "low_uncertainty",
        "samples": len(low_all),
        "error_rate": low_all["prediction_error"].mean(),
        "mean_probability": low_all["ensemble_probability"].mean(),
        "mean_variance": low_all["prediction_variance"].mean(),
    },
    {
        "group": "high_uncertainty",
        "samples": len(high_all),
        "error_rate": high_all["prediction_error"].mean(),
        "mean_probability": high_all["ensemble_probability"].mean(),
        "mean_variance": high_all["prediction_variance"].mean(),
    }
])

overall["error_difference_from_low"] = (
    overall["error_rate"] - overall.iloc[0]["error_rate"]
)

overall.to_csv(
    OUTPUT_DIR / "uncertainty_overall_median_split.csv",
    index=False
)


# ============================================================
# 4. DOES UNCERTAINTY ADD SIGNAL BEYOND PROBABILITY?
#
# Compare:
#   Model A = probability only
#   Model B = probability + uncertainty
#
# Target = whether the prediction was wrong.
#
# If Model B gives meaningfully higher AUC, uncertainty
# contains information about errors beyond probability.
# ============================================================

X_prob = df[["ensemble_probability"]].to_numpy()

# Log-transform variance because variance is heavily skewed.
X_unc = np.log1p(df["prediction_variance"].to_numpy()).reshape(-1, 1)

X_both = np.column_stack([
    df["ensemble_probability"].to_numpy(),
    X_unc.ravel()
])

y_error = df["prediction_error"].to_numpy()


model_prob = LogisticRegression(
    max_iter=2000,
    random_state=RANDOM_STATE
)

model_both = LogisticRegression(
    max_iter=2000,
    random_state=RANDOM_STATE
)

model_prob.fit(X_prob, y_error)
model_both.fit(X_both, y_error)

prob_error_score = model_prob.predict_proba(X_prob)[:, 1]
both_error_score = model_both.predict_proba(X_both)[:, 1]

auc_prob = roc_auc_score(y_error, prob_error_score)
auc_both = roc_auc_score(y_error, both_error_score)

model_comparison = pd.DataFrame([
    {
        "model": "probability_only",
        "features": "ensemble_probability",
        "error_prediction_auc": auc_prob,
    },
    {
        "model": "probability_plus_uncertainty",
        "features": "ensemble_probability + log1p(variance)",
        "error_prediction_auc": auc_both,
    }
])

model_comparison["auc_improvement"] = (
    auc_both - auc_prob
)

model_comparison.to_csv(
    OUTPUT_DIR / "uncertainty_incremental_signal.csv",
    index=False
)


# ============================================================
# 5. BOOTSTRAP WITHIN-BIN DIFFERENCES
# ============================================================

rng = np.random.default_rng(RANDOM_STATE)

boot_rows = []

for _, row in within_bin.iterrows():

    name = row["probability_bin"]

    g = df[df["probability_bin"].astype(str) == name].copy()

    if len(g) < 30:
        continue

    med = g["prediction_variance"].median()

    low = g[g["prediction_variance"] <= med]["prediction_error"].to_numpy()
    high = g[g["prediction_variance"] > med]["prediction_error"].to_numpy()

    diffs = []

    for _ in range(N_BOOT):

        low_sample = rng.choice(
            low,
            size=len(low),
            replace=True
        )

        high_sample = rng.choice(
            high,
            size=len(high),
            replace=True
        )

        diffs.append(
            high_sample.mean() - low_sample.mean()
        )

    diffs = np.array(diffs)

    boot_rows.append({
        "probability_bin": name,
        "observed_difference": high.mean() - low.mean(),
        "bootstrap_ci_low": np.percentile(diffs, 2.5),
        "bootstrap_ci_high": np.percentile(diffs, 97.5),
    })

bootstrap = pd.DataFrame(boot_rows)

bootstrap.to_csv(
    OUTPUT_DIR / "uncertainty_within_bin_bootstrap.csv",
    index=False
)


# ============================================================
# 6. PRINT RESULTS
# ============================================================

print("\n" + "=" * 70)
print("UNCERTAINTY ABLATION ANALYSIS")
print("=" * 70)

print("\nDataset:")
print(f"Test samples: {len(df)}")
print(f"Operating threshold: {OPERATING_THRESHOLD}")

print("\n" + "-" * 70)
print("WITHIN-PROBABILITY-BIN UNCERTAINTY ANALYSIS")
print("-" * 70)

if len(within_bin):
    print(
        within_bin[
            [
                "probability_bin",
                "samples",
                "low_uncertainty_error",
                "high_uncertainty_error",
                "error_difference_high_minus_low"
            ]
        ].to_string(index=False)
    )
else:
    print("No sufficiently large probability bins found.")


print("\n" + "-" * 70)
print("BOOTSTRAP CONFIDENCE INTERVALS")
print("-" * 70)

if len(bootstrap):
    print(bootstrap.to_string(index=False))
else:
    print("No bootstrap results.")


print("\n" + "-" * 70)
print("OVERALL MEDIAN UNCERTAINTY SPLIT")
print("-" * 70)

print(overall.to_string(index=False))


print("\n" + "-" * 70)
print("INCREMENTAL SIGNAL: DOES UNCERTAINTY ADD INFORMATION?")
print("-" * 70)

print(model_comparison.to_string(index=False))

print("\n" + "=" * 70)
print("FILES SAVED")
print("=" * 70)

print(OUTPUT_DIR / "uncertainty_within_probability_bins.csv")
print(OUTPUT_DIR / "uncertainty_overall_median_split.csv")
print(OUTPUT_DIR / "uncertainty_incremental_signal.csv")
print(OUTPUT_DIR / "uncertainty_within_bin_bootstrap.csv")

print("\nUNCERTAINTY_ABLATION_COMPLETE")