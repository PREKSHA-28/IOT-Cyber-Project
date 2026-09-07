import pandas as pd
from pathlib import Path


RESULTS_FILE = Path("results/edge_routing_results.csv")
OUTPUT_FILE = Path("results/emerging_false_negative_escalation.csv")

THRESHOLDS = [0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]


# Load the actual edge routing results
df = pd.read_csv(RESULTS_FILE)


# Emerging attacks incorrectly predicted as BENIGN
emerging_fn = df[
    (df["phase"] == "EMERGING") &
    (df["actual_label"] != "BENIGN") &
    (df["prediction"] == "BENIGN")
].copy()


total_fn = len(emerging_fn)

results = []

for threshold in THRESHOLDS:

    routed_rag = (
        emerging_fn["confidence"] < threshold
    ).sum()

    routed_local = (
        emerging_fn["confidence"] >= threshold
    ).sum()

    escalation_rate = (
        routed_rag / total_fn * 100
        if total_fn > 0
        else 0
    )

    results.append({
        "threshold": threshold,
        "emerging_model_false_negatives": total_fn,
        "false_negatives_routed_local": routed_local,
        "false_negatives_routed_rag": routed_rag,
        "false_negative_escalation_rate_percent": escalation_rate,
    })


summary = pd.DataFrame(results)

summary.to_csv(OUTPUT_FILE, index=False)


print("\n" + "=" * 85)
print("       EMERGING FALSE-NEGATIVE ESCALATION ANALYSIS")
print("=" * 85)

print(f"Total emerging model false negatives: {total_fn}")

print("\nResults:")
print(summary.to_string(index=False))

print("=" * 85)
print(f"Saved to: {OUTPUT_FILE}")
print("=" * 85)