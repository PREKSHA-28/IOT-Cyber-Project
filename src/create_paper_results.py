import pandas as pd
from pathlib import Path


RESULTS_DIR = Path("results")

INPUT_FILE = RESULTS_DIR / "adaptive_threshold_results.csv"
OUTPUT_FILE = RESULTS_DIR / "paper_results_summary.csv"


# Load existing threshold results
df = pd.read_csv(INPUT_FILE)


# Select the important paper-level metrics
summary_columns = [
    "dataset",
    "threshold",
    "total_samples",
    "local_count",
    "local_percentage",
    "rag_count",
    "rag_invocation_percentage",
    "attack_recall_percent",
    "false_positive_rate_percent",
    "emerging_model_false_negatives",
    "emerging_attacks_routed_local",
    "emerging_attacks_routed_rag",
    "mean_confidence",
]


# Verify required columns
missing_columns = [
    column for column in summary_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in input file: {missing_columns}"
    )


# Create clean summary
summary = df[summary_columns].copy()


# Save paper-oriented results
summary.to_csv(OUTPUT_FILE, index=False)


print("\n" + "=" * 70)
print("              PAPER RESULTS SUMMARY")
print("=" * 70)

print(f"Input file  : {INPUT_FILE}")
print(f"Output file : {OUTPUT_FILE}")
print(f"Rows        : {len(summary)}")

print("\nResults:")
print(summary.to_string(index=False))

print("=" * 70)
print("Saved successfully.")