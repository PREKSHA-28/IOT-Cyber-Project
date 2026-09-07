import pandas as pd
from pathlib import Path


RESULTS_DIR = Path("results")

THRESHOLD_FILE = RESULTS_DIR / "adaptive_threshold_results.csv"
FN_FILE = RESULTS_DIR / "emerging_false_negative_escalation.csv"
OUTPUT_FILE = RESULTS_DIR / "threshold_tradeoff.csv"


# Load existing threshold results
threshold_df = pd.read_csv(THRESHOLD_FILE)

# Load emerging false-negative escalation results
fn_df = pd.read_csv(FN_FILE)


# Keep overall routing cost information
overall = threshold_df[
    threshold_df["dataset"] == "ALL"
][[
    "threshold",
    "local_percentage",
    "rag_invocation_percentage"
]].copy()


# Keep emerging false-negative information
fn = fn_df[[
    "threshold",
    "emerging_model_false_negatives",
    "false_negatives_routed_rag",
    "false_negative_escalation_rate_percent"
]].copy()


# Combine the two result tables
tradeoff = overall.merge(
    fn,
    on="threshold",
    how="inner"
)


# Save the final trade-off table
tradeoff.to_csv(OUTPUT_FILE, index=False)


print("\n" + "=" * 90)
print("                 THRESHOLD TRADE-OFF ANALYSIS")
print("=" * 90)

print("\nResults:")
print(tradeoff.to_string(index=False))

print("=" * 90)
print(f"Saved to: {OUTPUT_FILE}")
print("=" * 90)