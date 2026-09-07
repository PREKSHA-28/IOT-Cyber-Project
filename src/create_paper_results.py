import pandas as pd
from pathlib import Path


# ============================================================
# Project paths
# ============================================================

RESULTS_DIR = Path("results")

INPUT_FILE = RESULTS_DIR / "adaptive_threshold_results.csv"
OUTPUT_FILE = RESULTS_DIR / "paper_results_summary.csv"


# ============================================================
# Load threshold evaluation results
# ============================================================

df = pd.read_csv(INPUT_FILE)


# ============================================================
# Select important paper-level metrics
# ============================================================
#
# The false-negative escalation metrics are included explicitly
# because they measure how many emerging attacks missed by the
# edge ML model would still be escalated to RAG.
#
# IMPORTANT:
# These metrics measure routing/escalation, not successful
# recovery by the RAG/LLM stage.
# ============================================================

summary_columns = [
    "dataset",
    "threshold",

    # Overall routing
    "total_samples",
    "local_count",
    "local_percentage",
    "rag_count",
    "rag_invocation_percentage",

    # Model-level classification
    "attack_recall_percent",
    "false_positive_rate_percent",

    # Emerging attack behavior
    "emerging_model_false_negatives",

    # Emerging ML false-negative escalation
    "emerging_false_negatives_routed_local",
    "emerging_false_negatives_routed_rag",
    "false_negative_escalation_rate_percent",

    # Correctly detected emerging attacks
    "emerging_attacks_routed_local",
    "emerging_attacks_routed_rag",
    "emerging_attack_escalation_rate_percent",

    # Confidence
    "mean_confidence",
]


# ============================================================
# Verify required columns
# ============================================================

missing_columns = [
    column
    for column in summary_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing columns in input file: {missing_columns}"
    )


# ============================================================
# Create clean paper-oriented summary
# ============================================================

summary = df[summary_columns].copy()


# ============================================================
# Save paper-oriented results
# ============================================================

summary.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# Print results
# ============================================================

print("\n" + "=" * 100)
print("                    PAPER RESULTS SUMMARY")
print("=" * 100)

print(f"Input file  : {INPUT_FILE}")
print(f"Output file : {OUTPUT_FILE}")
print(f"Rows        : {len(summary)}")

print("\nResults:")

print(
    summary.to_string(
        index=False,
        formatters={
            "threshold": "{:.2f}".format,
            "local_percentage": "{:.2f}".format,
            "rag_invocation_percentage": "{:.2f}".format,
            "attack_recall_percent": "{:.2f}".format,
            "false_positive_rate_percent": "{:.2f}".format,
            "false_negative_escalation_rate_percent": (
                "{:.2f}".format
            ),
            "emerging_attack_escalation_rate_percent": (
                "{:.2f}".format
            ),
            "mean_confidence": "{:.6f}".format,
        },
    )
)

print("=" * 100)
print("Saved successfully.")