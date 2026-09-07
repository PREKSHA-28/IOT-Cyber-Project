from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"

INPUT_FILE = RESULTS_DIR / "routing_evaluation_results.csv"

OUTPUT_FILE = RESULTS_DIR / "adaptive_threshold_results.csv"


# ============================================================
# Configuration
# ============================================================

THRESHOLDS = [
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90,
]


# ============================================================
# Load existing results
# ============================================================

def load_results() -> pd.DataFrame:
    """Load Person 1's existing routing evaluation results."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Required result file not found:\n{INPUT_FILE}"
        )

    return pd.read_csv(INPUT_FILE)


# ============================================================
# Classification metrics
# ============================================================

def calculate_classification_metrics(
    df: pd.DataFrame,
) -> dict:
    """
    Calculate model-level binary classification metrics.

    These metrics describe the Random Forest predictions.
    They are independent of the routing threshold because
    changing the confidence threshold does not change the
    model prediction.
    """

    actual_attack = (
        df["actual_label"]
        .astype(str)
        .str.upper()
        .ne("BENIGN")
    )

    predicted_attack = (
        df["prediction"]
        .astype(str)
        .str.upper()
        == "ATTACK"
    )

    true_positive = int(
        np.sum(actual_attack & predicted_attack)
    )

    false_negative = int(
        np.sum(actual_attack & ~predicted_attack)
    )

    false_positive = int(
        np.sum(~actual_attack & predicted_attack)
    )

    true_negative = int(
        np.sum(~actual_attack & ~predicted_attack)
    )

    attack_recall = (
        true_positive
        / (true_positive + false_negative)
        if (true_positive + false_negative) > 0
        else 0.0
    )

    false_positive_rate = (
        false_positive
        / (false_positive + true_negative)
        if (false_positive + true_negative) > 0
        else 0.0
    )

    return {
        "attack_recall_percent": attack_recall * 100,
        "false_positive_rate_percent": (
            false_positive_rate * 100
        ),
        "true_positive": true_positive,
        "false_negative": false_negative,
        "false_positive": false_positive,
        "true_negative": true_negative,
    }


# ============================================================
# Evaluate one threshold
# ============================================================

def evaluate_threshold(
    df: pd.DataFrame,
    threshold: float,
) -> dict:
    """
    Evaluate routing behavior at one confidence threshold.

    High confidence:
        confidence >= threshold

    Low confidence:
        confidence < threshold

    Low-confidence traffic is escalated to RAG.
    """

    # --------------------------------------------------------
    # Routing masks
    # --------------------------------------------------------

    rag_mask = df["confidence"] < threshold

    local_mask = ~rag_mask

    total = len(df)

    rag_count = int(rag_mask.sum())
    local_count = int(local_mask.sum())

    # --------------------------------------------------------
    # Model-level classification metrics
    # --------------------------------------------------------

    metrics = calculate_classification_metrics(df)

    # --------------------------------------------------------
    # Actual attacks
    # --------------------------------------------------------

    actual_attack = (
        df["actual_label"]
        .astype(str)
        .str.upper()
        .ne("BENIGN")
    )

    predicted_attack = (
        df["prediction"]
        .astype(str)
        .str.upper()
        == "ATTACK"
    )

    # --------------------------------------------------------
    # Emerging-phase analysis
    # --------------------------------------------------------

    emerging_mask = (
        df["dataset"]
        .astype(str)
        .str.upper()
        == "EMERGING"
    )

    emerging_df = df[emerging_mask].copy()

    emerging_actual_attack = (
        emerging_df["actual_label"]
        .astype(str)
        .str.upper()
        .ne("BENIGN")
    )

    emerging_predicted_attack = (
        emerging_df["prediction"]
        .astype(str)
        .str.upper()
        == "ATTACK"
    )

    emerging_rag_mask = (
        emerging_df["confidence"] < threshold
    )

    emerging_local_mask = ~emerging_rag_mask

    emerging_attack_count = int(
        emerging_actual_attack.sum()
    )

    # --------------------------------------------------------
    # Model false negatives in emerging traffic.
    #
    # This is a property of the ML model, NOT routing.
    # --------------------------------------------------------

    emerging_model_false_negatives = int(
        np.sum(
            emerging_actual_attack
            & ~emerging_predicted_attack
        )
    )

    # --------------------------------------------------------
    # Correctly detected emerging attacks routed LOCAL.
    #
    # These are attacks the model identified, but the edge
    # layer decided they were sufficiently confident to keep
    # local.
    # --------------------------------------------------------

    emerging_attacks_routed_local = int(
        np.sum(
            emerging_actual_attack
            & emerging_predicted_attack
            & emerging_local_mask
        )
    )

    # --------------------------------------------------------
    # Correctly detected emerging attacks escalated to RAG.
    # --------------------------------------------------------

    emerging_attacks_routed_rag = int(
        np.sum(
            emerging_actual_attack
            & emerging_predicted_attack
            & emerging_rag_mask
        )
    )

    # --------------------------------------------------------
    # Emerging attack escalation rate.
    # --------------------------------------------------------

    emerging_attack_escalation_rate = (
        emerging_attacks_routed_rag
        / emerging_attack_count * 100
        if emerging_attack_count > 0
        else 0.0
    )

    # --------------------------------------------------------
    # High-confidence emerging attacks routed LOCAL.
    # --------------------------------------------------------

    emerging_high_confidence_attacks_local = int(
        np.sum(
            emerging_actual_attack
            & emerging_local_mask
        )
    )

    return {
        "threshold": threshold,

        "total_samples": total,

        "local_count": local_count,

        "local_percentage": (
            local_count / total * 100
            if total > 0
            else 0.0
        ),

        "rag_count": rag_count,

        "rag_invocation_percentage": (
            rag_count / total * 100
            if total > 0
            else 0.0
        ),

        "attack_recall_percent": (
            metrics["attack_recall_percent"]
        ),

        "false_positive_rate_percent": (
            metrics["false_positive_rate_percent"]
        ),

        # Model-level false negatives.
        "emerging_model_false_negatives": (
            emerging_model_false_negatives
        ),

        # Correctly detected attacks that remained local.
        "emerging_attacks_routed_local": (
            emerging_attacks_routed_local
        ),

        # Correctly detected attacks escalated to RAG.
        "emerging_attacks_routed_rag": (
            emerging_attacks_routed_rag
        ),

        "emerging_attack_escalation_rate_percent": (
            emerging_attack_escalation_rate
        ),

        "emerging_high_confidence_attacks_local": (
            emerging_high_confidence_attacks_local
        ),

        "emerging_attack_count": (
            emerging_attack_count
        ),

        "mean_confidence": (
            df["confidence"].mean()
        ),
    }


# ============================================================
# Evaluate thresholds by dataset
# ============================================================

def run_threshold_evaluation(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Evaluate every threshold for each traffic phase."""

    records = []

    for dataset_name in [
        "KNOWN",
        "EMERGING",
        "ALL",
    ]:

        if dataset_name == "ALL":

            subset = df.copy()

        else:

            subset = df[
                df["dataset"]
                .astype(str)
                .str.upper()
                == dataset_name
            ].copy()

        if subset.empty:
            continue

        for threshold in THRESHOLDS:

            result = evaluate_threshold(
                subset,
                threshold,
            )

            result["dataset"] = dataset_name

            records.append(result)

    return pd.DataFrame(records)


# ============================================================
# Print results
# ============================================================

def print_results(
    results: pd.DataFrame,
) -> None:

    print("\n" + "=" * 110)
    print("ADAPTIVE CONFIDENCE THRESHOLD EVALUATION")
    print("=" * 110)

    display_columns = [
        "dataset",
        "threshold",
        "local_percentage",
        "rag_invocation_percentage",
        "attack_recall_percent",
        "false_positive_rate_percent",
        "emerging_model_false_negatives",
        "emerging_attacks_routed_local",
        "emerging_attacks_routed_rag",
        "emerging_attack_escalation_rate_percent",
    ]

    print(
        results[display_columns].to_string(
            index=False,
            formatters={
                "threshold": "{:.2f}".format,
                "local_percentage": "{:.2f}".format,
                "rag_invocation_percentage": "{:.2f}".format,
                "attack_recall_percent": "{:.2f}".format,
                "false_positive_rate_percent": "{:.2f}".format,
                "emerging_attack_escalation_rate_percent": (
                    "{:.2f}".format
                ),
            },
        )
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    print("Loading existing routing results...")

    df = load_results()

    required_columns = [
        "dataset",
        "actual_label",
        "prediction",
        "confidence",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Required columns are missing:\n"
            + "\n".join(missing_columns)
        )

    print(
        f"Loaded {len(df):,} samples."
    )

    results = run_threshold_evaluation(df)

    print_results(results)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\n" + "=" * 110)
    print("RESULTS SAVED")
    print("=" * 110)

    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()