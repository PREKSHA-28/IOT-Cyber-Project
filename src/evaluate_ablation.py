"""
Three-system ablation study for the IoT Edge-AI cybersecurity framework.

Systems
-------
A. EDGE_ML_ONLY
   Random Forest prediction is used directly. No RAG escalation.

B. ML_PLUS_CONFIDENCE_RAG
   Low-confidence predictions are escalated to RAG.

C. PROPOSED_CONFIDENCE_PLUS_DRIFT_RAG
   Low-confidence predictions OR drift-affected samples are escalated
   to RAG.

All three systems use the exact same ML predictions and probabilities
from the existing 40,000-sample evaluation. Therefore, differences in
the results are caused by the routing policy, not by retraining or
different test samples.

Important methodological note:
Routing cannot change the Random Forest prediction itself. Therefore,
attack recall, false-positive rate, and raw false-negative counts are
identical across the three systems. The key ablation contribution is
the allocation of uncertain/emerging cases to additional RAG analysis.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"

INPUT_FILE = (
    RESULTS_DIR / "routing_evaluation_results.csv"
)

OUTPUT_FILE = (
    RESULTS_DIR / "ablation_results.csv"
)


# ============================================================
# Configuration
# ============================================================

CONFIDENCE_THRESHOLD = 0.70


# ============================================================
# Data loading
# ============================================================

def load_results() -> pd.DataFrame:
    """Load the common 40,000-sample evaluation results."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Required result file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "dataset",
        "actual_label",
        "prediction",
        "confidence",
        "drift_detected",
        "routing_decision",
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

    return df


# ============================================================
# Policy definitions
# ============================================================

def edge_ml_only_decision(
    confidence: float,
    drift_detected: bool,
) -> str:
    """Baseline: all samples are handled locally."""

    del confidence
    del drift_detected

    return "LOCAL"


def confidence_only_decision(
    confidence: float,
    drift_detected: bool,
) -> str:
    """Selective RAG based only on confidence."""

    del drift_detected

    if confidence < CONFIDENCE_THRESHOLD:
        return "RAG"

    return "LOCAL"


def confidence_plus_drift_decision(
    confidence: float,
    drift_detected: bool,
) -> str:
    """Selective RAG based on confidence or detected drift."""

    if (
        confidence < CONFIDENCE_THRESHOLD
        or drift_detected
    ):
        return "RAG"

    return "LOCAL"


# ============================================================
# Metrics
# ============================================================

def calculate_metrics(
    df: pd.DataFrame,
    decisions: pd.Series,
) -> dict[str, object]:
    """
    Calculate baseline detection and routing metrics.

    Attack recall and FPR describe the underlying ML model and are
    therefore expected to remain unchanged across routing policies.
    """

    total = len(df)

    if total == 0:
        raise ValueError(
            "Cannot evaluate an empty dataset."
        )

    actual_attack = (
        df["actual_label"].astype(str).str.upper()
        != "BENIGN"
    )

    predicted_attack = (
        df["prediction"].astype(str).str.upper()
        == "ATTACK"
    )

    true_positive = int(
        (
            actual_attack
            & predicted_attack
        ).sum()
    )

    false_negative = int(
        (
            actual_attack
            & ~predicted_attack
        ).sum()
    )

    benign = ~actual_attack

    false_positive = int(
        (
            benign
            & predicted_attack
        ).sum()
    )

    attack_count = int(
        actual_attack.sum()
    )

    benign_count = int(
        benign.sum()
    )

    rag_mask = decisions == "RAG"
    local_mask = decisions == "LOCAL"

    emerging_mask = (
        df["dataset"].astype(str).str.upper()
        == "EMERGING"
    )

    emerging_attack_mask = (
        emerging_mask
        & actual_attack
    )

    emerging_false_negative_mask = (
        emerging_attack_mask
        & ~predicted_attack
    )

    emerging_fn_count = int(
        emerging_false_negative_mask.sum()
    )

    emerging_fn_rag = int(
        (
            emerging_false_negative_mask
            & rag_mask
        ).sum()
    )

    emerging_fn_local = int(
        (
            emerging_false_negative_mask
            & local_mask
        ).sum()
    )

    return {
        "total_samples": total,
        "attack_samples": attack_count,
        "benign_samples": benign_count,
        "true_positives": true_positive,
        "false_negatives": false_negative,
        "attack_recall_percent": (
            true_positive
            / attack_count
            * 100.0
            if attack_count
            else 0.0
        ),
        "false_positive_rate_percent": (
            false_positive
            / benign_count
            * 100.0
            if benign_count
            else 0.0
        ),
        "false_positives": false_positive,
        "local_count": int(
            local_mask.sum()
        ),
        "local_percentage": (
            float(local_mask.mean() * 100.0)
        ),
        "rag_count": int(
            rag_mask.sum()
        ),
        "rag_invocation_percentage": (
            float(rag_mask.mean() * 100.0)
        ),
        "emerging_false_negatives": emerging_fn_count,
        "emerging_false_negatives_routed_local": (
            emerging_fn_local
        ),
        "emerging_false_negatives_routed_rag": (
            emerging_fn_rag
        ),
        "false_negative_escalation_rate_percent": (
            emerging_fn_rag
            / emerging_fn_count
            * 100.0
            if emerging_fn_count
            else 0.0
        ),
        "drift_events_observed": int(
            df["drift_detected"].astype(bool).sum()
        ),
        "drift_routed_to_rag": int(
            (
                df["drift_detected"].astype(bool)
                & rag_mask
            ).sum()
        ),
    }


# ============================================================
# Evaluate one system
# ============================================================

def evaluate_system(
    df: pd.DataFrame,
    system_name: str,
    decision_function,
) -> list[dict[str, object]]:
    """Evaluate one routing system overall and by dataset."""

    decisions = df.apply(
        lambda row: decision_function(
            float(row["confidence"]),
            bool(row["drift_detected"]),
        ),
        axis=1,
    )

    rows: list[dict[str, object]] = []

    overall_metrics = calculate_metrics(
        df,
        decisions,
    )

    rows.append(
        {
            "system": system_name,
            "evaluation_scope": "ALL",
            **overall_metrics,
        }
    )

    for dataset_name in [
        "KNOWN",
        "EMERGING",
    ]:

        mask = (
            df["dataset"].astype(str).str.upper()
            == dataset_name
        )

        subset = df.loc[mask].copy()
        subset_decisions = decisions.loc[
            mask
        ]

        dataset_metrics = calculate_metrics(
            subset,
            subset_decisions,
        )

        rows.append(
            {
                "system": system_name,
                "evaluation_scope": dataset_name,
                **dataset_metrics,
            }
        )

    return rows


# ============================================================
# Main
# ============================================================

def main() -> None:
    print(
        "Loading common ablation evaluation results..."
    )

    df = load_results()

    print(
        f"Loaded {len(df):,} samples."
    )

    systems = [
        (
            "EDGE_ML_ONLY",
            edge_ml_only_decision,
        ),
        (
            "ML_PLUS_CONFIDENCE_RAG",
            confidence_only_decision,
        ),
        (
            "PROPOSED_CONFIDENCE_PLUS_DRIFT_RAG",
            confidence_plus_drift_decision,
        ),
    ]

    all_rows: list[dict[str, object]] = []

    for system_name, decision_function in systems:

        all_rows.extend(
            evaluate_system(
                df,
                system_name,
                decision_function,
            )
        )

    results_df = pd.DataFrame(
        all_rows
    )

    # ========================================================
    # Direct incremental contribution
    # ========================================================

    incremental_rows = []

    for scope in [
        "ALL",
        "KNOWN",
        "EMERGING",
    ]:

        scope_df = results_df[
            results_df["evaluation_scope"]
            == scope
        ]

        baseline = scope_df[
            scope_df["system"]
            == "EDGE_ML_ONLY"
        ].iloc[0]

        confidence = scope_df[
            scope_df["system"]
            == "ML_PLUS_CONFIDENCE_RAG"
        ].iloc[0]

        proposed = scope_df[
            scope_df["system"]
            == "PROPOSED_CONFIDENCE_PLUS_DRIFT_RAG"
        ].iloc[0]

        incremental_rows.append(
            {
                "evaluation_scope": scope,
                "confidence_rag_additional_rag_vs_ml_only": (
                    int(
                        confidence["rag_count"]
                    )
                    - int(
                        baseline["rag_count"]
                    )
                ),
                "proposed_additional_rag_vs_ml_only": (
                    int(
                        proposed["rag_count"]
                    )
                    - int(
                        baseline["rag_count"]
                    )
                ),
                "proposed_additional_rag_vs_confidence": (
                    int(
                        proposed["rag_count"]
                    )
                    - int(
                        confidence["rag_count"]
                    )
                ),
                "confidence_fn_escalation_rate_percent": (
                    float(
                        confidence[
                            "false_negative_escalation_rate_percent"
                        ]
                    )
                ),
                "proposed_fn_escalation_rate_percent": (
                    float(
                        proposed[
                            "false_negative_escalation_rate_percent"
                        ]
                    )
                ),
            }
        )

    incremental_df = pd.DataFrame(
        incremental_rows
    )

    # ========================================================
    # Print
    # ========================================================

    print(
        "\n" + "=" * 110
    )
    print(
        "THREE-SYSTEM ABLATION STUDY"
    )
    print(
        "=" * 110
    )

    print(
        f"Confidence threshold: "
        f"{CONFIDENCE_THRESHOLD:.2f}"
    )

    print(
        "\n" + "-" * 110
    )
    print(
        "SYSTEM PERFORMANCE"
    )
    print(
        "-" * 110
    )

    display_columns = [
        "system",
        "evaluation_scope",
        "total_samples",
        "attack_recall_percent",
        "false_positive_rate_percent",
        "false_negatives",
        "rag_count",
        "rag_invocation_percentage",
        "emerging_false_negatives_routed_rag",
        "false_negative_escalation_rate_percent",
    ]

    display_df = results_df[
        display_columns
    ].copy()

    for column in [
        "attack_recall_percent",
        "false_positive_rate_percent",
        "rag_invocation_percentage",
        "false_negative_escalation_rate_percent",
    ]:
        display_df[column] = display_df[
            column
        ].map(
            lambda value: round(
                float(value),
                2,
            )
        )

    print(
        display_df.to_string(
            index=False
        )
    )

    print(
        "\n" + "-" * 110
    )
    print(
        "INCREMENTAL CONTRIBUTION"
    )
    print(
        "-" * 110
    )

    print(
        incremental_df.to_string(
            index=False
        )
    )

    print(
        "\n" + "-" * 110
    )
    print(
        "INTERPRETATION"
    )
    print(
        "-" * 110
    )

    print(
        "1. EDGE_ML_ONLY is the baseline without RAG escalation."
    )

    print(
        "2. ML_PLUS_CONFIDENCE_RAG measures the value of "
        "uncertainty-driven selective escalation."
    )

    print(
        "3. PROPOSED_CONFIDENCE_PLUS_DRIFT_RAG is the complete "
        "architecture including drift-aware routing."
    )

    print(
        "4. ML prediction metrics remain unchanged across systems "
        "because routing does not retrain or alter the Random Forest."
    )

    print(
        "5. The key differences are RAG usage and the proportion "
        "of emerging false negatives receiving additional analysis."
    )

    # ========================================================
    # Save
    # ========================================================

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        "\n" + "=" * 110
    )
    print(
        "RESULTS SAVED"
    )
    print(
        "=" * 110
    )

    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()
