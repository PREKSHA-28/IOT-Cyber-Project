"""
Evaluate cybersecurity evidence retrieved for emerging ML false negatives.

This experiment uses the saved end-to-end RAG results. Ground-truth
attack labels are used ONLY by this evaluation script to determine
whether retrieved evidence belongs to the predefined evidence group
for the actual emerging attack class.

The ground-truth label is never passed into the RAG pipeline.

Emerging attack -> expected evidence groups
--------------------------------------------
VULNERABILITYSCAN -> MITRE T1046 records
DNS_SPOOFING      -> MITRE T1557.001 records
DOS-HTTP_FLOOD    -> MITRE T1499.002 records

The experiment evaluates RAG usefulness for cases where the edge ML
model predicted BENIGN for an actual emerging attack and the sample
was escalated to RAG.
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
    RESULTS_DIR / "end_to_end_rag_results.csv"
)

OUTPUT_FILE = (
    RESULTS_DIR / "emerging_rag_evidence_evaluation.csv"
)


# ============================================================
# Expected evidence groups
# ============================================================

EXPECTED_EVIDENCE = {
    "VULNERABILITYSCAN": frozenset(
        {
            "mitre-t1046-overview",
            "mitre-t1046-indicators",
            "mitre-t1046-mitigation",
            "mitre-t1046-interpretation",
        }
    ),
    "DNS_SPOOFING": frozenset(
        {
            "mitre-t1557-001-overview",
            "mitre-t1557-001-indicators",
            "mitre-t1557-001-mitigation",
            "mitre-t1557-001-caveat",
        }
    ),
    "DOS-HTTP_FLOOD": frozenset(
        {
            "mitre-t1499-002-overview",
            "mitre-t1499-002-indicators",
            "mitre-t1499-002-mitigation",
            "mitre-t1499-002-http",
        }
    ),
}


# ============================================================
# Data loading
# ============================================================

def load_results() -> pd.DataFrame:
    """Load saved end-to-end RAG results."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(
        INPUT_FILE
    )

    required_columns = [
        "sample_index",
        "phase",
        "actual_label",
        "prediction",
        "confidence",
        "routing_reason",
        "retrieved_evidence_ids",
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
# Evidence helpers
# ============================================================

def parse_evidence_ids(
    value: object,
) -> list[str]:
    """Parse pipe-separated evidence IDs from the CSV."""

    if pd.isna(value):
        return []

    text = str(
        value
    ).strip()

    if not text:
        return []

    return [
        item.strip()
        for item in text.split("|")
        if item.strip()
    ]


def calculate_case_metrics(
    actual_label: str,
    evidence_ids: list[str],
) -> dict[str, object]:
    """Calculate retrieval relevance metrics for one sample."""

    expected = EXPECTED_EVIDENCE[
        actual_label
    ]

    retrieved = evidence_ids[:4]

    relevant_ranks = [
        rank
        for rank, evidence_id in enumerate(
            retrieved,
            start=1,
        )
        if evidence_id in expected
    ]

    first_relevant_rank = (
        min(relevant_ranks)
        if relevant_ranks
        else 0
    )

    relevant_retrieved = len(
        relevant_ranks
    )

    retrieved_count = len(
        retrieved
    )

    precision_at_4 = (
        relevant_retrieved
        / retrieved_count
        if retrieved_count
        else 0.0
    )

    recall_at_4 = (
        relevant_retrieved
        / len(expected)
    )

    hit_at_1 = int(
        bool(retrieved)
        and retrieved[0] in expected
    )

    hit_at_4 = int(
        relevant_retrieved > 0
    )

    reciprocal_rank = (
        1.0 / first_relevant_rank
        if first_relevant_rank
        else 0.0
    )

    return {
        "hit_at_1": hit_at_1,
        "hit_at_4": hit_at_4,
        "first_relevant_rank": (
            first_relevant_rank
        ),
        "reciprocal_rank": (
            reciprocal_rank
        ),
        "relevant_items_retrieved": (
            relevant_retrieved
        ),
        "retrieved_items": (
            retrieved_count
        ),
        "precision_at_4": (
            precision_at_4
        ),
        "recall_at_4": (
            recall_at_4
        ),
    }


# ============================================================
# Class-level evaluation
# ============================================================

def evaluate_class(
    subset: pd.DataFrame,
    label: str,
) -> dict[str, object]:
    """Evaluate one emerging attack class."""

    rows = []

    for _, row in subset.iterrows():

        evidence_ids = parse_evidence_ids(
            row["retrieved_evidence_ids"]
        )

        metrics = calculate_case_metrics(
            label,
            evidence_ids,
        )

        rows.append(
            metrics
        )

    case_df = pd.DataFrame(
        rows
    )

    return {
        "actual_label": label,
        "false_negatives_routed_to_rag": (
            len(case_df)
        ),
        "hit_at_1": (
            case_df["hit_at_1"].mean()
        ),
        "hit_at_4": (
            case_df["hit_at_4"].mean()
        ),
        "mrr": (
            case_df["reciprocal_rank"].mean()
        ),
        "precision_at_4": (
            case_df["precision_at_4"].mean()
        ),
        "recall_at_4": (
            case_df["recall_at_4"].mean()
        ),
        "mean_relevant_items_retrieved": (
            case_df[
                "relevant_items_retrieved"
            ].mean()
        ),
        "cases_with_relevant_evidence": (
            int(
                case_df[
                    "hit_at_4"
                ].sum()
            )
        ),
        "relevant_evidence_case_rate_percent": (
            case_df[
                "hit_at_4"
            ].mean()
            * 100.0
        ),
    }


# ============================================================
# Overall evaluation
# ============================================================

def evaluate_overall(
    subset: pd.DataFrame,
) -> tuple[dict[str, object], pd.DataFrame]:
    """Evaluate all emerging false-negative RAG cases."""

    detailed_rows = []

    for _, row in subset.iterrows():

        label = str(
            row["actual_label"]
        ).strip()

        evidence_ids = parse_evidence_ids(
            row["retrieved_evidence_ids"]
        )

        metrics = calculate_case_metrics(
            label,
            evidence_ids,
        )

        detailed_rows.append(
            {
                "sample_index": int(
                    row["sample_index"]
                ),
                "actual_label": label,
                "prediction": str(
                    row["prediction"]
                ),
                "confidence": float(
                    row["confidence"]
                ),
                "routing_reason": str(
                    row["routing_reason"]
                ),
                "retrieved_evidence_ids": (
                    "|".join(
                        evidence_ids[:4]
                    )
                ),
                **metrics,
            }
        )

    detailed_df = pd.DataFrame(
        detailed_rows
    )

    if detailed_df.empty:
        raise RuntimeError(
            "No emerging false negatives were routed to RAG."
        )

    overall = {
        "actual_label": "ALL_EMERGING_FALSE_NEGATIVES",
        "false_negatives_routed_to_rag": (
            len(detailed_df)
        ),
        "hit_at_1": (
            detailed_df["hit_at_1"].mean()
        ),
        "hit_at_4": (
            detailed_df["hit_at_4"].mean()
        ),
        "mrr": (
            detailed_df["reciprocal_rank"].mean()
        ),
        "precision_at_4": (
            detailed_df["precision_at_4"].mean()
        ),
        "recall_at_4": (
            detailed_df["recall_at_4"].mean()
        ),
        "mean_relevant_items_retrieved": (
            detailed_df[
                "relevant_items_retrieved"
            ].mean()
        ),
        "cases_with_relevant_evidence": (
            int(
                detailed_df[
                    "hit_at_4"
                ].sum()
            )
        ),
        "relevant_evidence_case_rate_percent": (
            detailed_df[
                "hit_at_4"
            ].mean()
            * 100.0
        ),
    }

    return overall, detailed_df


# ============================================================
# Main
# ============================================================

def main() -> None:
    print(
        "Loading saved end-to-end RAG results..."
    )

    df = load_results()

    # Only emerging ML false negatives that were actually routed
    # to the RAG layer are evaluated.
    subset = df[
        (
            df["phase"]
            .astype(str)
            .str.upper()
            == "EMERGING"
        )
        & (
            df["prediction"]
            .astype(str)
            .str.upper()
            == "BENIGN"
        )
    ].copy()

    subset["actual_label"] = (
        subset["actual_label"]
        .astype(str)
        .str.strip()
        .str.upper()
    )

    unsupported_labels = sorted(
        set(
            subset["actual_label"]
        )
        - set(
            EXPECTED_EVIDENCE
        )
    )

    if unsupported_labels:
        raise ValueError(
            "Unexpected emerging labels encountered:\n"
            + "\n".join(
                unsupported_labels
            )
        )

    print(
        f"Emerging ML false negatives routed to RAG: "
        f"{len(subset):,}"
    )

    # --------------------------------------------------------
    # Overall
    # --------------------------------------------------------

    overall, detailed_df = evaluate_overall(
        subset
    )

    # --------------------------------------------------------
    # Class-wise
    # --------------------------------------------------------

    class_rows = [
        overall
    ]

    for label in [
        "VULNERABILITYSCAN",
        "DNS_SPOOFING",
        "DOS-HTTP_FLOOD",
    ]:

        class_subset = subset[
            subset["actual_label"]
            == label
        ]

        if class_subset.empty:
            class_rows.append(
                {
                    "actual_label": label,
                    "false_negatives_routed_to_rag": 0,
                    "hit_at_1": 0.0,
                    "hit_at_4": 0.0,
                    "mrr": 0.0,
                    "precision_at_4": 0.0,
                    "recall_at_4": 0.0,
                    "mean_relevant_items_retrieved": 0.0,
                    "cases_with_relevant_evidence": 0,
                    "relevant_evidence_case_rate_percent": 0.0,
                }
            )

            continue

        class_rows.append(
            evaluate_class(
                class_subset,
                label,
            )
        )

    summary_df = pd.DataFrame(
        class_rows
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        "\n" + "=" * 110
    )
    print(
        "EMERGING FALSE-NEGATIVE RAG EVIDENCE EVALUATION"
    )
    print(
        "=" * 110
    )

    print(
        "\nMetrics:"
    )

    display_df = summary_df.copy()

    for column in [
        "hit_at_1",
        "hit_at_4",
        "mrr",
        "precision_at_4",
        "recall_at_4",
    ]:
        display_df[column] = display_df[
            column
        ].map(
            lambda value: round(
                float(value),
                4,
            )
        )

    display_df[
        "relevant_evidence_case_rate_percent"
    ] = display_df[
        "relevant_evidence_case_rate_percent"
    ].map(
        lambda value: round(
            float(value),
            2,
        )
    )

    display_df[
        "mean_relevant_items_retrieved"
    ] = display_df[
        "mean_relevant_items_retrieved"
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
        "INTERPRETATION"
    )
    print(
        "-" * 110
    )

    print(
        "The ground-truth attack label is used only by this evaluator "
        "to determine whether retrieved evidence belongs to the "
        "predefined evidence group."
    )

    print(
        "It is never passed into the RAG request, retriever, or generator."
    )

    print(
        "Hit@4 measures whether at least one retrieved item is relevant "
        "to the actual emerging attack family."
    )

    print(
        "Precision@4 measures the fraction of the four retrieved items "
        "that belong to the relevant evidence group."
    )

    print(
        "Recall@4 measures how much of the predefined relevant evidence "
        "group is retrieved within the top four results."
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_file = (
        RESULTS_DIR
        / "emerging_rag_evidence_summary.csv"
    )

    details_file = (
        RESULTS_DIR
        / "emerging_rag_evidence_details.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False,
    )

    detailed_df.to_csv(
        details_file,
        index=False,
    )

    summary_df.to_csv(
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

    print(
        summary_file
    )

    print(
        details_file
    )


if __name__ == "__main__":
    main()
