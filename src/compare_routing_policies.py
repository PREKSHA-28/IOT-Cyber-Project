from pathlib import Path

import pandas as pd


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"

INPUT_FILE = (
    RESULTS_DIR / "edge_routing_results.csv"
)

OUTPUT_FILE = (
    RESULTS_DIR / "routing_policy_comparison.csv"
)


# ============================================================
# Configuration
# ============================================================

CONFIDENCE_THRESHOLD = 0.70


# ============================================================
# Load existing integrated results
# ============================================================

def load_results() -> pd.DataFrame:
    """Load the existing sequential edge-routing results."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Required result file not found:\n{INPUT_FILE}"
        )

    return pd.read_csv(INPUT_FILE)


# ============================================================
# Confidence-only policy
# ============================================================

def confidence_only_decision(
    confidence: float,
) -> str:
    """
    Confidence-only routing policy.

    High confidence -> LOCAL
    Low confidence  -> RAG
    """

    if confidence >= CONFIDENCE_THRESHOLD:
        return "LOCAL"

    return "RAG"


# ============================================================
# Confidence + drift policy
# ============================================================

def confidence_and_drift_decision(
    confidence: float,
    drift_detected: bool,
) -> str:
    """
    Confidence + drift routing policy.

    High confidence + no drift -> LOCAL
    Low confidence OR drift    -> RAG
    """

    if (
        confidence >= CONFIDENCE_THRESHOLD
        and not drift_detected
    ):
        return "LOCAL"

    return "RAG"


# ============================================================
# Evaluate one policy
# ============================================================

def evaluate_policy(
    df: pd.DataFrame,
    policy_name: str,
    decisions: pd.Series,
) -> dict:
    """Calculate routing statistics for one policy."""

    total = len(df)

    local_count = int(
        (decisions == "LOCAL").sum()
    )

    rag_count = int(
        (decisions == "RAG").sum()
    )

    drift_count = int(
        df["drift_detected"].sum()
    )

    # --------------------------------------------------------
    # Count samples that would be escalated specifically
    # because of drift.
    #
    # These are high-confidence samples for which drift is
    # true. They would remain LOCAL under confidence-only
    # routing but become RAG under drift-aware routing.
    # --------------------------------------------------------

    drift_only_escalations = int(
        (
            (df["confidence"] >= CONFIDENCE_THRESHOLD)
            & df["drift_detected"]
        ).sum()
    )

    return {
        "policy": policy_name,
        "total_samples": total,
        "local_count": local_count,
        "local_percentage": (
            local_count / total * 100
        ),
        "rag_count": rag_count,
        "rag_invocation_percentage": (
            rag_count / total * 100
        ),
        "drift_events_observed": drift_count,
        "drift_only_escalations": (
            drift_only_escalations
        ),
    }


# ============================================================
# Main comparison
# ============================================================

def main() -> None:

    print("Loading existing edge-routing results...")

    df = load_results()

    required_columns = [
        "confidence",
        "drift_detected",
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

    # --------------------------------------------------------
    # Generate both policy decisions for every sample.
    # --------------------------------------------------------

    confidence_only = df["confidence"].apply(
        confidence_only_decision
    )

    confidence_and_drift = df.apply(
        lambda row: confidence_and_drift_decision(
            row["confidence"],
            bool(row["drift_detected"]),
        ),
        axis=1,
    )

    # --------------------------------------------------------
    # Evaluate overall policies.
    # --------------------------------------------------------

    results = []

    results.append(
        evaluate_policy(
            df,
            "CONFIDENCE_ONLY",
            confidence_only,
        )
    )

    results.append(
        evaluate_policy(
            df,
            "CONFIDENCE_PLUS_DRIFT",
            confidence_and_drift,
        )
    )

    comparison_df = pd.DataFrame(results)

    # --------------------------------------------------------
    # Phase-level comparison.
    # --------------------------------------------------------

    phase_results = []

    for phase in df["phase"].unique():

        phase_df = df[
            df["phase"] == phase
        ].copy()

        phase_confidence_only = confidence_only[
            phase_df.index
        ]

        phase_confidence_and_drift = (
            confidence_and_drift[
                phase_df.index
            ]
        )

        phase_results.append(
            evaluate_policy(
                phase_df,
                f"CONFIDENCE_ONLY_{phase}",
                phase_confidence_only,
            )
        )

        phase_results.append(
            evaluate_policy(
                phase_df,
                f"CONFIDENCE_PLUS_DRIFT_{phase}",
                phase_confidence_and_drift,
            )
        )

    phase_df = pd.DataFrame(
        phase_results
    )

    # --------------------------------------------------------
    # Print comparison.
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print("ROUTING POLICY COMPARISON")
    print("=" * 90)

    print(
        comparison_df.to_string(
            index=False,
            formatters={
                "local_percentage": "{:.2f}".format,
                "rag_invocation_percentage": (
                    "{:.2f}".format
                ),
            },
        )
    )

    print("\n" + "=" * 90)
    print("PHASE-LEVEL COMPARISON")
    print("=" * 90)

    print(
        phase_df.to_string(
            index=False,
            formatters={
                "local_percentage": "{:.2f}".format,
                "rag_invocation_percentage": (
                    "{:.2f}".format
                ),
            },
        )
    )

    # --------------------------------------------------------
    # Direct contribution of drift.
    # --------------------------------------------------------

    additional_rag = int(
        (
            confidence_and_drift
            != confidence_only
        ).sum()
    )

    print("\n" + "=" * 90)
    print("DRIFT CONTRIBUTION")
    print("=" * 90)

    print(
        f"Additional RAG escalations caused by drift: "
        f"{additional_rag:,}"
    )

    # --------------------------------------------------------
    # Save results.
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_df = pd.concat(
        [
            comparison_df.assign(
                comparison_level="OVERALL"
            ),
            phase_df.assign(
                comparison_level="PHASE"
            ),
        ],
        ignore_index=True,
    )

    output_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\n" + "=" * 90)
    print("RESULTS SAVED")
    print("=" * 90)

    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()