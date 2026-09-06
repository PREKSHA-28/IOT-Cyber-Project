from pathlib import Path

import pandas as pd

from orchestrator import EdgeOrchestrator


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    PROJECT_ROOT
    / "results"
    / "edge_routing_results.csv"
)


# ============================================================
# Configuration
# ============================================================

CONFIDENCE_THRESHOLD = 0.70


# ============================================================
# Main
# ============================================================

def main() -> None:

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Results file not found:\n{INPUT_FILE}"
        )

    print("Loading existing edge-routing results...")

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "prediction",
        "raw_attack_probability",
        "calibrated_attack_probability",
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
            f"Missing required columns: {missing_columns}"
        )

    orchestrator = EdgeOrchestrator(
        confidence_threshold=CONFIDENCE_THRESHOLD
    )

    # --------------------------------------------------------
    # Process each sample sequentially.
    #
    # This deliberately mimics a streaming interface:
    # one Edge-AI result enters the orchestrator at a time.
    # --------------------------------------------------------

    orchestration_results = []

    for _, row in df.iterrows():

        result = orchestrator.decide(
            prediction=row["prediction"],
            raw_probability=float(
                row["raw_attack_probability"]
            ),
            calibrated_probability=float(
                row["calibrated_attack_probability"]
            ),
            confidence=float(
                row["confidence"]
            ),
            drift_detected=bool(
                row["drift_detected"]
            ),
        )

        orchestration_results.append(
            result.to_dict()
        )

    orchestration_df = pd.DataFrame(
        orchestration_results
    )

    # --------------------------------------------------------
    # Compare new orchestrator decisions with the
    # existing Person 1 routing decisions.
    # --------------------------------------------------------

    comparison = pd.DataFrame(
        {
            "existing_routing": df[
                "routing_decision"
            ].values,
            "orchestrator_routing": orchestration_df[
                "routing_decision"
            ].values,
            "routing_reason": orchestration_df[
                "routing_reason"
            ].values,
        }
    )

    differences = comparison[
        comparison["existing_routing"]
        != comparison["orchestrator_routing"]
    ]

    print("\n" + "=" * 70)
    print("ORCHESTRATOR INTEGRATION TEST")
    print("=" * 70)

    print(
        f"Total samples processed : {len(df):,}"
    )

    print(
        f"Routing differences     : {len(differences):,}"
    )

    print("\nRouting decisions:")

    print(
        orchestration_df[
            "routing_decision"
        ].value_counts()
    )

    print("\nRouting reasons:")

    print(
        orchestration_df[
            "routing_reason"
        ].value_counts()
    )

    # --------------------------------------------------------
    # Verify that the new orchestrator reproduces the
    # existing routing behavior.
    # --------------------------------------------------------

    if len(differences) == 0:

        print(
            "\nPASS: New orchestrator reproduces "
            "the existing routing decisions."
        )

    else:

        print(
            "\nWARNING: Routing differences detected."
        )

        print(
            differences.head(20).to_string(
                index=False
            )
        )


if __name__ == "__main__":
    main()