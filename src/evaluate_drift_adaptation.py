"""
Evaluate the incremental routing contribution of drift awareness.

This experiment uses the already-generated sequential edge-routing results
so that the underlying ML predictions remain identical across policies.

Policies
--------
1. CONFIDENCE_ONLY:
   confidence < threshold -> RAG
   otherwise -> LOCAL

2. CONFIDENCE_PLUS_DRIFT:
   confidence < threshold OR drift_detected -> RAG
   otherwise -> LOCAL

The experiment focuses on routing/adaptation rather than ML accuracy:
drift does not change the Random Forest prediction itself. It changes
whether a sample receives additional RAG analysis.

The detected drift point is taken from the existing edge-routing results.
A symmetric pre-drift and post-drift window is evaluated around that point.
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
    RESULTS_DIR / "edge_routing_results.csv"
)

OUTPUT_FILE = (
    RESULTS_DIR / "drift_adaptation_evaluation.csv"
)


# ============================================================
# Configuration
# ============================================================

CONFIDENCE_THRESHOLD = 0.70

# Symmetric evaluation window around the first detected drift event.
WINDOW_SIZE = 5_000


# ============================================================
# Data loading
# ============================================================

def load_results() -> pd.DataFrame:
    """Load the sequential edge-routing experiment results."""

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Required result file not found:\n{INPUT_FILE}"
        )

    df = pd.read_csv(INPUT_FILE)

    required_columns = [
        "sample_index",
        "phase",
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

def confidence_only_decision(
    confidence: float,
) -> str:
    """Route using prediction confidence only."""

    if confidence < CONFIDENCE_THRESHOLD:
        return "RAG"

    return "LOCAL"


def confidence_plus_drift_decision(
    confidence: float,
    drift_detected: bool,
) -> str:
    """Route using both confidence and drift information."""

    if (
        confidence < CONFIDENCE_THRESHOLD
        or drift_detected
    ):
        return "RAG"

    return "LOCAL"


# ============================================================
# Utility metrics
# ============================================================

def is_attack(
    actual_label: pd.Series,
) -> pd.Series:
    """
    Return a boolean attack mask.

    BENIGN is treated as the benign class; every other dataset
    label is treated as an attack class.
    """

    return (
        actual_label.astype(str).str.upper()
        != "BENIGN"
    )


def calculate_policy_metrics(
    df: pd.DataFrame,
    decisions: pd.Series,
) -> dict[str, float | int]:
    """
    Calculate routing and emerging/attack-risk metrics.

    The ML predictions remain unchanged. These metrics therefore
    describe how the routing policy allocates additional analysis.
    """

    total = len(df)

    if total == 0:
        raise ValueError(
            "Cannot calculate metrics for an empty window."
        )

    attack_mask = is_attack(
        df["actual_label"]
    )

    false_negative_mask = (
        attack_mask
        & (
            df["prediction"].astype(str).str.upper()
            == "BENIGN"
        )
    )

    local_mask = decisions == "LOCAL"
    rag_mask = decisions == "RAG"

    attack_count = int(
        attack_mask.sum()
    )

    false_negative_count = int(
        false_negative_mask.sum()
    )

    fn_routed_rag = int(
        (
            false_negative_mask
            & rag_mask
        ).sum()
    )

    fn_routed_local = int(
        (
            false_negative_mask
            & local_mask
        ).sum()
    )

    drift_mask = (
        df["drift_detected"]
        .astype(bool)
    )

    high_confidence_drift = (
        (
            df["confidence"]
            >= CONFIDENCE_THRESHOLD
        )
        & drift_mask
    )

    drift_only_escalations = int(
        high_confidence_drift.sum()
    )

    return {
        "total_samples": total,
        "attack_samples": attack_count,
        "false_negatives": false_negative_count,
        "local_count": int(
            local_mask.sum()
        ),
        "rag_count": int(
            rag_mask.sum()
        ),
        "rag_invocation_rate_percent": (
            float(rag_mask.mean() * 100.0)
        ),
        "false_negative_escalated_to_rag": fn_routed_rag,
        "false_negative_left_local": fn_routed_local,
        "false_negative_escalation_rate_percent": (
            (
                fn_routed_rag
                / false_negative_count
                * 100.0
            )
            if false_negative_count
            else 0.0
        ),
        "drift_only_escalations": (
            drift_only_escalations
        ),
        "mean_confidence": float(
            df["confidence"].mean()
        ),
        "drift_events": int(
            drift_mask.sum()
        ),
    }


# ============================================================
# Window construction
# ============================================================

def find_first_drift_sample(
    df: pd.DataFrame,
) -> int:
    """Return the first sample index where ADWIN reported drift."""

    drift_rows = df[
        df["drift_detected"].astype(bool)
    ]

    if drift_rows.empty:
        raise RuntimeError(
            "No drift event was found in edge_routing_results.csv."
        )

    return int(
        drift_rows.iloc[0]["sample_index"]
    )


def build_windows(
    df: pd.DataFrame,
    drift_sample: int,
) -> dict[str, pd.DataFrame]:
    """
    Build symmetric pre-drift and post-drift windows.

    Pre-drift:
        [drift_sample - WINDOW_SIZE, drift_sample - 1]

    Post-drift:
        [drift_sample, drift_sample + WINDOW_SIZE - 1]
    """

    sorted_df = df.sort_values(
        "sample_index"
    ).reset_index(drop=True)

    pre_start = max(
        1,
        drift_sample - WINDOW_SIZE,
    )

    pre_end = drift_sample - 1

    post_start = drift_sample

    post_end = (
        drift_sample
        + WINDOW_SIZE
        - 1
    )

    pre_df = sorted_df[
        sorted_df["sample_index"].between(
            pre_start,
            pre_end,
        )
    ].copy()

    post_df = sorted_df[
        sorted_df["sample_index"].between(
            post_start,
            post_end,
        )
    ].copy()

    if len(pre_df) < WINDOW_SIZE:
        raise RuntimeError(
            "Insufficient samples available before the drift point "
            f"for a {WINDOW_SIZE}-sample window."
        )

    if len(post_df) < WINDOW_SIZE:
        raise RuntimeError(
            "Insufficient samples available after the drift point "
            f"for a {WINDOW_SIZE}-sample window."
        )

    return {
        "PRE_DRIFT": pre_df,
        "POST_DRIFT": post_df,
    }


# ============================================================
# Evaluate one window
# ============================================================

def evaluate_window(
    window_name: str,
    window_df: pd.DataFrame,
) -> list[dict[str, object]]:
    """Evaluate both routing policies within one time window."""

    confidence_only = window_df["confidence"].apply(
        confidence_only_decision
    )

    confidence_plus_drift = window_df.apply(
        lambda row: confidence_plus_drift_decision(
            float(row["confidence"]),
            bool(row["drift_detected"]),
        ),
        axis=1,
    )

    rows: list[dict[str, object]] = []

    for policy_name, decisions in [
        (
            "CONFIDENCE_ONLY",
            confidence_only,
        ),
        (
            "CONFIDENCE_PLUS_DRIFT",
            confidence_plus_drift,
        ),
    ]:
        metrics = calculate_policy_metrics(
            window_df,
            decisions,
        )

        rows.append(
            {
                "window": window_name,
                "policy": policy_name,
                **metrics,
            }
        )

    return rows


# ============================================================
# Main
# ============================================================

def main() -> None:
    print(
        "Loading existing edge-routing results..."
    )

    df = load_results()

    print(
        f"Loaded {len(df):,} samples."
    )

    drift_sample = find_first_drift_sample(
        df
    )

    print(
        f"First detected drift sample: "
        f"{drift_sample:,}"
    )

    windows = build_windows(
        df,
        drift_sample,
    )

    print(
        "\nUsing symmetric windows:"
    )

    for name, window_df in windows.items():
        print(
            f"  {name:<12}: "
            f"{int(window_df['sample_index'].min()):,}"
            f" - "
            f"{int(window_df['sample_index'].max()):,} "
            f"({len(window_df):,} samples)"
        )

    all_rows: list[dict[str, object]] = []

    for window_name, window_df in windows.items():

        rows = evaluate_window(
            window_name,
            window_df,
        )

        all_rows.extend(
            rows
        )

    results_df = pd.DataFrame(
        all_rows
    )

    # ========================================================
    # Direct incremental contribution of drift
    # ========================================================

    incremental_rows = []

    for window_name in windows:

        subset = results_df[
            results_df["window"]
            == window_name
        ].copy()

        confidence_row = subset[
            subset["policy"]
            == "CONFIDENCE_ONLY"
        ].iloc[0]

        drift_row = subset[
            subset["policy"]
            == "CONFIDENCE_PLUS_DRIFT"
        ].iloc[0]

        incremental_rows.append(
            {
                "window": window_name,
                "additional_rag_invocations": (
                    int(
                        drift_row["rag_count"]
                    )
                    - int(
                        confidence_row["rag_count"]
                    )
                ),
                "additional_rag_rate_percentage_points": (
                    float(
                        drift_row[
                            "rag_invocation_rate_percent"
                        ]
                    )
                    - float(
                        confidence_row[
                            "rag_invocation_rate_percent"
                        ]
                    )
                ),
                "additional_false_negative_escalations": (
                    int(
                        drift_row[
                            "false_negative_escalated_to_rag"
                        ]
                    )
                    - int(
                        confidence_row[
                            "false_negative_escalated_to_rag"
                        ]
                    )
                ),
                "additional_drift_only_escalations": (
                    int(
                        drift_row[
                            "drift_only_escalations"
                        ]
                    )
                    - int(
                        confidence_row[
                            "drift_only_escalations"
                        ]
                    )
                ),
            }
        )

    incremental_df = pd.DataFrame(
        incremental_rows
    )

    # ========================================================
    # Print results
    # ========================================================

    print(
        "\n" + "=" * 100
    )
    print(
        "DRIFT ADAPTATION EVALUATION"
    )
    print(
        "=" * 100
    )

    print(
        f"First drift sample : {drift_sample:,}"
    )

    print(
        f"Window size        : {WINDOW_SIZE:,} "
        "samples before + after"
    )

    print(
        f"Confidence threshold: "
        f"{CONFIDENCE_THRESHOLD:.2f}"
    )

    print(
        "\n" + "-" * 100
    )
    print(
        "POLICY PERFORMANCE BY WINDOW"
    )
    print(
        "-" * 100
    )

    display_df = results_df.copy()

    numeric_columns = [
        "rag_invocation_rate_percent",
        "false_negative_escalation_rate_percent",
        "mean_confidence",
    ]

    for column in numeric_columns:
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
        "\n" + "-" * 100
    )
    print(
        "INCREMENTAL DRIFT CONTRIBUTION"
    )
    print(
        "-" * 100
    )

    print(
        incremental_df.to_string(
            index=False
        )
    )

    # ========================================================
    # Interpretation
    # ========================================================

    post_increment = incremental_df[
        incremental_df["window"]
        == "POST_DRIFT"
    ].iloc[0]

    print(
        "\n" + "-" * 100
    )
    print(
        "INTERPRETATION"
    )
    print(
        "-" * 100
    )

    if (
        int(
            post_increment[
                "additional_rag_invocations"
            ]
        )
        > 0
    ):
        print(
            "Drift awareness produced additional "
            "RAG escalations in the post-drift window."
        )

    else:
        print(
            "Drift awareness produced no additional "
            "RAG escalations in the post-drift window "
            "because the affected samples were already "
            "below the confidence threshold."
        )

    print(
        "The Random Forest predictions are unchanged "
        "between policies; the experiment evaluates the "
        "incremental routing contribution of drift awareness."
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
        "\n" + "=" * 100
    )
    print(
        "RESULTS SAVED"
    )
    print(
        "=" * 100
    )

    print(
        OUTPUT_FILE
    )


if __name__ == "__main__":
    main()
