"""
Compare Always-RAG against the existing Selective-RAG pipeline.

Selective-RAG:
    Uses the already completed 20,000-sample end-to-end experiment.
    Only samples routed to RAG are counted, with their measured
    retrieval/generation/total latencies.

Always-RAG:
    Reconstructs the exact same 20,000-sample stream used by the
    end-to-end experiment (10,000 KNOWN + 10,000 EMERGING, random_state=42)
    and executes the deterministic grounded RAG pipeline for every sample.

The comparison focuses on resource/workload efficiency. The underlying
Random Forest predictions are identical; this experiment does not claim
that routing changes ML classification accuracy.
"""

from __future__ import annotations

import time
from pathlib import Path

import pandas as pd

from .rag_integration import RAGIntegration
from .rag_pipeline import DeterministicGroundedGenerator
from .rag_request import RAGRequest
from .run_end_to_end import build_behavior_context


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"

KNOWN_TEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
    / "known_test.csv"
)

EMERGING_TEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
    / "emerging_test.csv"
)

END_TO_END_RESULTS_FILE = (
    RESULTS_DIR / "end_to_end_results.csv"
)

SELECTIVE_RAG_FILE = (
    RESULTS_DIR / "end_to_end_rag_results.csv"
)

OUTPUT_FILE = (
    RESULTS_DIR / "always_vs_selective_rag.csv"
)


# ============================================================
# Configuration
# ============================================================

RANDOM_STATE = 42
KNOWN_STREAM_SIZE = 10_000
EMERGING_STREAM_SIZE = 10_000


# ============================================================
# Validation / loading
# ============================================================

def load_end_to_end_results() -> pd.DataFrame:
    if not END_TO_END_RESULTS_FILE.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{END_TO_END_RESULTS_FILE}"
        )

    df = pd.read_csv(
        END_TO_END_RESULTS_FILE
    )

    required = [
        "sample_index",
        "phase",
        "prediction",
        "raw_attack_probability",
        "calibrated_attack_probability",
        "confidence",
        "drift_detected",
        "routing_decision",
    ]

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            "Missing columns in end-to-end results:\n"
            + "\n".join(missing)
        )

    return df


def load_selective_rag_results() -> pd.DataFrame:
    if not SELECTIVE_RAG_FILE.exists():
        raise FileNotFoundError(
            f"Required file not found:\n{SELECTIVE_RAG_FILE}"
        )

    return pd.read_csv(
        SELECTIVE_RAG_FILE
    )


def reconstruct_stream() -> pd.DataFrame:
    for path in [
        KNOWN_TEST_FILE,
        EMERGING_TEST_FILE,
    ]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    known_df = pd.read_csv(
        KNOWN_TEST_FILE
    )

    emerging_df = pd.read_csv(
        EMERGING_TEST_FILE
    )

    known_sample = known_df.sample(
        n=min(
            KNOWN_STREAM_SIZE,
            len(known_df),
        ),
        random_state=RANDOM_STATE,
    ).reset_index(
        drop=True
    )

    emerging_sample = emerging_df.sample(
        n=min(
            EMERGING_STREAM_SIZE,
            len(emerging_df),
        ),
        random_state=RANDOM_STATE,
    ).reset_index(
        drop=True
    )

    return pd.concat(
        [
            known_sample,
            emerging_sample,
        ],
        ignore_index=True,
    )


# ============================================================
# Always-RAG execution
# ============================================================

def build_always_rag_request(
    row: pd.Series,
    end_to_end_row: pd.Series,
) -> RAGRequest:
    """
    Build a RAG request without using the ground-truth label.

    The actual Label column remains outside the request and is used
    only later for aggregate analysis, never for generation/retrieval.
    """

    behavior_context = build_behavior_context(
        row
    )

    context = {
        "retrieval_hint": (
            "cybersecurity incident detection analysis "
            "containment recovery"
        ),
        "behavior_context": behavior_context,
    }

    return RAGRequest(
        prediction=str(
            end_to_end_row["prediction"]
        ),
        raw_probability=float(
            end_to_end_row[
                "raw_attack_probability"
            ]
        ),
        calibrated_probability=float(
            end_to_end_row[
                "calibrated_attack_probability"
            ]
        ),
        confidence=float(
            end_to_end_row["confidence"]
        ),
        drift_detected=bool(
            end_to_end_row[
                "drift_detected"
            ]
        ),
        routing_reason="ALWAYS_RAG",
        behavior_context=behavior_context,
    )


def run_always_rag(
    stream_df: pd.DataFrame,
    end_to_end_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Execute deterministic RAG for every sample.
    """

    rag_layer = RAGIntegration.create(
        DeterministicGroundedGenerator()
    )

    rows: list[dict[str, object]] = []

    started_total = time.perf_counter()

    for index in range(
        len(stream_df)
    ):

        sample_number = index + 1

        source_row = stream_df.iloc[
            index
        ]

        model_row = end_to_end_df.iloc[
            index
        ]

        request = build_always_rag_request(
            source_row,
            model_row,
        )

        result = rag_layer.run(
            request
        )

        response = result.response

        rows.append(
            {
                "sample_index": sample_number,
                "phase": model_row["phase"],
                "actual_label": source_row["Label"],
                "prediction": model_row["prediction"],
                "confidence": model_row["confidence"],
                "drift_detected": model_row[
                    "drift_detected"
                ],
                "threat": response.threat,
                "retrieval_latency_ms": (
                    response.retrieval_latency_ms
                ),
                "generation_latency_ms": (
                    response.generation_latency_ms
                ),
                "total_rag_latency_ms": (
                    response.total_latency_ms
                ),
                "evidence_count": len(
                    response.evidence
                ),
            }
        )

    elapsed_ms = (
        time.perf_counter()
        - started_total
    ) * 1000

    result_df = pd.DataFrame(
        rows
    )

    return result_df, elapsed_ms


# ============================================================
# Aggregate comparison
# ============================================================

def summarize_mode(
    mode_name: str,
    rag_df: pd.DataFrame,
    total_samples: int,
    execution_elapsed_ms: float | None,
) -> dict[str, object]:

    rag_calls = len(
        rag_df
    )

    total_rag_latency_ms = float(
        rag_df[
            "total_rag_latency_ms"
        ].sum()
    )

    mean_rag_latency_ms = float(
        rag_df[
            "total_rag_latency_ms"
        ].mean()
    )

    median_rag_latency_ms = float(
        rag_df[
            "total_rag_latency_ms"
        ].median()
    )

    p95_rag_latency_ms = float(
        rag_df[
            "total_rag_latency_ms"
        ].quantile(0.95)
    )

    p99_rag_latency_ms = float(
        rag_df[
            "total_rag_latency_ms"
        ].quantile(0.99)
    )

    return {
        "mode": mode_name,
        "total_samples": total_samples,
        "rag_calls": rag_calls,
        "rag_invocation_percentage": (
            rag_calls
            / total_samples
            * 100.0
        ),
        "total_measured_rag_latency_ms": (
            total_rag_latency_ms
        ),
        "mean_rag_latency_ms": (
            mean_rag_latency_ms
        ),
        "median_rag_latency_ms": (
            median_rag_latency_ms
        ),
        "p95_rag_latency_ms": (
            p95_rag_latency_ms
        ),
        "p99_rag_latency_ms": (
            p99_rag_latency_ms
        ),
        "measured_execution_elapsed_ms": (
            execution_elapsed_ms
            if execution_elapsed_ms is not None
            else ""
        ),
    }


def phase_summary(
    mode_name: str,
    rag_df: pd.DataFrame,
    total_samples_by_phase: dict[str, int],
) -> list[dict[str, object]]:

    rows = []

    for phase in [
        "KNOWN",
        "EMERGING",
    ]:

        subset = rag_df[
            rag_df["phase"]
            == phase
        ]

        total_samples = total_samples_by_phase[
            phase
        ]

        rag_calls = len(
            subset
        )

        rows.append(
            {
                "mode": mode_name,
                "phase": phase,
                "total_samples": total_samples,
                "rag_calls": rag_calls,
                "rag_invocation_percentage": (
                    rag_calls
                    / total_samples
                    * 100.0
                ),
                "total_measured_rag_latency_ms": float(
                    subset[
                        "total_rag_latency_ms"
                    ].sum()
                ),
                "mean_rag_latency_ms": (
                    float(
                        subset[
                            "total_rag_latency_ms"
                        ].mean()
                    )
                    if not subset.empty
                    else 0.0
                ),
            }
        )

    return rows


# ============================================================
# Main
# ============================================================

def main() -> None:

    print(
        "Loading existing end-to-end results..."
    )

    end_to_end_df = (
        load_end_to_end_results()
    )

    selective_rag_df = (
        load_selective_rag_results()
    )

    print(
        f"Existing end-to-end samples: "
        f"{len(end_to_end_df):,}"
    )

    print(
        f"Existing selective RAG calls: "
        f"{len(selective_rag_df):,}"
    )

    if len(end_to_end_df) != 20_000:
        raise ValueError(
            "Expected exactly 20,000 end-to-end samples."
        )

    # --------------------------------------------------------
    # Reconstruct the exact original stream
    # --------------------------------------------------------

    print(
        "\nReconstructing the exact 20,000-sample stream..."
    )

    stream_df = reconstruct_stream()

    if len(stream_df) != len(
        end_to_end_df
    ):
        raise ValueError(
            "Reconstructed stream length does not match "
            "the end-to-end result length."
        )

    # Check the phase/label alignment before running Always-RAG.
    reconstructed_labels = (
        stream_df["Label"]
        .astype(str)
        .tolist()
    )

    saved_labels = (
        end_to_end_df[
            "actual_label"
        ]
        .astype(str)
        .tolist()
    )

    if reconstructed_labels != saved_labels:
        raise RuntimeError(
            "Reconstructed stream does not align with the saved "
            "end-to-end result labels."
        )

    print(
        "Stream alignment verified."
    )

    # --------------------------------------------------------
    # Run Always-RAG
    # --------------------------------------------------------

    print(
        "\nExecuting Always-RAG on all 20,000 samples..."
    )

    always_rag_df, always_elapsed_ms = (
        run_always_rag(
            stream_df,
            end_to_end_df,
        )
    )

    print(
        f"Always-RAG execution elapsed: "
        f"{always_elapsed_ms:.2f} ms"
    )

    # --------------------------------------------------------
    # Build summaries
    # --------------------------------------------------------

    total_samples = len(
        end_to_end_df
    )

    selective_summary = summarize_mode(
        "SELECTIVE_RAG",
        selective_rag_df,
        total_samples,
        None,
    )

    always_summary = summarize_mode(
        "ALWAYS_RAG",
        always_rag_df,
        total_samples,
        always_elapsed_ms,
    )

    summary_df = pd.DataFrame(
        [
            selective_summary,
            always_summary,
        ]
    )

    # --------------------------------------------------------
    # Phase summaries
    # --------------------------------------------------------

    total_samples_by_phase = {
        phase: int(
            (
                end_to_end_df["phase"]
                == phase
            ).sum()
        )
        for phase in [
            "KNOWN",
            "EMERGING",
        ]
    }

    phase_rows = []

    phase_rows.extend(
        phase_summary(
            "SELECTIVE_RAG",
            selective_rag_df,
            total_samples_by_phase,
        )
    )

    phase_rows.extend(
        phase_summary(
            "ALWAYS_RAG",
            always_rag_df,
            total_samples_by_phase,
        )
    )

    phase_df = pd.DataFrame(
        phase_rows
    )

    # --------------------------------------------------------
    # Savings
    # --------------------------------------------------------

    selective_calls = int(
        len(selective_rag_df)
    )

    always_calls = int(
        len(always_rag_df)
    )

    saved_calls = (
        always_calls
        - selective_calls
    )

    workload_reduction_percent = (
        saved_calls
        / always_calls
        * 100.0
    )

    selective_total_latency = float(
        selective_rag_df[
            "total_rag_latency_ms"
        ].sum()
    )

    always_total_latency = float(
        always_rag_df[
            "total_rag_latency_ms"
        ].sum()
    )

    estimated_latency_saved_ms = (
        always_total_latency
        - selective_total_latency
    )

    latency_reduction_percent = (
        estimated_latency_saved_ms
        / always_total_latency
        * 100.0
        if always_total_latency
        else 0.0
    )

    savings_df = pd.DataFrame(
        [
            {
                "total_samples": total_samples,
                "always_rag_calls": always_calls,
                "selective_rag_calls": selective_calls,
                "rag_calls_avoided": saved_calls,
                "rag_workload_reduction_percent": (
                    workload_reduction_percent
                ),
                "always_total_measured_rag_latency_ms": (
                    always_total_latency
                ),
                "selective_total_measured_rag_latency_ms": (
                    selective_total_latency
                ),
                "measured_rag_latency_reduction_ms": (
                    estimated_latency_saved_ms
                ),
                "measured_rag_latency_reduction_percent": (
                    latency_reduction_percent
                ),
            }
        ]
    )

    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print(
        "\n" + "=" * 105
    )
    print(
        "ALWAYS-RAG VS SELECTIVE-RAG"
    )
    print(
        "=" * 105
    )

    print(
        summary_df.to_string(
            index=False
        )
    )

    print(
        "\n" + "-" * 105
    )
    print(
        "PHASE COMPARISON"
    )
    print(
        "-" * 105
    )

    print(
        phase_df.to_string(
            index=False
        )
    )

    print(
        "\n" + "-" * 105
    )
    print(
        "RESOURCE SAVINGS"
    )
    print(
        "-" * 105
    )

    print(
        savings_df.to_string(
            index=False
        )
    )

    print(
        "\n" + "-" * 105
    )
    print(
        "INTERPRETATION"
    )
    print(
        "-" * 105
    )

    print(
        "Selective-RAG executes RAG only for samples selected by "
        "the edge routing policy."
    )

    print(
        "Always-RAG executes the same deterministic RAG pipeline "
        "for every sample."
    )

    print(
        "The Random Forest predictions are held constant; this is "
        "a resource/workload comparison, not a classification-accuracy comparison."
    )

    print(
        f"Selective routing avoided "
        f"{saved_calls:,} RAG calls, corresponding to a "
        f"{workload_reduction_percent:.2f}% reduction in RAG workload "
        f"relative to Always-RAG."
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
        / "always_vs_selective_rag_summary.csv"
    )

    phase_file = (
        RESULTS_DIR
        / "always_vs_selective_rag_phase.csv"
    )

    savings_file = (
        RESULTS_DIR
        / "always_vs_selective_rag_savings.csv"
    )

    detail_file = (
        RESULTS_DIR
        / "always_rag_results.csv"
    )

    summary_df.to_csv(
        summary_file,
        index=False,
    )

    phase_df.to_csv(
        phase_file,
        index=False,
    )

    savings_df.to_csv(
        savings_file,
        index=False,
    )

    always_rag_df.to_csv(
        detail_file,
        index=False,
    )

    # Main named output is a combined compact file.
    combined_rows = []

    for _, row in summary_df.iterrows():
        combined_rows.append(
            {
                "section": "OVERALL",
                **row.to_dict(),
            }
        )

    for _, row in phase_df.iterrows():
        combined_rows.append(
            {
                "section": "PHASE",
                **row.to_dict(),
            }
        )

    output_df = pd.DataFrame(
        combined_rows
    )

    output_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(
        "\n" + "=" * 105
    )
    print(
        "RESULTS SAVED"
    )
    print(
        "=" * 105
    )

    print(
        OUTPUT_FILE
    )

    print(
        summary_file
    )

    print(
        phase_file
    )

    print(
        savings_file
    )

    print(
        detail_file
    )


if __name__ == "__main__":
    main()