from pathlib import Path
from time import perf_counter

import os
import sys

import pandas as pd

from orchestrator import EdgeOrchestrator


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_DIR = PROJECT_ROOT / "results"

INPUT_FILE = (
    RESULTS_DIR / "edge_routing_results.csv"
)

OUTPUT_FILE = (
    RESULTS_DIR / "orchestrator_resource_results.csv"
)


# ============================================================
# Configuration
# ============================================================

CONFIDENCE_THRESHOLD = 0.70


# ============================================================
# Optional process-resource support
# ============================================================

try:
    import psutil

    PSUTIL_AVAILABLE = True

except ImportError:

    PSUTIL_AVAILABLE = False


# ============================================================
# Main
# ============================================================

def main() -> None:

    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Required result file not found:\n{INPUT_FILE}"
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
            "Required columns are missing:\n"
            + "\n".join(missing_columns)
        )

    orchestrator = EdgeOrchestrator(
        confidence_threshold=CONFIDENCE_THRESHOLD
    )

    # --------------------------------------------------------
    # Process handle for resource measurements.
    # --------------------------------------------------------

    process = None

    if PSUTIL_AVAILABLE:

        process = psutil.Process(
            os.getpid()
        )

        process.cpu_percent(
            interval=None
        )

    # --------------------------------------------------------
    # Warm-up.
    # --------------------------------------------------------

    for _, row in df.head(100).iterrows():

        orchestrator.decide(
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

    # --------------------------------------------------------
    # Starting memory.
    # --------------------------------------------------------

    starting_memory_mb = None

    if process is not None:

        starting_memory_mb = (
            process.memory_info().rss
            / (1024 * 1024)
        )

    # --------------------------------------------------------
    # Run sequential orchestration workload.
    # --------------------------------------------------------

    local_count = 0
    rag_count = 0

    start_time = perf_counter()

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

        if result.routing_decision == "LOCAL":

            local_count += 1

        else:

            rag_count += 1

    end_time = perf_counter()

    # --------------------------------------------------------
    # Ending resource measurements.
    # --------------------------------------------------------

    ending_memory_mb = None
    cpu_percent = None

    if process is not None:

        ending_memory_mb = (
            process.memory_info().rss
            / (1024 * 1024)
        )

        cpu_percent = process.cpu_percent(
            interval=None
        )

    # --------------------------------------------------------
    # Calculate metrics.
    # --------------------------------------------------------

    total_samples = len(df)

    elapsed_seconds = (
        end_time - start_time
    )

    throughput = (
        total_samples / elapsed_seconds
        if elapsed_seconds > 0
        else 0.0
    )

    rag_percentage = (
        rag_count / total_samples * 100
        if total_samples > 0
        else 0.0
    )

    local_percentage = (
        local_count / total_samples * 100
        if total_samples > 0
        else 0.0
    )

    memory_change_mb = None

    if (
        starting_memory_mb is not None
        and ending_memory_mb is not None
    ):

        memory_change_mb = (
            ending_memory_mb
            - starting_memory_mb
        )

    # --------------------------------------------------------
    # Build result.
    # --------------------------------------------------------

    result = {
        "samples_processed": total_samples,

        "confidence_threshold": (
            CONFIDENCE_THRESHOLD
        ),

        "local_count": local_count,

        "local_percentage": (
            local_percentage
        ),

        "rag_count": rag_count,

        "rag_invocation_percentage": (
            rag_percentage
        ),

        "elapsed_seconds": (
            elapsed_seconds
        ),

        "throughput_samples_per_second": (
            throughput
        ),

        "psutil_available": (
            PSUTIL_AVAILABLE
        ),

        "cpu_percent": (
            cpu_percent
        ),

        "starting_memory_mb": (
            starting_memory_mb
        ),

        "ending_memory_mb": (
            ending_memory_mb
        ),

        "memory_change_mb": (
            memory_change_mb
        ),
    }

    results_df = pd.DataFrame(
        [result]
    )

    # --------------------------------------------------------
    # Print results.
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("ORCHESTRATOR RESOURCE BENCHMARK")
    print("=" * 80)

    for key, value in result.items():

        if isinstance(value, float):

            print(
                f"{key}: {value:.6f}"
            )

        else:

            print(
                f"{key}: {value}"
            )

    # --------------------------------------------------------
    # Explain measurement scope.
    # --------------------------------------------------------

    print("\nMeasurement scope:")
    print(
        "These measurements describe the orchestration "
        "workload only."
    )

    print(
        "They do not represent Random Forest inference "
        "or RAG/LLM execution."
    )

    # --------------------------------------------------------
    # Save results.
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print("\n" + "=" * 80)
    print("RESULTS SAVED")
    print("=" * 80)

    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()