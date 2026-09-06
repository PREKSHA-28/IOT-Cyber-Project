from pathlib import Path
from time import perf_counter

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
    RESULTS_DIR / "orchestrator_latency_results.csv"
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
    # Warm-up
    #
    # This avoids measuring first-call/import effects as part
    # of the benchmark.
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
    # Measure sequential orchestration latency.
    #
    # Each row represents one incoming stream sample.
    # --------------------------------------------------------

    latencies_microseconds = []

    start_total = perf_counter()

    for _, row in df.iterrows():

        start = perf_counter()

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

        end = perf_counter()

        latency_us = (
            end - start
        ) * 1_000_000

        latencies_microseconds.append(
            latency_us
        )

    end_total = perf_counter()

    # --------------------------------------------------------
    # Calculate metrics.
    # --------------------------------------------------------

    latency_series = pd.Series(
        latencies_microseconds
    )

    total_samples = len(df)

    total_seconds = (
        end_total - start_total
    )

    throughput = (
        total_samples / total_seconds
        if total_seconds > 0
        else 0.0
    )

    results = {
        "samples_processed": total_samples,

        "total_measurement_seconds": (
            total_seconds
        ),

        "mean_latency_microseconds": (
            latency_series.mean()
        ),

        "median_latency_microseconds": (
            latency_series.median()
        ),

        "p95_latency_microseconds": (
            latency_series.quantile(0.95)
        ),

        "p99_latency_microseconds": (
            latency_series.quantile(0.99)
        ),

        "min_latency_microseconds": (
            latency_series.min()
        ),

        "max_latency_microseconds": (
            latency_series.max()
        ),

        "orchestrator_throughput_samples_per_second": (
            throughput
        ),
    }

    results_df = pd.DataFrame(
        [results]
    )

    # --------------------------------------------------------
    # Print results.
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("ORCHESTRATOR LATENCY BENCHMARK")
    print("=" * 80)

    for key, value in results.items():

        if isinstance(value, float):

            print(
                f"{key}: {value:.6f}"
            )

        else:

            print(
                f"{key}: {value}"
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