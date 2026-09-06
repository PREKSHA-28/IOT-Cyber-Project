from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from river.drift import ADWIN


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
)

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "binary_ids_random_forest.joblib"
)

KNOWN_TEST_FILE = SPLIT_DIR / "known_test.csv"
EMERGING_TEST_FILE = SPLIT_DIR / "emerging_test.csv"

RESULTS_DIR = PROJECT_ROOT / "results"

DRIFT_RESULTS_FILE = (
    RESULTS_DIR / "drift_detection_results.csv"
)


# ============================================================
# Configuration
# ============================================================

RANDOM_STATE = 42

# Number of known samples used before the emerging phase.
# We use a controlled stream so we can observe the transition.
KNOWN_STREAM_SIZE = 50_000

# Number of emerging samples introduced after known traffic.
EMERGING_STREAM_SIZE = 20_000

# ADWIN confidence parameter.
# Smaller delta = more conservative drift detection.
ADWIN_DELTA = 0.002


# ============================================================
# Feature preparation
# ============================================================

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the 39 ML input features."""

    feature_columns = [
        column
        for column in df.columns
        if column not in ["Label", "Label_Binary"]
    ]

    return df[feature_columns].copy()


# ============================================================
# Main
# ============================================================

def main() -> None:

    # --------------------------------------------------------
    # Check required files
    # --------------------------------------------------------

    required_files = [
        MODEL_FILE,
        KNOWN_TEST_FILE,
        EMERGING_TEST_FILE,
    ]

    for file_path in required_files:
        if not file_path.exists():
            raise FileNotFoundError(
                f"Required file not found:\n{file_path}"
            )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("Loading trained Random Forest...")
    model = joblib.load(MODEL_FILE)

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print("Loading known-test data...")
    known_df = pd.read_csv(KNOWN_TEST_FILE)

    print("Loading emerging-test data...")
    emerging_df = pd.read_csv(EMERGING_TEST_FILE)

    # --------------------------------------------------------
    # Create controlled stream
    # --------------------------------------------------------

    known_sample = known_df.sample(
        n=min(KNOWN_STREAM_SIZE, len(known_df)),
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    emerging_sample = emerging_df.sample(
        n=min(EMERGING_STREAM_SIZE, len(emerging_df)),
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    # Known phase followed by emerging phase.
    stream_df = pd.concat(
        [
            known_sample,
            emerging_sample,
        ],
        ignore_index=True,
    )

    print("\n" + "=" * 70)
    print("CONTROLLED STREAM")
    print("=" * 70)

    print(f"Known phase    : {len(known_sample):,} samples")
    print(f"Emerging phase : {len(emerging_sample):,} samples")
    print(f"Total stream   : {len(stream_df):,} samples")

    # --------------------------------------------------------
    # Generate attack probabilities
    # --------------------------------------------------------

    X_stream = prepare_features(stream_df)

    print("\nGenerating model probabilities...")

    attack_probabilities = model.predict_proba(
        X_stream
    )[:, 1]

    # --------------------------------------------------------
    # ADWIN drift detector
    #
    # We monitor the attack-probability stream.
    # A significant change in its distribution can indicate
    # that the model is seeing behavior different from the
    # earlier stream.
    # --------------------------------------------------------

    detector = ADWIN(
        delta=ADWIN_DELTA
    )

    drift_events = []

    print("\nRunning ADWIN...")

    for index, probability in enumerate(
        attack_probabilities,
        start=1,
    ):

        detector.update(
            float(probability)
        )

        if detector.drift_detected:

            phase = (
                "KNOWN"
                if index <= len(known_sample)
                else "EMERGING"
            )

            drift_events.append(
                {
                    "sample_index": index,
                    "phase": phase,
                    "attack_probability": float(
                        probability
                    ),
                }
            )

            print(
                f"Drift detected at sample "
                f"{index:,} "
                f"({phase})"
            )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DRIFT DETECTION SUMMARY")
    print("=" * 70)

    print(
        f"Total drift events: "
        f"{len(drift_events)}"
    )

    if drift_events:

        first_drift = drift_events[0]

        print(
            f"First drift detected at sample: "
            f"{first_drift['sample_index']:,}"
        )

        print(
            f"First drift phase: "
            f"{first_drift['phase']}"
        )

    else:

        print(
            "No drift was detected in this controlled stream."
        )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df = pd.DataFrame(
        drift_events
    )

    results_df.to_csv(
        DRIFT_RESULTS_FILE,
        index=False,
    )

    print("\nResults saved to:")
    print(DRIFT_RESULTS_FILE)


if __name__ == "__main__":
    main()