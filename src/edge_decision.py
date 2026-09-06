from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from river.drift import ADWIN


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_FILE = (
    PROJECT_ROOT
    / "models"
    / "binary_ids_random_forest.joblib"
)

CALIBRATOR_FILE = (
    PROJECT_ROOT
    / "models"
    / "attack_probability_calibrator.joblib"
)

SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
)

KNOWN_TEST_FILE = SPLIT_DIR / "known_test.csv"
EMERGING_TEST_FILE = SPLIT_DIR / "emerging_test.csv"

RESULTS_DIR = PROJECT_ROOT / "results"

EDGE_RESULTS_FILE = (
    RESULTS_DIR / "edge_routing_results.csv"
)


# ============================================================
# Configuration
# ============================================================

# Temporary confidence threshold.
# We will evaluate multiple thresholds later.
CONFIDENCE_THRESHOLD = 0.70

# ADWIN parameter used in our previous drift experiment.
ADWIN_DELTA = 0.002

RANDOM_STATE = 42

# Keep the first integrated experiment manageable.
KNOWN_STREAM_SIZE = 10_000
EMERGING_STREAM_SIZE = 10_000


# ============================================================
# Feature preparation
# ============================================================

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the 39 features used by the ML model."""

    feature_columns = [
        column
        for column in df.columns
        if column not in ["Label", "Label_Binary"]
    ]

    return df[feature_columns].copy()


# ============================================================
# Confidence calculation
# ============================================================

def calculate_confidence(
    calibrated_attack_probability: float,
) -> float:
    """
    Confidence is the probability of the predicted class.

    For binary classification:

        confidence = max(P(ATTACK), P(BENIGN))
    """

    return float(
        max(
            calibrated_attack_probability,
            1.0 - calibrated_attack_probability,
        )
    )


# ============================================================
# Routing decision
# ============================================================

def make_decision(
    confidence: float,
    drift_detected: bool,
) -> str:
    """
    Decide whether traffic should be handled locally
    or escalated to RAG/LLM.

    LOCAL:
        High confidence AND no drift.

    RAG:
        Low confidence OR detected drift.
    """

    if (
        confidence >= CONFIDENCE_THRESHOLD
        and not drift_detected
    ):
        return "LOCAL"

    return "RAG"


# ============================================================
# Main
# ============================================================

def main() -> None:

    # --------------------------------------------------------
    # Check required files
    # --------------------------------------------------------

    required_files = [
        MODEL_FILE,
        CALIBRATOR_FILE,
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
    # Load calibrator
    # --------------------------------------------------------

    print("Loading probability calibrator...")
    calibrator = joblib.load(
        CALIBRATOR_FILE
    )

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print("\nLoading known-test data...")

    known_df = pd.read_csv(
        KNOWN_TEST_FILE
    )

    print("Loading emerging-test data...")

    emerging_df = pd.read_csv(
        EMERGING_TEST_FILE
    )

    # --------------------------------------------------------
    # Create controlled stream
    #
    # Known traffic comes first.
    # Emerging traffic comes afterwards.
    #
    # This lets ADWIN observe a distribution transition.
    # --------------------------------------------------------

    known_sample = known_df.sample(
        n=min(
            KNOWN_STREAM_SIZE,
            len(known_df),
        ),
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    emerging_sample = emerging_df.sample(
        n=min(
            EMERGING_STREAM_SIZE,
            len(emerging_df),
        ),
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    stream_df = pd.concat(
        [
            known_sample,
            emerging_sample,
        ],
        ignore_index=True,
    )

    phase_boundary = len(known_sample)

    print("\n" + "=" * 70)
    print("INTEGRATED EDGE STREAM")
    print("=" * 70)

    print(
        f"Known phase    : "
        f"{len(known_sample):,} samples"
    )

    print(
        f"Emerging phase : "
        f"{len(emerging_sample):,} samples"
    )

    print(
        f"Total stream   : "
        f"{len(stream_df):,} samples"
    )

    # --------------------------------------------------------
    # Prepare features
    # --------------------------------------------------------

    X_stream = prepare_features(
        stream_df
    )

    # --------------------------------------------------------
    # Generate ML probabilities
    # --------------------------------------------------------

    print("\nGenerating ML predictions...")

    probabilities = model.predict_proba(
        X_stream
    )

    predictions = model.predict(
        X_stream
    )

    raw_attack_probabilities = probabilities[:, 1]

    # --------------------------------------------------------
    # Calibrate probabilities
    # --------------------------------------------------------

    print(
        "Applying probability calibration..."
    )

    calibrated_attack_probabilities = (
        calibrator.predict(
            raw_attack_probabilities
        )
    )

    # --------------------------------------------------------
    # Initialize ADWIN
    # --------------------------------------------------------

    detector = ADWIN(
        delta=ADWIN_DELTA
    )

    # --------------------------------------------------------
    # Process stream sequentially
    #
    # This is important:
    # ADWIN sees one sample at a time, just as an edge
    # streaming system would observe incoming traffic.
    # --------------------------------------------------------

    results = []

    drift_events = []

    print("\nRunning integrated edge pipeline...")

    for index in range(
        len(stream_df)
    ):

        sample_number = index + 1

        raw_probability = float(
            raw_attack_probabilities[index]
        )

        calibrated_probability = float(
            calibrated_attack_probabilities[index]
        )

        prediction = int(
            predictions[index]
        )

        confidence = calculate_confidence(
            calibrated_probability
        )

        # ----------------------------------------------------
        # ADWIN observes the model's calibrated attack
        # probability.
        # ----------------------------------------------------

        detector.update(
            calibrated_probability
        )

        drift_detected = bool(
            detector.drift_detected
        )

        # ----------------------------------------------------
        # Determine which routing action is required.
        # ----------------------------------------------------

        routing_decision = make_decision(
            confidence,
            drift_detected,
        )

        phase = (
            "KNOWN"
            if sample_number <= phase_boundary
            else "EMERGING"
        )

        # ----------------------------------------------------
        # Save sample-level result
        # ----------------------------------------------------

        results.append(
            {
                "sample_index": sample_number,
                "phase": phase,
                "actual_label": stream_df[
                    "Label"
                ].iloc[index],
                "prediction": (
                    "ATTACK"
                    if prediction == 1
                    else "BENIGN"
                ),
                "raw_attack_probability": (
                    raw_probability
                ),
                "calibrated_attack_probability": (
                    calibrated_probability
                ),
                "confidence": confidence,
                "drift_detected": drift_detected,
                "routing_decision": routing_decision,
            }
        )

        # ----------------------------------------------------
        # Record drift events
        # ----------------------------------------------------

        if drift_detected:

            drift_events.append(
                {
                    "sample_index": sample_number,
                    "phase": phase,
                    "confidence": confidence,
                    "calibrated_attack_probability": (
                        calibrated_probability
                    ),
                }
            )

            print(
                f"Drift detected at sample "
                f"{sample_number:,} "
                f"({phase})"
            )

    # --------------------------------------------------------
    # Convert results to DataFrame
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Summary function
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("INTEGRATED EDGE ROUTING SUMMARY")
    print("=" * 70)

    for phase in [
        "KNOWN",
        "EMERGING",
    ]:

        subset = results_df[
            results_df["phase"] == phase
        ]

        total = len(subset)

        local_count = np.sum(
            subset["routing_decision"]
            == "LOCAL"
        )

        rag_count = np.sum(
            subset["routing_decision"]
            == "RAG"
        )

        drift_count = np.sum(
            subset["drift_detected"]
        )

        low_confidence_count = np.sum(
            subset["confidence"]
            < CONFIDENCE_THRESHOLD
        )

        print("\n" + phase)

        print(
            f"Total samples       : "
            f"{total:,}"
        )

        print(
            f"LOCAL               : "
            f"{local_count:,} "
            f"({local_count / total * 100:.2f}%)"
        )

        print(
            f"RAG                 : "
            f"{rag_count:,} "
            f"({rag_count / total * 100:.2f}%)"
        )

        print(
            f"Low-confidence      : "
            f"{low_confidence_count:,} "
            f"({low_confidence_count / total * 100:.2f}%)"
        )

        print(
            f"Drift detections    : "
            f"{drift_count:,}"
        )

        print(
            f"Mean confidence     : "
            f"{subset['confidence'].mean():.6f}"
        )

    # --------------------------------------------------------
    # Drift summary
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("DRIFT SUMMARY")
    print("=" * 70)

    print(
        f"Total drift events: "
        f"{len(drift_events):,}"
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
            "No drift detected."
        )

    # --------------------------------------------------------
    # Save detailed results
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        EDGE_RESULTS_FILE,
        index=False,
    )

    print("\n" + "=" * 70)
    print("RESULTS SAVED")
    print("=" * 70)

    print(
        EDGE_RESULTS_FILE
    )


if __name__ == "__main__":
    main()