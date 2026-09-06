from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss


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

CALIBRATOR_FILE = (
    PROJECT_ROOT
    / "models"
    / "attack_probability_calibrator.joblib"
)

VALIDATION_FILE = SPLIT_DIR / "validation.csv"
KNOWN_TEST_FILE = SPLIT_DIR / "known_test.csv"
EMERGING_TEST_FILE = SPLIT_DIR / "emerging_test.csv"


# ============================================================
# Feature preparation
# ============================================================

def prepare_features(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Separate model features and binary target."""

    y = df["Label_Binary"].map(
        {
            "BENIGN": 0,
            "ATTACK": 1,
        }
    )

    feature_columns = [
        column
        for column in df.columns
        if column not in ["Label", "Label_Binary"]
    ]

    X = df[feature_columns].copy()

    return X, y


# ============================================================
# Calibration metrics
# ============================================================

def expected_calibration_error(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    number_of_bins: int = 10,
) -> float:
    """
    Calculate Expected Calibration Error (ECE).

    A well-calibrated model should have predicted probabilities
    that agree reasonably well with observed correctness.
    """

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    bins = np.linspace(0.0, 1.0, number_of_bins + 1)

    ece = 0.0

    for lower, upper in zip(bins[:-1], bins[1:]):

        if upper == 1.0:
            mask = (
                (probabilities >= lower)
                & (probabilities <= upper)
            )
        else:
            mask = (
                (probabilities >= lower)
                & (probabilities < upper)
            )

        if not np.any(mask):
            continue

        mean_confidence = probabilities[mask].mean()
        mean_accuracy = y_true[mask].mean()

        bin_fraction = mask.mean()

        ece += (
            abs(mean_confidence - mean_accuracy)
            * bin_fraction
        )

    return float(ece)


# ============================================================
# Analyze probabilities
# ============================================================

def analyze_probabilities(
    name: str,
    y_true: pd.Series,
    raw_probabilities: np.ndarray,
    calibrated_probabilities: np.ndarray,
) -> None:

    raw_brier = brier_score_loss(
        y_true,
        raw_probabilities,
    )

    calibrated_brier = brier_score_loss(
        y_true,
        calibrated_probabilities,
    )

    raw_ece = expected_calibration_error(
        y_true,
        raw_probabilities,
    )

    calibrated_ece = expected_calibration_error(
        y_true,
        calibrated_probabilities,
    )

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    print(f"Samples: {len(y_true):,}")

    print("\nBrier score:")
    print(f"Raw        : {raw_brier:.6f}")
    print(f"Calibrated : {calibrated_brier:.6f}")

    print("\nExpected Calibration Error (ECE):")
    print(f"Raw        : {raw_ece:.6f}")
    print(f"Calibrated : {calibrated_ece:.6f}")

    print("\nProbability statistics:")
    print(
        pd.DataFrame(
            {
                "Raw": raw_probabilities,
                "Calibrated": calibrated_probabilities,
            }
        ).describe()
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    # --------------------------------------------------------
    # Check required files
    # --------------------------------------------------------

    required_files = [
        MODEL_FILE,
        VALIDATION_FILE,
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
    # Load validation data
    # --------------------------------------------------------

    print("Loading validation data...")
    validation_df = pd.read_csv(VALIDATION_FILE)

    X_validation, y_validation = prepare_features(
        validation_df
    )

    # --------------------------------------------------------
    # Generate raw probabilities
    # --------------------------------------------------------

    print("Generating validation probabilities...")

    validation_probabilities = model.predict_proba(
        X_validation
    )[:, 1]

    # --------------------------------------------------------
    # Fit isotonic calibration
    #
    # IMPORTANT:
    # Calibration is learned ONLY from validation data.
    # The test sets remain untouched.
    # --------------------------------------------------------

    print("Fitting probability calibrator...")

    calibrator = IsotonicRegression(
        y_min=0.0,
        y_max=1.0,
        out_of_bounds="clip",
    )

    calibrator.fit(
        validation_probabilities,
        y_validation.to_numpy(),
    )

    calibrated_validation_probabilities = calibrator.predict(
        validation_probabilities
    )

    # --------------------------------------------------------
    # Save calibrator
    # --------------------------------------------------------

    CALIBRATOR_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        calibrator,
        CALIBRATOR_FILE,
    )

    print("\nCalibrator saved to:")
    print(CALIBRATOR_FILE)

    # --------------------------------------------------------
    # Evaluate calibration on validation data
    # --------------------------------------------------------

    analyze_probabilities(
        "VALIDATION CALIBRATION",
        y_validation,
        validation_probabilities,
        calibrated_validation_probabilities,
    )

    # --------------------------------------------------------
    # Known test
    # --------------------------------------------------------

    print("\nLoading known-test data...")

    known_test_df = pd.read_csv(
        KNOWN_TEST_FILE
    )

    X_known, y_known = prepare_features(
        known_test_df
    )

    print("Generating known-test probabilities...")

    known_raw_probabilities = model.predict_proba(
        X_known
    )[:, 1]

    known_calibrated_probabilities = calibrator.predict(
        known_raw_probabilities
    )

    analyze_probabilities(
        "KNOWN TEST CALIBRATION",
        y_known,
        known_raw_probabilities,
        known_calibrated_probabilities,
    )

    # --------------------------------------------------------
    # Emerging test
    # --------------------------------------------------------

    print("\nLoading emerging-test data...")

    emerging_test_df = pd.read_csv(
        EMERGING_TEST_FILE
    )

    X_emerging, y_emerging = prepare_features(
        emerging_test_df
    )

    print("Generating emerging-test probabilities...")

    emerging_raw_probabilities = model.predict_proba(
        X_emerging
    )[:, 1]

    emerging_calibrated_probabilities = calibrator.predict(
        emerging_raw_probabilities
    )

    analyze_probabilities(
        "EMERGING TEST CALIBRATION",
        y_emerging,
        emerging_raw_probabilities,
        emerging_calibrated_probabilities,
    )

    # --------------------------------------------------------
    # Example confidence thresholds
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("EMERGING TEST - CALIBRATED LOW-CONFIDENCE SAMPLES")
    print("=" * 70)

    predicted_class = np.where(
        emerging_calibrated_probabilities >= 0.5,
        1,
        0,
    )

    confidence = np.maximum(
        emerging_calibrated_probabilities,
        1.0 - emerging_calibrated_probabilities,
    )

    for threshold in [0.60, 0.70, 0.80, 0.90]:

        count = np.sum(
            confidence < threshold
        )

        percentage = (
            count
            / len(confidence)
            * 100
        )

        print(
            f"Confidence < {threshold:.2f}: "
            f"{count:,} "
            f"({percentage:.2f}%)"
        )

    print("\nCalibration completed successfully.")


if __name__ == "__main__":
    main()