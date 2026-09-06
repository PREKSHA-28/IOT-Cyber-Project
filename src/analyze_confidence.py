from pathlib import Path

import joblib
import pandas as pd
import numpy as np


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


# ============================================================
# Feature preparation
# ============================================================

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return only the features used by the ML model."""

    feature_columns = [
        column
        for column in df.columns
        if column not in ["Label", "Label_Binary"]
    ]

    return df[feature_columns].copy()


# ============================================================
# Analyze confidence
# ============================================================

def analyze_dataset(
    model,
    df: pd.DataFrame,
    dataset_name: str,
) -> None:
    """Analyze prediction confidence for one dataset."""

    X = prepare_features(df)

    # Random Forest class probabilities
    probabilities = model.predict_proba(X)

    # Probability of ATTACK
    attack_probability = probabilities[:, 1]

    # Predicted class
    predictions = model.predict(X)

    # Confidence = probability of the predicted class
    confidence = np.max(probabilities, axis=1)

    print("\n" + "=" * 70)
    print(dataset_name)
    print("=" * 70)

    print(f"Samples: {len(df):,}")

    print("\nPrediction counts:")
    predicted_labels = np.where(
        predictions == 1,
        "ATTACK",
        "BENIGN",
    )

    print(
        pd.Series(predicted_labels)
        .value_counts()
    )

    print("\nAttack probability statistics:")
    print(
        pd.Series(attack_probability).describe()
    )

    print("\nPrediction confidence statistics:")
    print(
        pd.Series(confidence).describe()
    )

    # --------------------------------------------------------
    # Confidence buckets
    # --------------------------------------------------------

    buckets = pd.cut(
        confidence,
        bins=[
            0.0,
            0.50,
            0.60,
            0.70,
            0.80,
            0.90,
            0.95,
            1.00,
        ],
        include_lowest=True,
    )

    print("\nConfidence distribution:")
    print(
        buckets.value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # Low-confidence count
    # --------------------------------------------------------

    print("\nLow-confidence samples:")

    for threshold in [0.60, 0.70, 0.80, 0.90]:

        count = np.sum(confidence < threshold)

        percentage = (
            count / len(confidence) * 100
        )

        print(
            f"Confidence < {threshold:.2f}: "
            f"{count:,} "
            f"({percentage:.2f}%)"
        )


# ============================================================
# Main
# ============================================================

def main() -> None:

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_FILE}"
        )

    known_test_file = SPLIT_DIR / "known_test.csv"
    emerging_test_file = SPLIT_DIR / "emerging_test.csv"

    if not known_test_file.exists():
        raise FileNotFoundError(
            f"Known test file not found:\n"
            f"{known_test_file}"
        )

    if not emerging_test_file.exists():
        raise FileNotFoundError(
            f"Emerging test file not found:\n"
            f"{emerging_test_file}"
        )

    print("Loading trained Random Forest...")
    model = joblib.load(MODEL_FILE)

    print("Loading known-test data...")
    known_df = pd.read_csv(known_test_file)

    print("Loading emerging-test data...")
    emerging_df = pd.read_csv(emerging_test_file)

    analyze_dataset(
        model,
        known_df,
        "KNOWN TEST CONFIDENCE",
    )

    analyze_dataset(
        model,
        emerging_df,
        "EMERGING TEST CONFIDENCE",
    )


if __name__ == "__main__":
    main()