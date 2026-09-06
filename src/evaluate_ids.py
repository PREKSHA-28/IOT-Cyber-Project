from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


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

TEST_FILE = SPLIT_DIR / "known_test.csv"


# ============================================================
# Feature preparation
# ============================================================

def prepare_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Prepare features and binary target."""

    y = (
        df["Label_Binary"]
        .map(
            {
                "BENIGN": 0,
                "ATTACK": 1,
            }
        )
    )

    feature_columns = [
        column
        for column in df.columns
        if column not in ["Label", "Label_Binary"]
    ]

    X = df[feature_columns].copy()

    return X, y


# ============================================================
# Main
# ============================================================

def main() -> None:

    # --------------------------------------------------------
    # Check required files
    # --------------------------------------------------------

    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model not found:\n{MODEL_FILE}"
        )

    if not TEST_FILE.exists():
        raise FileNotFoundError(
            f"Known test file not found:\n{TEST_FILE}"
        )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    print("Loading trained model...")
    model = joblib.load(MODEL_FILE)

    # --------------------------------------------------------
    # Load untouched test set
    # --------------------------------------------------------

    print("Loading known test data...")
    test_df = pd.read_csv(TEST_FILE)

    print(f"Test rows: {len(test_df):,}")

    # --------------------------------------------------------
    # Prepare features
    # --------------------------------------------------------

    X_test, y_test = prepare_features(test_df)

    print(f"Number of features: {X_test.shape[1]}")

    # --------------------------------------------------------
    # Predict
    # --------------------------------------------------------

    print("\nRunning prediction...")
    predictions = model.predict(X_test)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    precision = precision_score(
        y_test,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_test,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0,
    )

    # --------------------------------------------------------
    # Display results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("KNOWN-ATTACK TEST RESULTS")
    print("=" * 70)

    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=["BENIGN", "ATTACK"],
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )


if __name__ == "__main__":
    main()