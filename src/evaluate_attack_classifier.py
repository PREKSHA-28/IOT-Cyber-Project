from pathlib import Path

import joblib
import pandas as pd

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
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
    / "attack_type_random_forest.joblib"
)

TEST_FILE = SPLIT_DIR / "known_test.csv"


# ============================================================
# Feature preparation
# ============================================================

def prepare_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Keep only ATTACK rows and prepare the attack-type target.
    """

    attack_df = df[df["Label"] != "BENIGN"].copy()

    feature_columns = [
        column
        for column in attack_df.columns
        if column not in ["Label", "Label_Binary"]
    ]

    X = attack_df[feature_columns].copy()
    y = attack_df["Label"].copy()

    return X, y


# ============================================================
# Main
# ============================================================

def main() -> None:

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

    print("Loading attack-type model...")
    model = joblib.load(MODEL_FILE)

    # --------------------------------------------------------
    # Load known test data
    # --------------------------------------------------------

    print("Loading known-test data...")
    test_df = pd.read_csv(TEST_FILE)

    print(
        f"Total known-test rows: "
        f"{len(test_df):,}"
    )

    # --------------------------------------------------------
    # Prepare attack-only test data
    # --------------------------------------------------------

    X_test, y_test = prepare_features(test_df)

    print(
        f"Attack-only test rows: "
        f"{len(X_test):,}"
    )

    print(
        f"Number of features: "
        f"{X_test.shape[1]}"
    )

    print(
        f"Number of attack classes: "
        f"{y_test.nunique()}"
    )

    # --------------------------------------------------------
    # Verify emerging attacks are absent
    # --------------------------------------------------------

    emerging_attacks = {
        "DNS_SPOOFING",
        "VULNERABILITYSCAN",
        "DOS-HTTP_FLOOD",
    }

    leaked_classes = (
        set(y_test.unique())
        .intersection(emerging_attacks)
    )

    if leaked_classes:
        raise ValueError(
            "Emerging attack leakage detected in "
            f"known-test set: {leaked_classes}"
        )

    print("\nEmerging attack leakage check: PASSED")

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    print("\nRunning prediction...")

    predictions = model.predict(
        X_test
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
        zero_division=0,
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("ATTACK-TYPE CLASSIFIER — KNOWN TEST RESULTS")
    print("=" * 80)

    print(f"Macro F1    : {macro_f1:.4f}")
    print(f"Weighted F1 : {weighted_f1:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    labels = sorted(
        y_test.unique()
    )

    confusion = confusion_matrix(
        y_test,
        predictions,
        labels=labels,
    )

    print("\nConfusion matrix labels:")
    print(labels)

    print("\nConfusion matrix:")
    print(confusion)


if __name__ == "__main__":
    main()