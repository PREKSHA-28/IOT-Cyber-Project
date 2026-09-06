from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
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

MODEL_DIR = PROJECT_ROOT / "models"


TRAIN_FILE = SPLIT_DIR / "train.csv"
VALIDATION_FILE = SPLIT_DIR / "validation.csv"

MODEL_FILE = MODEL_DIR / "binary_ids_random_forest.joblib"


# ============================================================
# Configuration
# ============================================================

RANDOM_STATE = 42

N_ESTIMATORS = 100
MAX_DEPTH = 20


# ============================================================
# Feature preparation
# ============================================================

def prepare_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate model features from target.

    Label and Label_Binary are targets/metadata and must never
    be given to the ML model as input features.
    """

    target = (
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

    features = df[feature_columns].copy()

    return features, target


# ============================================================
# Main
# ============================================================

def main() -> None:

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not TRAIN_FILE.exists():
        raise FileNotFoundError(
            f"Training file not found:\n{TRAIN_FILE}"
        )

    if not VALIDATION_FILE.exists():
        raise FileNotFoundError(
            f"Validation file not found:\n{VALIDATION_FILE}"
        )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("Loading training data...")
    train_df = pd.read_csv(TRAIN_FILE)

    print("Loading validation data...")
    validation_df = pd.read_csv(VALIDATION_FILE)

    print("\nTraining rows:", f"{len(train_df):,}")
    print("Validation rows:", f"{len(validation_df):,}")

    # --------------------------------------------------------
    # Prepare features and targets
    # --------------------------------------------------------

    X_train, y_train = prepare_features(train_df)
    X_validation, y_validation = prepare_features(
        validation_df
    )

    print("\nNumber of input features:", X_train.shape[1])

    print("\nTraining label distribution:")
    print(y_train.value_counts())

    print("\nValidation label distribution:")
    print(y_validation.value_counts())

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    print("\nCreating Random Forest...")

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\nTraining model...")
    model.fit(X_train, y_train)

    print("Training completed.")

    # --------------------------------------------------------
    # Validation prediction
    # --------------------------------------------------------

    print("\nRunning validation...")

    predictions = model.predict(X_validation)

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    precision = precision_score(
        y_validation,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        y_validation,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        y_validation,
        predictions,
        zero_division=0,
    )

    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1-score  : {f1:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_validation,
            predictions,
            target_names=["BENIGN", "ATTACK"],
            zero_division=0,
        )
    )

    print("Confusion matrix:")
    print(
        confusion_matrix(
            y_validation,
            predictions
        )
    )

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_FILE
    )

    print("\n" + "=" * 70)
    print("MODEL SAVED")
    print("=" * 70)

    print(MODEL_FILE)


if __name__ == "__main__":
    main()