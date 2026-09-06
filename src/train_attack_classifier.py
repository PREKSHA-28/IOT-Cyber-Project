from pathlib import Path

import joblib
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
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

MODEL_DIR = PROJECT_ROOT / "models"

TRAIN_FILE = SPLIT_DIR / "train.csv"
VALIDATION_FILE = SPLIT_DIR / "validation.csv"

MODEL_FILE = (
    MODEL_DIR
    / "attack_type_random_forest.joblib"
)


# ============================================================
# Configuration
# ============================================================

RANDOM_STATE = 42

N_ESTIMATORS = 100
MAX_DEPTH = 25


# ============================================================
# Feature preparation
# ============================================================

def prepare_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Prepare features and multiclass attack target.

    BENIGN rows are removed because Stage 2 is executed only
    after Stage 1 has identified traffic as ATTACK.
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

    # --------------------------------------------------------
    # Check required files
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

    print(
        f"\nTraining rows before removing BENIGN: "
        f"{len(train_df):,}"
    )

    print(
        f"Validation rows before removing BENIGN: "
        f"{len(validation_df):,}"
    )

    # --------------------------------------------------------
    # Prepare attack-only datasets
    # --------------------------------------------------------

    X_train, y_train = prepare_features(train_df)

    X_validation, y_validation = prepare_features(
        validation_df
    )

    print(
        f"\nTraining attack rows: "
        f"{len(X_train):,}"
    )

    print(
        f"Validation attack rows: "
        f"{len(X_validation):,}"
    )

    print(
        f"\nNumber of input features: "
        f"{X_train.shape[1]}"
    )

    print(
        f"Number of attack classes: "
        f"{y_train.nunique()}"
    )

    # --------------------------------------------------------
    # Show class distribution
    # --------------------------------------------------------

    print("\nTraining attack distribution:")
    print(y_train.value_counts())

    # --------------------------------------------------------
    # Verify emerging attack classes are absent
    # --------------------------------------------------------

    emerging_attacks = {
        "DNS_SPOOFING",
        "VULNERABILITYSCAN",
        "DOS-HTTP_FLOOD",
    }

    leaked_classes = (
        set(y_train.unique())
        .intersection(emerging_attacks)
    )

    if leaked_classes:
        raise ValueError(
            "Emerging attack leakage detected: "
            f"{leaked_classes}"
        )

    print("\nEmerging attack leakage check: PASSED")

    # --------------------------------------------------------
    # Create Random Forest
    # --------------------------------------------------------

    print("\nCreating multiclass Random Forest...")

    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        max_depth=MAX_DEPTH,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print("\nTraining attack classifier...")

    model.fit(
        X_train,
        y_train,
    )

    print("Training completed.")

    # --------------------------------------------------------
    # Validation prediction
    # --------------------------------------------------------

    print("\nRunning validation...")

    predictions = model.predict(
        X_validation
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    macro_f1 = f1_score(
        y_validation,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        y_validation,
        predictions,
        average="weighted",
        zero_division=0,
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n" + "=" * 80)
    print("ATTACK-TYPE CLASSIFIER VALIDATION RESULTS")
    print("=" * 80)

    print(f"Macro F1    : {macro_f1:.4f}")
    print(f"Weighted F1 : {weighted_f1:.4f}")

    print("\nClassification report:")

    print(
        classification_report(
            y_validation,
            predictions,
            zero_division=0,
        )
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    labels = sorted(
        y_validation.unique()
    )

    confusion = confusion_matrix(
        y_validation,
        predictions,
        labels=labels,
    )

    print("Confusion matrix labels:")
    print(labels)

    print("\nConfusion matrix:")
    print(confusion)

    # --------------------------------------------------------
    # Save model
    # --------------------------------------------------------

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    joblib.dump(
        model,
        MODEL_FILE,
    )

    print("\n" + "=" * 80)
    print("MODEL SAVED")
    print("=" * 80)

    print(MODEL_FILE)


if __name__ == "__main__":
    main()