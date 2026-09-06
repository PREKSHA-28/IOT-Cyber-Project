from pathlib import Path

import joblib
import numpy as np
import pandas as pd


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

ROUTING_RESULTS_FILE = (
    RESULTS_DIR / "routing_evaluation_results.csv"
)


# ============================================================
# Configuration
# ============================================================

# Temporary threshold.
# We will later evaluate several thresholds and choose one
# based on experimental results.
CONFIDENCE_THRESHOLD = 0.70

RANDOM_STATE = 42

# Number of samples evaluated from each dataset.
# We use a fixed sample size first so the experiment is
# reasonably fast on a laptop.
KNOWN_SAMPLE_SIZE = 20_000
EMERGING_SAMPLE_SIZE = 20_000


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
# Evaluate one dataset
# ============================================================

def evaluate_dataset(
    model,
    calibrator,
    df: pd.DataFrame,
    dataset_name: str,
    drift_detected: bool,
) -> pd.DataFrame:
    """
    Generate model predictions, calibrated probabilities,
    confidence values, and routing decisions.
    """

    X = prepare_features(df)

    print(
        f"\nGenerating predictions for "
        f"{dataset_name}..."
    )

    probabilities = model.predict_proba(X)

    predictions = model.predict(X)

    raw_attack_probabilities = probabilities[:, 1]

    calibrated_attack_probabilities = (
        calibrator.predict(
            raw_attack_probabilities
        )
    )

    confidences = np.maximum(
        calibrated_attack_probabilities,
        1.0 - calibrated_attack_probabilities,
    )

    routing_decisions = [
        make_decision(
            float(confidence),
            drift_detected,
        )
        for confidence in confidences
    ]

    results = pd.DataFrame(
        {
            "dataset": dataset_name,
            "actual_label": df["Label"].to_numpy(),
            "prediction": np.where(
                predictions == 1,
                "ATTACK",
                "BENIGN",
            ),
            "raw_attack_probability": (
                raw_attack_probabilities
            ),
            "calibrated_attack_probability": (
                calibrated_attack_probabilities
            ),
            "confidence": confidences,
            "drift_detected": drift_detected,
            "routing_decision": routing_decisions,
        }
    )

    return results


# ============================================================
# Print summary
# ============================================================

def print_summary(
    results: pd.DataFrame,
    dataset_name: str,
) -> None:

    print("\n" + "=" * 70)
    print(
        f"{dataset_name} ROUTING SUMMARY"
    )
    print("=" * 70)

    total = len(results)

    local_count = np.sum(
        results["routing_decision"] == "LOCAL"
    )

    rag_count = np.sum(
        results["routing_decision"] == "RAG"
    )

    low_confidence_count = np.sum(
        results["confidence"]
        < CONFIDENCE_THRESHOLD
    )

    print(
        f"Total samples       : {total:,}"
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
        f"Mean confidence     : "
        f"{results['confidence'].mean():.6f}"
    )

    print(
        f"Mean attack prob.   : "
        f"{results['calibrated_attack_probability'].mean():.6f}"
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
    # Load known test
    # --------------------------------------------------------

    print("\nLoading known-test data...")

    known_df = pd.read_csv(
        KNOWN_TEST_FILE
    )

    # --------------------------------------------------------
    # Load emerging test
    # --------------------------------------------------------

    print("Loading emerging-test data...")

    emerging_df = pd.read_csv(
        EMERGING_TEST_FILE
    )

    # --------------------------------------------------------
    # Sample data
    # --------------------------------------------------------

    known_sample = known_df.sample(
        n=min(
            KNOWN_SAMPLE_SIZE,
            len(known_df),
        ),
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    emerging_sample = emerging_df.sample(
        n=min(
            EMERGING_SAMPLE_SIZE,
            len(emerging_df),
        ),
        random_state=RANDOM_STATE,
    ).reset_index(drop=True)

    print("\n" + "=" * 70)
    print("ROUTING EVALUATION")
    print("=" * 70)

    print(
        f"Known samples evaluated    : "
        f"{len(known_sample):,}"
    )

    print(
        f"Emerging samples evaluated : "
        f"{len(emerging_sample):,}"
    )

    print(
        f"Confidence threshold       : "
        f"{CONFIDENCE_THRESHOLD:.2f}"
    )

    # --------------------------------------------------------
    # Known traffic
    #
    # At this stage we use drift=False because the known-test
    # set represents the normal known distribution.
    # --------------------------------------------------------

    known_results = evaluate_dataset(
        model,
        calibrator,
        known_sample,
        "KNOWN",
        drift_detected=False,
    )

    # --------------------------------------------------------
    # Emerging traffic
    #
    # IMPORTANT:
    # We do NOT manually declare emerging traffic as drift.
    #
    # This experiment specifically measures whether confidence
    # alone causes selective escalation.
    #
    # Real ADWIN-based drift integration will be evaluated
    # separately.
    # --------------------------------------------------------

    emerging_results = evaluate_dataset(
        model,
        calibrator,
        emerging_sample,
        "EMERGING",
        drift_detected=False,
    )

    # --------------------------------------------------------
    # Print summaries
    # --------------------------------------------------------

    print_summary(
        known_results,
        "KNOWN",
    )

    print_summary(
        emerging_results,
        "EMERGING",
    )

    # --------------------------------------------------------
    # Combine results
    # --------------------------------------------------------

    all_results = pd.concat(
        [
            known_results,
            emerging_results,
        ],
        ignore_index=True,
    )

    # --------------------------------------------------------
    # Save detailed results
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    all_results.to_csv(
        ROUTING_RESULTS_FILE,
        index=False,
    )

    print("\n" + "=" * 70)
    print("RESULTS SAVED")
    print("=" * 70)

    print(
        ROUTING_RESULTS_FILE
    )

    # --------------------------------------------------------
    # Final comparison
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("KNOWN vs EMERGING")
    print("=" * 70)

    for dataset_name in [
        "KNOWN",
        "EMERGING",
    ]:

        subset = all_results[
            all_results["dataset"]
            == dataset_name
        ]

        rag_rate = (
            np.mean(
                subset["routing_decision"]
                == "RAG"
            )
            * 100
        )

        low_confidence_rate = (
            np.mean(
                subset["confidence"]
                < CONFIDENCE_THRESHOLD
            )
            * 100
        )

        print(
            f"\n{dataset_name}"
        )

        print(
            f"RAG invocation rate : "
            f"{rag_rate:.2f}%"
        )

        print(
            f"Low-confidence rate : "
            f"{low_confidence_rate:.2f}%"
        )


if __name__ == "__main__":
    main()