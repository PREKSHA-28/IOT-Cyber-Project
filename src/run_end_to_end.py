"""
Run the complete IoT cybersecurity pipeline end-to-end.

Pipeline:

    Random Forest
        |
    Probability calibration
        |
    Confidence
        |
    ADWIN drift detection
        |
    Person 2 EdgeOrchestrator
        |
    LOCAL or RAG
        |
    Person 3 RAGIntegration
        |
    Provenance-aware RAG response
"""
from __future__ import annotations


from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from river.drift import ADWIN

from .orchestrator import EdgeOrchestrator
from .rag_integration import RAGIntegration
from .rag_pipeline import DeterministicGroundedGenerator


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
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

SPLIT_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
)

KNOWN_TEST_FILE = (
    SPLIT_DIR / "known_test.csv"
)

EMERGING_TEST_FILE = (
    SPLIT_DIR / "emerging_test.csv"
)

RESULTS_DIR = (
    PROJECT_ROOT / "results"
)

END_TO_END_RESULTS_FILE = (
    RESULTS_DIR
    / "end_to_end_results.csv"
)


# ============================================================
# Configuration
# ============================================================

CONFIDENCE_THRESHOLD = 0.70

ADWIN_DELTA = 0.002

RANDOM_STATE = 42

# Controlled stream:
# known traffic first, emerging traffic second.
KNOWN_STREAM_SIZE = 10_000
EMERGING_STREAM_SIZE = 10_000


# ============================================================
# Feature preparation
# ============================================================

def prepare_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Return only the 39 features used by the
    trained Random Forest.
    """

    feature_columns = [
        column
        for column in df.columns
        if column not in [
            "Label",
            "Label_Binary",
        ]
    ]

    return df[
        feature_columns
    ].copy()


def build_behavior_context(
    row: pd.Series,
) -> dict[str, object]:
    """
    Build a compact, label-free representation of observable
    network behavior for the RAG layer.

    The ground-truth Label and Label_Binary columns are deliberately
    excluded so that RAG cannot access the evaluation answer.
    """

    protocol_features = [
        "TCP",
        "UDP",
        "ARP",
        "ICMP",
        "IGMP",
        "IPv",
        "LLC",
    ]

    application_features = [
        "HTTP",
        "HTTPS",
        "DNS",
        "Telnet",
        "SMTP",
        "SSH",
        "IRC",
        "DHCP",
    ]

    tcp_flag_features = [
        "fin_flag_number",
        "syn_flag_number",
        "rst_flag_number",
        "psh_flag_number",
        "ack_flag_number",
        "ece_flag_number",
        "cwr_flag_number",
    ]

    traffic_features = [
        "Rate",
        "Number",
        "Tot size",
        "IAT",
        "Header_Length",
        "Min",
        "Max",
        "AVG",
        "Std",
        "Variance",
    ]

    def active_features(
        feature_names: list[str],
    ) -> list[str]:
        active = []

        for feature in feature_names:
            if feature not in row.index:
                continue

            try:
                value = float(row[feature])
            except (TypeError, ValueError):
                continue

            if value > 0:
                active.append(feature)

        return active

    def numeric_snapshot(
        feature_names: list[str],
    ) -> dict[str, float]:
        snapshot = {}

        for feature in feature_names:
            if feature not in row.index:
                continue

            try:
                value = float(row[feature])
            except (TypeError, ValueError):
                continue

            snapshot[feature] = round(value, 6)

        return snapshot

    active_protocols = active_features(
        protocol_features
    )

    active_application_protocols = active_features(
        application_features
    )

    active_tcp_flags = active_features(
        tcp_flag_features
    )

    behavior_terms = (
        active_protocols
        + active_application_protocols
        + active_tcp_flags
    )

    if not behavior_terms:
        behavior_terms = ["network traffic"]

    retrieval_query = (
        "Observed IoT network behavior: "
        + ", ".join(behavior_terms)
    )

    return {
        "retrieval_query": retrieval_query,
        "active_protocols": active_protocols,
        "active_application_protocols": (
            active_application_protocols
        ),
        "active_tcp_flags": active_tcp_flags,
        "traffic_statistics": numeric_snapshot(
            traffic_features
        ),
    }


# ============================================================
# Confidence calculation
# ============================================================

def calculate_confidence(
    calibrated_probability: float,
) -> float:
    """
    Calculate confidence as the probability of
    the predicted class.
    """

    return float(
        max(
            calibrated_probability,
            1.0 - calibrated_probability,
        )
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
                f"Required file not found:\n"
                f"{file_path}"
            )

    # --------------------------------------------------------
    # Load trained ML model
    # --------------------------------------------------------

    print(
        "Loading trained Random Forest..."
    )

    model = joblib.load(
        MODEL_FILE
    )

    # --------------------------------------------------------
    # Load probability calibrator
    # --------------------------------------------------------

    print(
        "Loading probability calibrator..."
    )

    calibrator = joblib.load(
        CALIBRATOR_FILE
    )

    # --------------------------------------------------------
    # Load test data
    # --------------------------------------------------------

    print(
        "\nLoading known-test data..."
    )

    known_df = pd.read_csv(
        KNOWN_TEST_FILE
    )

    print(
        "Loading emerging-test data..."
    )

    emerging_df = pd.read_csv(
        EMERGING_TEST_FILE
    )

    # --------------------------------------------------------
    # Create controlled stream
    # --------------------------------------------------------

    known_sample = known_df.sample(
        n=min(
            KNOWN_STREAM_SIZE,
            len(known_df),
        ),
        random_state=RANDOM_STATE,
    ).reset_index(
        drop=True
    )

    emerging_sample = emerging_df.sample(
        n=min(
            EMERGING_STREAM_SIZE,
            len(emerging_df),
        ),
        random_state=RANDOM_STATE,
    ).reset_index(
        drop=True
    )

    stream_df = pd.concat(
        [
            known_sample,
            emerging_sample,
        ],
        ignore_index=True,
    )

    phase_boundary = len(
        known_sample
    )

    # --------------------------------------------------------
    # Prepare model features
    # --------------------------------------------------------

    print(
        "\nPreparing features..."
    )

    X_stream = prepare_features(
        stream_df
    )

    # --------------------------------------------------------
    # Run actual ML model
    # --------------------------------------------------------

    print(
        "Generating Random Forest predictions..."
    )

    probabilities = model.predict_proba(
        X_stream
    )

    predictions = model.predict(
        X_stream
    )

    raw_attack_probabilities = (
        probabilities[:, 1]
    )

    # --------------------------------------------------------
    # Apply actual probability calibration
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
    # Initialize real ADWIN detector
    # --------------------------------------------------------

    print(
        "Initializing ADWIN..."
    )

    detector = ADWIN(
        delta=ADWIN_DELTA
    )

    # --------------------------------------------------------
    # Initialize Person 2 orchestrator
    # --------------------------------------------------------

    orchestrator = EdgeOrchestrator(
        confidence_threshold=CONFIDENCE_THRESHOLD
    )

    # --------------------------------------------------------
    # Initialize Person 3 RAG integration
    #
    # First end-to-end run uses the deterministic
    # grounded generator.
    # --------------------------------------------------------

    rag_layer = RAGIntegration.create(
        DeterministicGroundedGenerator()
    )

    # --------------------------------------------------------
    # Storage
    # --------------------------------------------------------

    results = []

    drift_events = []

    rag_responses = []

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    local_count = 0
    rag_count = 0

    drift_count = 0

    low_confidence_count = 0

    # --------------------------------------------------------
    # Process stream sequentially
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )
    print(
        "REAL END-TO-END PIPELINE"
    )
    print(
        "=" * 70
    )

    print(
        f"Known samples    : "
        f"{len(known_sample):,}"
    )

    print(
        f"Emerging samples : "
        f"{len(emerging_sample):,}"
    )

    print(
        f"Total samples    : "
        f"{len(stream_df):,}"
    )

    print(
        f"Confidence threshold : "
        f"{CONFIDENCE_THRESHOLD:.2f}"
    )

    print(
        f"ADWIN delta          : "
        f"{ADWIN_DELTA}"
    )

    print(
        "\nProcessing stream..."
    )

    for index in range(
        len(stream_df)
    ):

        sample_number = index + 1

        # ----------------------------------------------------
        # ML signals
        # ----------------------------------------------------

        raw_probability = float(
            raw_attack_probabilities[
                index
            ]
        )

        calibrated_probability = float(
            calibrated_attack_probabilities[
                index
            ]
        )

        prediction_value = int(
            predictions[index]
        )

        prediction = (
            "ATTACK"
            if prediction_value == 1
            else "BENIGN"
        )

        confidence = calculate_confidence(
            calibrated_probability
        )

        # ----------------------------------------------------
        # ADWIN receives the calibrated probability
        # one sample at a time.
        # ----------------------------------------------------

        detector.update(
            calibrated_probability
        )

        drift_detected = bool(
            detector.drift_detected
        )

        # ----------------------------------------------------
        # Person 2 orchestrator
        # ----------------------------------------------------

        orchestration_result = (
            orchestrator.decide(
                prediction=prediction,
                raw_probability=raw_probability,
                calibrated_probability=(
                    calibrated_probability
                ),
                confidence=confidence,
                drift_detected=drift_detected,
            )
        )

        # ----------------------------------------------------
        # Routing counters
        # ----------------------------------------------------

        if (
            orchestration_result.routing_decision
            == "LOCAL"
        ):

            local_count += 1

        else:

            rag_count += 1

        if confidence < CONFIDENCE_THRESHOLD:

            low_confidence_count += 1

        if drift_detected:

            drift_count += 1

        # ----------------------------------------------------
        # Determine phase
        # ----------------------------------------------------

        phase = (
            "KNOWN"
            if sample_number <= phase_boundary
            else "EMERGING"
        )

        # ----------------------------------------------------
        # Person 3 RAG branch
        # ----------------------------------------------------

        rag_response = None

        if (
            orchestration_result.routing_decision
            == "RAG"
        ):

            sample_row = stream_df.iloc[index]

            behavior_context = build_behavior_context(
                sample_row
            )

            rag_response = (
                rag_layer.run_orchestration_result(
                    orchestrator,
                    orchestration_result,
                    behavior_context=behavior_context,
                )
            )

            if rag_response is None:

                raise RuntimeError(
                    "Orchestrator selected RAG "
                    "but no RAG response was produced."
                )

            rag_responses.append(
                {
                    "sample_index": sample_number,
                    "phase": phase,
                    "actual_label": stream_df[
                        "Label"
                    ].iloc[index],
                    "prediction": prediction,
                    "confidence": confidence,
                    "drift_detected": (
                        drift_detected
                    ),
                    "routing_reason": (
                        orchestration_result
                        .routing_reason
                    ),
                    "threat": (
                        rag_response.response.threat
                    ),
                    "recommendation": (
                        rag_response.response
                        .recommendation
                    ),
                    "retrieval_latency_ms": (
                        rag_response.response
                        .retrieval_latency_ms
                    ),
                    "generation_latency_ms": (
                        rag_response.response
                        .generation_latency_ms
                    ),
                    "total_rag_latency_ms": (
                        rag_response.response
                        .total_latency_ms
                    ),
                    "evidence_count": len(
                        rag_response.response
                        .evidence
                    ),
                    "retrieved_evidence_ids": "|".join(
                        evidence.evidence_id
                        for evidence in rag_response.response.evidence
                    ),
                }
            )

        # ----------------------------------------------------
        # Save sample-level pipeline result
        # ----------------------------------------------------

        results.append(
            {
                "sample_index": sample_number,
                "phase": phase,
                "actual_label": stream_df[
                    "Label"
                ].iloc[index],
                "prediction": prediction,
                "raw_attack_probability": (
                    raw_probability
                ),
                "calibrated_attack_probability": (
                    calibrated_probability
                ),
                "confidence": confidence,
                "drift_detected": (
                    drift_detected
                ),
                "routing_decision": (
                    orchestration_result
                    .routing_decision
                ),
                "routing_reason": (
                    orchestration_result
                    .routing_reason
                ),
                "rag_invoked": (
                    orchestration_result
                    .routing_decision
                    == "RAG"
                ),
            }
        )

        # ----------------------------------------------------
        # Record drift event
        # ----------------------------------------------------

        if drift_detected:

            drift_event = {
                "sample_index": sample_number,
                "phase": phase,
                "actual_label": stream_df[
                    "Label"
                ].iloc[index],
                "prediction": prediction,
                "confidence": confidence,
                "calibrated_attack_probability": (
                    calibrated_probability
                ),
                "routing_decision": (
                    orchestration_result
                    .routing_decision
                ),
                "routing_reason": (
                    orchestration_result
                    .routing_reason
                ),
            }

            drift_events.append(
                drift_event
            )

            print(
                f"Drift detected at sample "
                f"{sample_number:,} "
                f"({phase})"
            )

    # --------------------------------------------------------
    # Convert to DataFrames
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    rag_df = pd.DataFrame(
        rag_responses
    )

    drift_df = pd.DataFrame(
        drift_events
    )

    # --------------------------------------------------------
    # Overall summary
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )
    print(
        "END-TO-END SUMMARY"
    )
    print(
        "=" * 70
    )

    total_samples = len(
        results_df
    )

    print(
        f"Total samples       : "
        f"{total_samples:,}"
    )

    print(
        f"LOCAL decisions     : "
        f"{local_count:,} "
        f"({local_count / total_samples * 100:.2f}%)"
    )

    print(
        f"RAG decisions       : "
        f"{rag_count:,} "
        f"({rag_count / total_samples * 100:.2f}%)"
    )

    print(
        f"Low-confidence      : "
        f"{low_confidence_count:,} "
        f"({low_confidence_count / total_samples * 100:.2f}%)"
    )

    print(
        f"Drift detections    : "
        f"{drift_count:,}"
    )

    # --------------------------------------------------------
    # Phase summaries
    # --------------------------------------------------------

    for phase in [
        "KNOWN",
        "EMERGING",
    ]:

        subset = results_df[
            results_df["phase"] == phase
        ]

        total = len(subset)

        local_phase = np.sum(
            subset["routing_decision"]
            == "LOCAL"
        )

        rag_phase = np.sum(
            subset["routing_decision"]
            == "RAG"
        )

        drift_phase = np.sum(
            subset["drift_detected"]
        )

        print(
            "\n" + "-" * 70
        )

        print(
            phase
        )

        print(
            f"Samples             : "
            f"{total:,}"
        )

        print(
            f"LOCAL               : "
            f"{local_phase:,} "
            f"({local_phase / total * 100:.2f}%)"
        )

        print(
            f"RAG                 : "
            f"{rag_phase:,} "
            f"({rag_phase / total * 100:.2f}%)"
        )

        print(
            f"Drift detections    : "
            f"{drift_phase:,}"
        )

        print(
            f"Mean confidence     : "
            f"{subset['confidence'].mean():.6f}"
        )

    # --------------------------------------------------------
    # RAG summary
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )
    print(
        "RAG EXECUTION SUMMARY"
    )
    print(
        "=" * 70
    )

    print(
        f"RAG requests executed : "
        f"{len(rag_df):,}"
    )

    if not rag_df.empty:

        print(
            f"Mean retrieval latency : "
            f"{rag_df['retrieval_latency_ms'].mean():.6f} ms"
        )

        print(
            f"Mean generation latency: "
            f"{rag_df['generation_latency_ms'].mean():.6f} ms"
        )

        print(
            f"Mean total RAG latency : "
            f"{rag_df['total_rag_latency_ms'].mean():.6f} ms"
        )

        print(
            f"Mean evidence count    : "
            f"{rag_df['evidence_count'].mean():.2f}"
        )

        print(
            "\nFirst RAG response:"
        )

        first_rag = rag_df.iloc[0]

        print(
            f"Sample index      : "
            f"{first_rag['sample_index']}"
        )

        print(
            f"Phase             : "
            f"{first_rag['phase']}"
        )

        print(
            f"Actual label      : "
            f"{first_rag['actual_label']}"
        )

        print(
            f"Prediction        : "
            f"{first_rag['prediction']}"
        )

        print(
            f"Confidence        : "
            f"{first_rag['confidence']:.6f}"
        )

        print(
            f"Drift detected    : "
            f"{first_rag['drift_detected']}"
        )

        print(
            f"Routing reason    : "
            f"{first_rag['routing_reason']}"
        )

        print(
            f"Threat            : "
            f"{first_rag['threat']}"
        )

        print(
            f"Recommendation    : "
            f"{first_rag['recommendation']}"
        )

    else:

        print(
            "No RAG requests were executed."
        )

    # --------------------------------------------------------
    # Drift summary
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )
    print(
        "DRIFT SUMMARY"
    )
    print(
        "=" * 70
    )

    print(
        f"Total drift events: "
        f"{len(drift_df):,}"
    )

    if not drift_df.empty:

        first_drift = drift_df.iloc[0]

        print(
            f"First drift sample : "
            f"{first_drift['sample_index']}"
        )

        print(
            f"Drift phase        : "
            f"{first_drift['phase']}"
        )

        print(
            f"Drift routing      : "
            f"{first_drift['routing_decision']}"
        )

        print(
            f"Drift reason       : "
            f"{first_drift['routing_reason']}"
        )

    else:

        print(
            "No drift was detected."
        )

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    results_df.to_csv(
        END_TO_END_RESULTS_FILE,
        index=False,
    )

    rag_results_file = (
        RESULTS_DIR
        / "end_to_end_rag_results.csv"
    )

    rag_df.to_csv(
        rag_results_file,
        index=False,
    )

    drift_results_file = (
        RESULTS_DIR
        / "end_to_end_drift_events.csv"
    )

    drift_df.to_csv(
        drift_results_file,
        index=False,
    )

    # --------------------------------------------------------
    # Final message
    # --------------------------------------------------------

    print(
        "\n" + "=" * 70
    )
    print(
        "END-TO-END PIPELINE COMPLETED"
    )
    print(
        "=" * 70
    )

    print(
        "\nSaved:"
    )

    print(
        END_TO_END_RESULTS_FILE
    )

    print(
        rag_results_file
    )

    print(
        drift_results_file
    )


if __name__ == "__main__":
    main()
