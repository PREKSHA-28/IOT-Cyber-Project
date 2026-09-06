from dataclasses import dataclass, asdict
from typing import Any, Dict

from rag_request import create_rag_request


@dataclass
class OrchestrationResult:
    prediction: str
    raw_probability: float
    calibrated_probability: float
    confidence: float
    drift_detected: bool
    routing_decision: str
    routing_reason: str

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EdgeOrchestrator:
    """
    Adaptive edge orchestration component.

    Routes traffic to LOCAL or RAG based on:
    1. Prediction confidence
    2. Drift detection
    """

    def __init__(self, confidence_threshold: float = 0.70):
        if not 0.0 <= confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1")

        self.confidence_threshold = confidence_threshold

    @staticmethod
    def calculate_confidence(calibrated_probability: float) -> float:
        """
        Calculate prediction confidence from calibrated attack probability.
        """
        if not 0.0 <= calibrated_probability <= 1.0:
            raise ValueError(
                "calibrated_probability must be between 0 and 1"
            )

        return max(
            calibrated_probability,
            1.0 - calibrated_probability
        )

    def decide(
        self,
        prediction: str,
        raw_probability: float,
        calibrated_probability: float,
        confidence: float,
        drift_detected: bool,
    ) -> OrchestrationResult:

        if not 0.0 <= raw_probability <= 1.0:
            raise ValueError("raw_probability must be between 0 and 1")

        if not 0.0 <= calibrated_probability <= 1.0:
            raise ValueError(
                "calibrated_probability must be between 0 and 1"
            )

        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be between 0 and 1")

        low_confidence = confidence < self.confidence_threshold

        # Both uncertainty and drift detected
        if low_confidence and drift_detected:
            decision = "RAG"
            reason = "LOW_CONFIDENCE_AND_DRIFT"

        # Drift detected even though confidence is high
        elif drift_detected:
            decision = "RAG"
            reason = "DRIFT_DETECTED"

        # Confidence is too low
        elif low_confidence:
            decision = "RAG"
            reason = "LOW_CONFIDENCE"

        # Confident prediction and no drift
        else:
            decision = "LOCAL"
            reason = "HIGH_CONFIDENCE_NO_DRIFT"

        return OrchestrationResult(
            prediction=prediction,
            raw_probability=float(raw_probability),
            calibrated_probability=float(calibrated_probability),
            confidence=float(confidence),
            drift_detected=bool(drift_detected),
            routing_decision=decision,
            routing_reason=reason,
        )

    def create_rag_request(
        self,
        result: OrchestrationResult
    ):
        """
        Convert a RAG routing decision into a structured
        RAG request for the RAG/LLM layer.
        """

        # LOCAL decisions do not need a RAG request
        if result.routing_decision != "RAG":
            return None

        return create_rag_request(
            prediction=result.prediction,
            raw_probability=result.raw_probability,
            calibrated_probability=result.calibrated_probability,
            confidence=result.confidence,
            drift_detected=result.drift_detected,
            routing_reason=result.routing_reason,
        )


def create_orchestrator(
    confidence_threshold: float = 0.70
) -> EdgeOrchestrator:

    return EdgeOrchestrator(
        confidence_threshold=confidence_threshold
    )