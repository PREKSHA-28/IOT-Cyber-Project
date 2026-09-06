from dataclasses import dataclass, asdict
from typing import Any, Dict


@dataclass
class RAGRequest:
    """
    Structured request passed from the edge orchestrator
    to the RAG/LLM layer.

    Person 3 can use this object as the integration contract.
    """

    prediction: str
    raw_probability: float
    calibrated_probability: float
    confidence: float
    drift_detected: bool
    routing_reason: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert the request into a dictionary."""

        return asdict(self)


def create_rag_request(
    prediction: str,
    raw_probability: float,
    calibrated_probability: float,
    confidence: float,
    drift_detected: bool,
    routing_reason: str,
) -> RAGRequest:
    """
    Create the structured RAG request that Person 3's
    RAG/LLM layer can consume.
    """

    return RAGRequest(
        prediction=prediction,
        raw_probability=float(raw_probability),
        calibrated_probability=float(
            calibrated_probability
        ),
        confidence=float(confidence),
        drift_detected=bool(drift_detected),
        routing_reason=routing_reason,
    )