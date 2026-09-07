"""Stable request and response contracts for selective cybersecurity RAG."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Mapping


_ALLOWED_PREDICTIONS = {"ATTACK", "BENIGN", "UNKNOWN"}
_ALLOWED_ROUTING_DECISIONS = {"RAG", "LOCAL"}


def _utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _require_probability(name: str, value: float) -> float:
    numeric_value = float(value)
    if not 0.0 <= numeric_value <= 1.0:
        raise ValueError(f"{name} must be between 0.0 and 1.0")
    return numeric_value


@dataclass(frozen=True)
class RAGRequest:
    """Evidence request emitted by the edge/orchestration layer."""

    prediction: str
    confidence: float
    drift_detected: bool
    attack_probability: float
    context: Mapping[str, Any] = field(default_factory=dict)
    escalation_reason: str | None = None
    request_id: str | None = None
    routing_decision: str = "RAG"
    created_at: str = field(default_factory=_utc_timestamp)

    def __post_init__(self) -> None:
        prediction = self.prediction.upper()
        if prediction not in _ALLOWED_PREDICTIONS:
            raise ValueError(
                f"prediction must be one of {sorted(_ALLOWED_PREDICTIONS)}"
            )
        object.__setattr__(self, "prediction", prediction)
        object.__setattr__(
            self,
            "confidence",
            _require_probability("confidence", self.confidence),
        )
        object.__setattr__(
            self,
            "attack_probability",
            _require_probability(
                "attack_probability",
                self.attack_probability,
            ),
        )
        routing_decision = self.routing_decision.upper()
        if routing_decision not in _ALLOWED_ROUTING_DECISIONS:
            raise ValueError(
                "routing_decision must be either 'LOCAL' or 'RAG'"
            )
        object.__setattr__(self, "routing_decision", routing_decision)
        if not isinstance(self.context, Mapping):
            raise TypeError("context must be a mapping")

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "RAGRequest":
        """Construct a request from the orchestrator's JSON object."""

        required_fields = {
            "prediction",
            "confidence",
            "drift_detected",
            "attack_probability",
        }
        missing_fields = required_fields - payload.keys()
        if missing_fields:
            raise ValueError(
                f"RAG request is missing fields: {sorted(missing_fields)}"
            )
        return cls(
            prediction=payload["prediction"],
            confidence=payload["confidence"],
            drift_detected=payload["drift_detected"],
            attack_probability=payload["attack_probability"],
            context=payload.get("context", {}),
            escalation_reason=payload.get("escalation_reason"),
            request_id=payload.get("request_id"),
            routing_decision=payload.get("routing_decision", "RAG"),
            created_at=payload.get("created_at", _utc_timestamp()),
        )

    @classmethod
    def from_edge_result(
        cls,
        payload: Mapping[str, Any],
        context: Mapping[str, Any] | None = None,
        request_id: str | None = None,
    ) -> "RAGRequest":
        """Convert the current edge-routing result into a RAG request."""

        required_fields = {
            "prediction",
            "confidence",
            "drift_detected",
            "calibrated_attack_probability",
        }
        missing_fields = required_fields - payload.keys()
        if missing_fields:
            raise ValueError(
                f"edge result is missing fields: {sorted(missing_fields)}"
            )
        drift_detected = payload["drift_detected"]
        if isinstance(drift_detected, str):
            drift_detected = drift_detected.strip().lower() == "true"
        confidence = float(payload["confidence"])
        return cls(
            prediction=payload["prediction"],
            confidence=confidence,
            drift_detected=drift_detected,
            attack_probability=payload["calibrated_attack_probability"],
            context=context or {},
            escalation_reason=(
                "DRIFT_DETECTED_AND_LOW_CONFIDENCE"
                if drift_detected and confidence < 0.70
                else "DRIFT_DETECTED"
                if drift_detected
                else "LOW_CONFIDENCE"
                if confidence < 0.70
                else "NO_ESCALATION"
            ),
            request_id=request_id,
            routing_decision=payload.get("routing_decision", "RAG"),
        )

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable request object."""

        return asdict(self)


@dataclass(frozen=True)
class EvidenceReference:
    """A provenance record for one retrieved knowledge item or chunk."""

    evidence_id: str
    source_name: str
    document_title: str
    source_url: str
    document_type: str
    excerpt: str
    relevance_score: float
    chunk_id: str | None = None
    publication_date: str | None = None
    retrieved_at: str = field(default_factory=_utc_timestamp)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "relevance_score",
            _require_probability("relevance_score", self.relevance_score),
        )
        if not self.excerpt.strip():
            raise ValueError("evidence excerpt must not be empty")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RAGResponse:
    """Structured, traceable result returned to the orchestrator."""

    request_id: str | None
    threat: str
    detection_interpretation: str
    uncertainty_reason: str
    likely_attack_behavior: str
    recommendation: str
    containment_action: str
    caveat: str
    evidence: tuple[EvidenceReference, ...] = ()
    retrieval_latency_ms: float | None = None
    generation_latency_ms: float | None = None
    total_latency_ms: float | None = None
    generated_at: str = field(default_factory=_utc_timestamp)

    def __post_init__(self) -> None:
        for field_name in (
            "threat",
            "detection_interpretation",
            "uncertainty_reason",
            "likely_attack_behavior",
            "recommendation",
            "containment_action",
            "caveat",
        ):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{field_name} must not be empty")
        for field_name in (
            "retrieval_latency_ms",
            "generation_latency_ms",
            "total_latency_ms",
        ):
            value = getattr(self, field_name)
            if value is not None and float(value) < 0:
                raise ValueError(f"{field_name} must not be negative")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable response with explicit evidence."""

        response = asdict(self)
        response["evidence"] = [item.to_dict() for item in self.evidence]
        return response
