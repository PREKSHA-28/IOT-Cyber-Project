"""Adapter from Person 2's exact RAG request to Person 3's RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rag_contract import RAGRequest as Person3Request
from orchestrator import OrchestrationResult, EdgeOrchestrator
from rag_pipeline import GroundedGenerator, GroundedRAGPipeline, PipelineResult
from rag_request import RAGRequest as OrchestratorRAGRequest
from rag_retriever import KnowledgeBase


_LOCAL_ROUTING_REASON = "HIGH_CONFIDENCE_NO_DRIFT"


@dataclass(frozen=True)
class RAGIntegration:
    """Connect Person 2's request factory to the existing Person 3 pipeline."""

    pipeline: GroundedRAGPipeline

    @classmethod
    def create(
        cls,
        generator: GroundedGenerator,
        knowledge_base: KnowledgeBase | None = None,
        top_k: int = 4,
        min_score: float = 0.05,
    ) -> "RAGIntegration":
        return cls(
            pipeline=GroundedRAGPipeline(
                knowledge_base or KnowledgeBase.from_json(),
                generator,
                top_k=top_k,
                min_score=min_score,
            )
        )

    def run(self, request: OrchestratorRAGRequest) -> PipelineResult:
        """Process an already-escalated Person 2 request.

        The incoming request is consumed exactly as defined by Person 2. The
        additional fields below are an internal projection only and are never
        returned as part of the Person 2 request.
        """

        if request.routing_reason == _LOCAL_ROUTING_REASON:
            raise ValueError("LOCAL routing reason must not invoke the RAG layer")

        internal_request = Person3Request(
            prediction=request.prediction,
            confidence=request.confidence,
            drift_detected=request.drift_detected,
            attack_probability=request.calibrated_probability,
            context=self._build_context(request),
            escalation_reason=request.routing_reason,
            routing_decision="RAG",
        )
        return self.pipeline.run(internal_request)

    def run_orchestration_result(
        self,
        orchestrator: EdgeOrchestrator,
        result: OrchestrationResult,
    ) -> PipelineResult | None:
        """Run RAG only when Person 2's result is routed to RAG."""

        request = orchestrator.create_rag_request(result)
        if request is None:
            return None
        return self.run(request)

    @staticmethod
    def _build_context(request: OrchestratorRAGRequest) -> dict[str, Any]:
        return {
            "retrieval_hint": (
                "cybersecurity incident detection analysis containment recovery"
            ),
            "raw_probability": request.raw_probability,
            "calibrated_probability": request.calibrated_probability,
            "confidence": request.confidence,
            "drift_detected": request.drift_detected,
            "routing_reason": request.routing_reason,
        }
