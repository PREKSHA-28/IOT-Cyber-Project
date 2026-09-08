"""Adapter from Person 2's exact RAG request to Person 3's RAG pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .rag_contract import RAGRequest as Person3Request
from .orchestrator import (
    OrchestrationResult,
    EdgeOrchestrator,
)
from .rag_pipeline import (
    GroundedGenerator,
    GroundedRAGPipeline,
    PipelineResult,
)
from .rag_request import (
    RAGRequest as OrchestratorRAGRequest,
)
from .rag_retriever import KnowledgeBase


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = (
    Path(__file__).resolve().parent.parent
)

EXPANDED_CORPUS_FILE = (
    PROJECT_ROOT
    / "knowledge_base"
    / "cybersecurity_sources_expanded.json"
)


_LOCAL_ROUTING_REASON = (
    "HIGH_CONFIDENCE_NO_DRIFT"
)


# ============================================================
# RAG integration
# ============================================================

@dataclass(frozen=True)
class RAGIntegration:
    """
    Connect Person 2's request factory to the existing
    Person 3 RAG pipeline.
    """

    pipeline: GroundedRAGPipeline

    @classmethod
    def create(
        cls,
        generator: GroundedGenerator,
        knowledge_base: KnowledgeBase | None = None,
        top_k: int = 4,
        min_score: float = 0.05,
    ) -> "RAGIntegration":
        """
        Create the integrated RAG layer.

        By default, the expanded 37-record cybersecurity corpus
        is used. A custom KnowledgeBase can still be supplied for
        experiments such as retrieval-method comparisons.
        """

        if knowledge_base is None:

            if not EXPANDED_CORPUS_FILE.exists():
                raise FileNotFoundError(
                    "Expanded cybersecurity corpus not found:\n"
                    f"{EXPANDED_CORPUS_FILE}"
                )

            knowledge_base = (
                KnowledgeBase.from_json(
                    EXPANDED_CORPUS_FILE
                )
            )

        return cls(
            pipeline=GroundedRAGPipeline(
                knowledge_base,
                generator,
                top_k=top_k,
                min_score=min_score,
            )
        )


    # ========================================================
    # Process RAG request
    # ========================================================

    def run(
        self,
        request: OrchestratorRAGRequest,
    ) -> PipelineResult:
        """
        Process an already-escalated Person 2 request.
        """

        if (
            request.routing_reason
            == _LOCAL_ROUTING_REASON
        ):
            raise ValueError(
                "LOCAL routing reason must not invoke "
                "the RAG layer"
            )

        internal_request = Person3Request(
            prediction=request.prediction,
            confidence=request.confidence,
            drift_detected=request.drift_detected,
            attack_probability=(
                request.calibrated_probability
            ),
            context=self._build_context(
                request
            ),
            escalation_reason=(
                request.routing_reason
            ),
            routing_decision="RAG",
        )

        return self.pipeline.run(
            internal_request
        )


    # ========================================================
    # Run from orchestration result
    # ========================================================

    def run_orchestration_result(
        self,
        orchestrator: EdgeOrchestrator,
        result: OrchestrationResult,
        behavior_context: dict[str, Any] | None = None,
    ) -> PipelineResult | None:
        """
        Run RAG only when Person 2's result is routed to RAG.

        behavior_context contains observable network behavior
        extracted from the original input sample.
        """

        request = (
            orchestrator.create_rag_request(
                result,
                behavior_context=(
                    behavior_context
                ),
            )
        )

        if request is None:
            return None

        return self.run(
            request
        )


    # ========================================================
    # Context construction
    # ========================================================

    @staticmethod
    def _build_context(
        request: OrchestratorRAGRequest,
    ) -> dict[str, Any]:
        """
        Build the context consumed by the RAG pipeline.

        The generic incident-response hint is retained as
        supporting context, while observable network behavior
        is passed separately so the retriever can prioritize
        attack-specific terms.
        """

        context: dict[str, Any] = {
            "retrieval_hint": (
                "cybersecurity incident detection "
                "analysis containment recovery"
            ),
            "raw_probability": (
                request.raw_probability
            ),
            "calibrated_probability": (
                request.calibrated_probability
            ),
            "confidence": request.confidence,
            "drift_detected": (
                request.drift_detected
            ),
            "routing_reason": (
                request.routing_reason
            ),
        }

        if request.behavior_context:

            context["behavior_context"] = (
                request.behavior_context
            )

        return context