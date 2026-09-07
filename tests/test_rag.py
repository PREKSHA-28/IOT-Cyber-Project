"""Unit tests for the provenance-aware RAG pipeline."""

from __future__ import annotations

import unittest

from src.rag_contract import RAGRequest
from src.rag_pipeline import (
    DeterministicGroundedGenerator,
    GroundedRAGPipeline,
)
from src.rag_retriever import KnowledgeBase


class RagPipelineTests(unittest.TestCase):

    def setUp(self):
        self.knowledge_base = KnowledgeBase.from_json()

        self.pipeline = GroundedRAGPipeline(
            knowledge_base=self.knowledge_base,
            generator=DeterministicGroundedGenerator(),
        )

        self.request = RAGRequest(
            prediction="ATTACK",
            confidence=0.58,
            drift_detected=True,
            attack_probability=0.58,
            context={
                "retrieval_hint": (
                    "cybersecurity incident detection analysis "
                    "containment recovery"
                )
            },
            escalation_reason="LOW_CONFIDENCE_AND_DRIFT",
            routing_decision="RAG",
        )

    def test_rag_pipeline_returns_evidence(self):
        result = self.pipeline.run(
            self.request
        )

        self.assertTrue(
            result.response.evidence
        )

    def test_all_returned_evidence_has_provenance(self):
        result = self.pipeline.run(
            self.request
        )

        for evidence in result.response.evidence:
            self.assertTrue(
                evidence.evidence_id
            )
            self.assertTrue(
                evidence.source_name
            )
            self.assertTrue(
                evidence.document_title
            )
            self.assertTrue(
                evidence.source_url
            )
            self.assertTrue(
                evidence.excerpt
            )

    def test_response_contains_required_fields(self):
        result = self.pipeline.run(
            self.request
        )

        response = result.response

        self.assertTrue(
            response.threat
        )

        self.assertTrue(
            response.detection_interpretation
        )

        self.assertTrue(
            response.uncertainty_reason
        )

        self.assertTrue(
            response.likely_attack_behavior
        )

        self.assertTrue(
            response.recommendation
        )

        self.assertTrue(
            response.containment_action
        )

        self.assertTrue(
            response.caveat
        )

    def test_rag_pipeline_records_latency(self):
        result = self.pipeline.run(
            self.request
        )

        self.assertGreaterEqual(
            result.response.retrieval_latency_ms,
            0,
        )

        self.assertGreaterEqual(
            result.response.generation_latency_ms,
            0,
        )

        self.assertGreaterEqual(
            result.response.total_latency_ms,
            0,
        )

    def test_local_request_is_rejected(self):
        local_request = RAGRequest(
            prediction="BENIGN",
            confidence=0.99,
            drift_detected=False,
            attack_probability=0.01,
            routing_decision="LOCAL",
        )

        with self.assertRaisesRegex(
            ValueError,
            "LOCAL",
        ):
            self.pipeline.run(
                local_request
            )


if __name__ == "__main__":
    unittest.main()