"""Automated checks for the provenance-aware RAG boundary."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_contract import RAGRequest
from rag_pipeline import DeterministicGroundedGenerator, GroundedRAGPipeline
from rag_retriever import KnowledgeBase


class InvalidCitationGenerator:
    def generate(self, request, evidence, prompt_context):
        del request, evidence, prompt_context
        return {
            "threat": "Unsupported claim",
            "detection_interpretation": "Unsupported interpretation",
            "uncertainty_reason": "Unknown",
            "likely_attack_behavior": "Unknown",
            "recommendation": "Unknown",
            "containment_action": "Hypothetical only: do nothing automatically.",
            "caveat": "Verification required.",
            "evidence_ids": ["invented-source-id"],
        }


class RagBoundaryTests(unittest.TestCase):
    def setUp(self):
        self.knowledge_base = KnowledgeBase.from_json()
        self.request = RAGRequest(
            prediction="ATTACK",
            confidence=0.42,
            drift_detected=True,
            attack_probability=0.78,
            escalation_reason="LOW_CONFIDENCE_AND_DRIFT",
            context={"behavior": "network service scanning repeated port probes"},
        )

    def test_retrieval_preserves_provenance(self):
        result = self.knowledge_base.search(self.request, top_k=3)
        self.assertTrue(result.evidence)
        self.assertEqual(
            result.evidence[0].evidence_id,
            "mitre-attack-t1046-network-service-scanning",
        )
        self.assertTrue(result.evidence[0].source_url)

    def test_pipeline_returns_structured_grounded_response(self):
        result = GroundedRAGPipeline(
            self.knowledge_base,
            DeterministicGroundedGenerator(),
        ).run(self.request)
        self.assertTrue(result.response.evidence)
        self.assertGreaterEqual(result.response.total_latency_ms, 0)
        self.assertEqual(
            result.response.evidence[0].evidence_id,
            result.retrieved.evidence[0].evidence_id,
        )

    def test_pipeline_rejects_unknown_citation(self):
        with self.assertRaisesRegex(ValueError, "not retrieved"):
            GroundedRAGPipeline(
                self.knowledge_base,
                InvalidCitationGenerator(),
            ).run(self.request)

    def test_pipeline_rejects_local_request(self):
        local_request = RAGRequest(
            prediction="BENIGN",
            confidence=0.99,
            drift_detected=False,
            attack_probability=0.01,
            routing_decision="LOCAL",
        )
        with self.assertRaisesRegex(ValueError, "LOCAL"):
            GroundedRAGPipeline(
                self.knowledge_base,
                DeterministicGroundedGenerator(),
            ).run(local_request)

    def test_edge_row_conversion_preserves_local_decision(self):
        request = RAGRequest.from_edge_result(
            {
                "prediction": "BENIGN",
                "confidence": "0.99",
                "drift_detected": "False",
                "calibrated_attack_probability": "0.01",
                "routing_decision": "LOCAL",
            }
        )
        self.assertEqual(request.routing_decision, "LOCAL")
        self.assertEqual(request.escalation_reason, "NO_ESCALATION")


if __name__ == "__main__":
    unittest.main()
