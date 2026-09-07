"""Integration tests for Person 2's request and Person 3's RAG layer."""

from __future__ import annotations

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from rag_integration import RAGIntegration
from rag_pipeline import DeterministicGroundedGenerator
from orchestrator import EdgeOrchestrator
from rag_request import create_rag_request


class Person2RagIntegrationTests(unittest.TestCase):
    def test_orchestration_result_bridge_handles_rag_and_local(self):
        orchestrator = EdgeOrchestrator(confidence_threshold=0.70)
        rag_result = orchestrator.decide(
            prediction="ATTACK",
            raw_probability=0.62,
            calibrated_probability=0.58,
            confidence=0.58,
            drift_detected=True,
        )
        local_result = orchestrator.decide(
            prediction="BENIGN",
            raw_probability=0.01,
            calibrated_probability=0.01,
            confidence=0.99,
            drift_detected=False,
        )
        layer = RAGIntegration.create(DeterministicGroundedGenerator())

        rag_response = layer.run_orchestration_result(
            orchestrator,
            rag_result,
        )

        self.assertIsNotNone(rag_response)
        self.assertTrue(rag_response.response.evidence)
        self.assertIsNone(
            layer.run_orchestration_result(
                orchestrator,
                local_result,
            )
        )

    def test_orchestrator_to_rag_response(self):
        orchestrator = EdgeOrchestrator(confidence_threshold=0.70)
        orchestration_result = orchestrator.decide(
            prediction="ATTACK",
            raw_probability=0.62,
            calibrated_probability=0.58,
            confidence=0.58,
            drift_detected=True,
        )
        request = orchestrator.create_rag_request(orchestration_result)

        self.assertIsNotNone(request)
        self.assertEqual(
            set(request.to_dict()),
            {
                "prediction",
                "raw_probability",
                "calibrated_probability",
                "confidence",
                "drift_detected",
                "routing_reason",
            },
        )
        result = RAGIntegration.create(
            DeterministicGroundedGenerator()
        ).run(request)

        self.assertTrue(result.response.evidence)
        self.assertTrue(result.response.recommendation)

    def test_exact_request_reaches_grounded_response(self):
        request = create_rag_request(
            prediction="ATTACK",
            raw_probability=0.62,
            calibrated_probability=0.58,
            confidence=0.58,
            drift_detected=True,
            routing_reason="LOW_CONFIDENCE_AND_DRIFT",
        )

        self.assertEqual(
            set(request.to_dict()),
            {
                "prediction",
                "raw_probability",
                "calibrated_probability",
                "confidence",
                "drift_detected",
                "routing_reason",
            },
        )
        result = RAGIntegration.create(
            DeterministicGroundedGenerator()
        ).run(request)

        self.assertTrue(result.response.evidence)
        self.assertTrue(result.response.evidence[0].source_url)
        self.assertGreaterEqual(result.response.total_latency_ms, 0)

    def test_local_reason_does_not_invoke_rag(self):
        request = create_rag_request(
            prediction="BENIGN",
            raw_probability=0.01,
            calibrated_probability=0.01,
            confidence=0.99,
            drift_detected=False,
            routing_reason="HIGH_CONFIDENCE_NO_DRIFT",
        )

        with self.assertRaisesRegex(ValueError, "LOCAL"):
            RAGIntegration.create(
                DeterministicGroundedGenerator()
            ).run(request)


if __name__ == "__main__":
    unittest.main()
