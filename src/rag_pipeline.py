"""Selective, provenance-constrained RAG orchestration."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from rag_contract import EvidenceReference, RAGRequest, RAGResponse
from rag_retriever import KnowledgeBase, RetrievedEvidence


class GroundedGenerator(Protocol):
    """Adapter implemented by a local or hosted LLM provider."""

    def generate(
        self,
        request: RAGRequest,
        evidence: tuple[EvidenceReference, ...],
        prompt_context: str,
    ) -> Mapping[str, Any]:
        """Return structured fields and evidence IDs only from supplied evidence."""


@dataclass(frozen=True)
class PipelineResult:
    """RAG response plus retrieval and generation accounting."""

    response: RAGResponse
    retrieved: RetrievedEvidence
    prompt_size_bytes: int
    output_size_bytes: int


class GroundedRAGPipeline:
    """Retrieve evidence and enforce provenance on generated fields."""

    def __init__(
        self,
        knowledge_base: KnowledgeBase,
        generator: GroundedGenerator,
        top_k: int = 4,
        min_score: float = 0.05,
    ) -> None:
        self.knowledge_base = knowledge_base
        self.generator = generator
        self.top_k = top_k
        self.min_score = min_score

    def run(self, request: RAGRequest) -> PipelineResult:
        """Run RAG for an already-escalated request."""

        if request.routing_decision != "RAG":
            raise ValueError("RAG pipeline cannot run a LOCAL request")

        pipeline_started = time.perf_counter()
        retrieved = self.knowledge_base.search(
            request,
            top_k=self.top_k,
            min_score=self.min_score,
        )
        prompt_context = build_prompt_context(request, retrieved.evidence)
        prompt_size_bytes = len(prompt_context.encode("utf-8"))

        generation_started = time.perf_counter()
        generated = self.generator.generate(
            request,
            retrieved.evidence,
            prompt_context,
        )
        generation_latency_ms = (time.perf_counter() - generation_started) * 1000
        response = self._build_response(
            request,
            generated,
            retrieved.evidence,
            generation_latency_ms,
            retrieved.latency_ms,
            (time.perf_counter() - pipeline_started) * 1000,
        )
        output_size_bytes = len(
            json.dumps(response.to_dict(), sort_keys=True).encode("utf-8")
        )
        return PipelineResult(
            response=response,
            retrieved=retrieved,
            prompt_size_bytes=prompt_size_bytes,
            output_size_bytes=output_size_bytes,
        )

    @staticmethod
    def _build_response(
        request: RAGRequest,
        generated: Mapping[str, Any],
        evidence: tuple[EvidenceReference, ...],
        generation_latency_ms: float,
        retrieval_latency_ms: float,
        total_latency_ms: float,
    ) -> RAGResponse:
        evidence_by_id = {item.evidence_id: item for item in evidence}
        cited_ids = generated.get("evidence_ids", [])
        if not isinstance(cited_ids, list):
            raise ValueError("generator evidence_ids must be a list")
        if any(evidence_id not in evidence_by_id for evidence_id in cited_ids):
            raise ValueError("generator cited an evidence ID not retrieved")
        selected_evidence = tuple(evidence_by_id[evidence_id] for evidence_id in cited_ids)
        if not selected_evidence:
            raise ValueError("generator must cite at least one retrieved evidence item")

        required_fields = (
            "threat",
            "detection_interpretation",
            "uncertainty_reason",
            "likely_attack_behavior",
            "recommendation",
            "containment_action",
            "caveat",
        )
        missing_fields = [field for field in required_fields if not generated.get(field)]
        if missing_fields:
            raise ValueError(f"generator omitted fields: {missing_fields}")

        return RAGResponse(
            request_id=request.request_id,
            threat=str(generated["threat"]),
            detection_interpretation=str(generated["detection_interpretation"]),
            uncertainty_reason=str(generated["uncertainty_reason"]),
            likely_attack_behavior=str(generated["likely_attack_behavior"]),
            recommendation=str(generated["recommendation"]),
            containment_action=str(generated["containment_action"]),
            caveat=str(generated["caveat"]),
            evidence=selected_evidence,
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=generation_latency_ms,
            total_latency_ms=total_latency_ms,
        )


def build_prompt_context(
    request: RAGRequest,
    evidence: tuple[EvidenceReference, ...],
) -> str:
    """Build a bounded context containing only request data and selected evidence."""

    evidence_payload = [
        {
            "evidence_id": item.evidence_id,
            "source_name": item.source_name,
            "document_title": item.document_title,
            "source_url": item.source_url,
            "excerpt": item.excerpt,
        }
        for item in evidence
    ]
    return json.dumps(
        {
            "instruction": (
                "Use only the supplied evidence for factual claims. "
                "Separate evidence, inference, uncertainty, recommendation, "
                "and hypothetical containment. Cite evidence_id values only."
            ),
            "request": request.to_dict(),
            "evidence": evidence_payload,
        },
        ensure_ascii=True,
        sort_keys=True,
    )


class DeterministicGroundedGenerator:
    """Offline baseline generator for tests and reproducible experiments."""

    def generate(
        self,
        request: RAGRequest,
        evidence: tuple[EvidenceReference, ...],
        prompt_context: str,
    ) -> Mapping[str, Any]:
        del prompt_context
        evidence_ids = [item.evidence_id for item in evidence]
        first_title = evidence[0].document_title if evidence else "selected guidance"
        recommendation = _recommendation_for(evidence_ids[0] if evidence_ids else "")
        return {
            "threat": f"Possible {request.prediction.lower()} activity",
            "detection_interpretation": (
                f"The edge system escalated this request because "
                f"{request.escalation_reason or 'additional analysis was required'}."
            ),
            "uncertainty_reason": (
                f"Confidence was {request.confidence:.3f}; drift_detected="
                f"{request.drift_detected}. The edge result is not a confirmed diagnosis."
            ),
            "likely_attack_behavior": (
                f"The retrieved guidance most closely matches {first_title}; "
                "this is an inference from limited structured context."
            ),
            "recommendation": recommendation,
            "containment_action": (
                "Hypothetical only: an operator may isolate or filter the affected "
                "test segment after confirming impact and preserving evidence."
            ),
            "caveat": (
                "This offline baseline is evidence-grounded but not a substitute "
                "for analyst verification or a validated production LLM."
            ),
            "evidence_ids": evidence_ids,
        }


def _recommendation_for(evidence_id: str) -> str:
    recommendations = {
        "mitre-attack-t1046-network-service-scanning": (
            "Review source and destination port patterns, apply appropriate "
            "network filtering and segmentation, and monitor repeated probes."
        ),
        "mitre-attack-t1110-brute-force": (
            "Apply authentication rate limiting, review targeted accounts, "
            "rotate suspected credentials, and strengthen authentication controls."
        ),
        "owasp-command-injection": (
            "Replace shell construction with safe APIs, validate input, apply "
            "least privilege, and monitor attempted command execution."
        ),
        "owasp-sql-injection": (
            "Use parameterized queries, validate input, restrict database account "
            "privileges, and review database access logs."
        ),
        "cisa-ddos-guidance": (
            "Use traffic monitoring, rate limiting, filtering, resilient service "
            "capacity, and provider coordination where appropriate."
        ),
        "nist-ir-8259a-iot-baseline": (
            "Verify device identity, access control, secure updates, configuration "
            "hardening, and visibility into the device security state."
        ),
        "nist-sp-800-61r2-incident-response": (
            "Preserve relevant evidence, analyze the incident, and select "
            "containment and recovery actions according to operational impact."
        ),
        "cwe-798-hardcoded-credentials": (
            "Remove hard-coded credentials, rotate potentially exposed secrets, "
            "and use protected secrets-management controls."
        ),
    }
    return recommendations.get(
        evidence_id,
        "Review the cited guidance, monitor related activity, and apply "
        "least-privilege controls appropriate to the confirmed investigation.",
    )
