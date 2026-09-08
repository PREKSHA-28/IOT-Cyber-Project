"""Selective, provenance-constrained RAG orchestration."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from typing import Any, Mapping, Protocol

from .rag_contract import (
    EvidenceReference,
    RAGRequest,
    RAGResponse,
)
from .rag_retriever import (
    KnowledgeBase,
    RetrievedEvidence,
)


# ============================================================
# Generator contract
# ============================================================

class GroundedGenerator(Protocol):
    """Adapter implemented by a local or hosted LLM provider."""

    def generate(
        self,
        request: RAGRequest,
        evidence: tuple[EvidenceReference, ...],
        prompt_context: str,
    ) -> Mapping[str, Any]:
        """Return structured fields and evidence IDs only from supplied evidence."""


# ============================================================
# Pipeline result
# ============================================================

@dataclass(frozen=True)
class PipelineResult:
    """RAG response plus retrieval and generation accounting."""

    response: RAGResponse
    retrieved: RetrievedEvidence
    prompt_size_bytes: int
    output_size_bytes: int


# ============================================================
# RAG pipeline
# ============================================================

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

    def run(
        self,
        request: RAGRequest,
    ) -> PipelineResult:
        """Run RAG for an already-escalated request."""

        if request.routing_decision != "RAG":
            raise ValueError(
                "RAG pipeline cannot run a LOCAL request"
            )

        pipeline_started = time.perf_counter()

        # --------------------------------------------------------
        # Retrieval
        # --------------------------------------------------------

        retrieved = self.knowledge_base.search(
            request,
            top_k=self.top_k,
            min_score=self.min_score,
        )

        # --------------------------------------------------------
        # Prompt construction
        # --------------------------------------------------------

        prompt_context = build_prompt_context(
            request,
            retrieved.evidence,
        )

        prompt_size_bytes = len(
            prompt_context.encode("utf-8")
        )

        # --------------------------------------------------------
        # Generation
        # --------------------------------------------------------

        generation_started = time.perf_counter()

        generated = self.generator.generate(
            request,
            retrieved.evidence,
            prompt_context,
        )

        generation_latency_ms = (
            time.perf_counter()
            - generation_started
        ) * 1000

        # --------------------------------------------------------
        # Provenance validation + response construction
        # --------------------------------------------------------

        response = self._build_response(
            request,
            generated,
            retrieved.evidence,
            generation_latency_ms,
            retrieved.latency_ms,
            (
                time.perf_counter()
                - pipeline_started
            ) * 1000,
        )

        output_size_bytes = len(
            json.dumps(
                response.to_dict(),
                sort_keys=True,
            ).encode("utf-8")
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

        evidence_by_id = {
            item.evidence_id: item
            for item in evidence
        }

        cited_ids = generated.get(
            "evidence_ids",
            [],
        )

        if not isinstance(cited_ids, list):
            raise ValueError(
                "generator evidence_ids must be a list"
            )

        if any(
            evidence_id not in evidence_by_id
            for evidence_id in cited_ids
        ):
            raise ValueError(
                "generator cited an evidence ID not retrieved"
            )

        selected_evidence = tuple(
            evidence_by_id[evidence_id]
            for evidence_id in cited_ids
        )

        if not selected_evidence:
            raise ValueError(
                "generator must cite at least one retrieved evidence item"
            )

        required_fields = (
            "threat",
            "detection_interpretation",
            "uncertainty_reason",
            "likely_attack_behavior",
            "recommendation",
            "containment_action",
            "caveat",
        )

        missing_fields = [
            field
            for field in required_fields
            if not generated.get(field)
        ]

        if missing_fields:
            raise ValueError(
                f"generator omitted fields: {missing_fields}"
            )

        return RAGResponse(
            request_id=request.request_id,
            threat=str(
                generated["threat"]
            ),
            detection_interpretation=str(
                generated["detection_interpretation"]
            ),
            uncertainty_reason=str(
                generated["uncertainty_reason"]
            ),
            likely_attack_behavior=str(
                generated["likely_attack_behavior"]
            ),
            recommendation=str(
                generated["recommendation"]
            ),
            containment_action=str(
                generated["containment_action"]
            ),
            caveat=str(
                generated["caveat"]
            ),
            evidence=selected_evidence,
            retrieval_latency_ms=retrieval_latency_ms,
            generation_latency_ms=generation_latency_ms,
            total_latency_ms=total_latency_ms,
        )


# ============================================================
# Prompt construction
# ============================================================

def build_prompt_context(
    request: RAGRequest,
    evidence: tuple[EvidenceReference, ...],
) -> str:
    """
    Build a bounded context containing only request data
    and selected evidence.
    """

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
                "and hypothetical containment. Cite evidence_id values only. "
                "When evidence suggests a possible threat, describe it as a "
                "candidate or evidence-supported threat hypothesis rather "
                "than a confirmed diagnosis."
            ),
            "request": request.to_dict(),
            "evidence": evidence_payload,
        },
        ensure_ascii=True,
        sort_keys=True,
    )


# ============================================================
# Deterministic baseline generator
# ============================================================

class DeterministicGroundedGenerator:
    """
    Offline baseline generator for tests and reproducible experiments.

    Important:
    - It never receives the ground-truth label.
    - Candidate threat interpretation is derived from retrieved evidence.
    - It does not claim that retrieved evidence proves an attack occurred.
    """

    def generate(
        self,
        request: RAGRequest,
        evidence: tuple[EvidenceReference, ...],
        prompt_context: str,
    ) -> Mapping[str, Any]:

        del prompt_context

        evidence_ids = [
            item.evidence_id
            for item in evidence
        ]

        if not evidence:
            raise ValueError(
                "DeterministicGroundedGenerator requires at least one evidence item."
            )

        first_evidence = evidence[0]

        first_title = (
            first_evidence.document_title
        )

        candidate_threat = (
            _candidate_threat_from_evidence(
                evidence
            )
        )

        recommendation = _recommendation_for_evidence(
            evidence
        )

        return {
            "threat": candidate_threat,

            "detection_interpretation": (
                "The edge system escalated this request because "
                f"{request.escalation_reason or 'additional analysis was required'}. "
                "The retrieved cybersecurity evidence provides a candidate "
                "threat interpretation for analyst review."
            ),

            "uncertainty_reason": (
                f"Confidence was {request.confidence:.3f}; "
                f"drift_detected={request.drift_detected}. "
                "The edge result is not a confirmed diagnosis, and the "
                "retrieved evidence should be interpreted as supporting "
                "context rather than ground truth."
            ),

            "likely_attack_behavior": (
                f"The retrieved guidance most closely matches "
                f"{first_title}. This is an evidence-supported inference "
                "from the observed network behavior and selected guidance."
            ),

            "recommendation": recommendation,

            "containment_action": (
                "Hypothetical only: an operator may isolate or filter "
                "the affected test segment after confirming impact, "
                "validating the candidate threat, and preserving evidence."
            ),

            "caveat": (
                "This offline deterministic baseline is evidence-grounded "
                "but is not a substitute for analyst verification or a "
                "validated production LLM."
            ),

            "evidence_ids": evidence_ids,
        }


# ============================================================
# Candidate threat extraction
# ============================================================

def _candidate_threat_from_evidence(
    evidence: tuple[EvidenceReference, ...],
) -> str:
    """
    Derive an evidence-supported candidate threat from the retrieved
    evidence titles.

    The function intentionally uses only retrieved evidence. It does not
    inspect the dataset's ground-truth label.
    """

    if not evidence:
        return "No evidence-supported candidate threat"

    first_title = evidence[0].document_title.strip()

    if not first_title:
        return "No evidence-supported candidate threat"

    # --------------------------------------------------------
    # Explicit MITRE technique identifiers
    # --------------------------------------------------------

    mitre_mappings = {
        "T1046": "Network Service Scanning (T1046)",
        "T1110": "Brute Force (T1110)",
        "T1210": "Exploitation of Remote Services (T1210)",
        "T1498": "Network Denial of Service (T1498)",
        "T1499.002": (
            "Service Exhaustion / HTTP Flooding (T1499.002)"
        ),
        "T1557.001": (
            "Name Resolution Poisoning (T1557.001)"
        ),
    }

    title_upper = first_title.upper()

    for technique_id, threat_name in mitre_mappings.items():

        if technique_id in title_upper:

            return (
                f"Evidence-supported candidate: "
                f"{threat_name}"
            )

    # --------------------------------------------------------
    # Common OWASP/CWE evidence
    # --------------------------------------------------------

    title_lower = first_title.lower()

    if "command injection" in title_lower:
        return (
            "Evidence-supported candidate: "
            "Command Injection"
        )

    if "sql injection" in title_lower:
        return (
            "Evidence-supported candidate: "
            "SQL Injection"
        )

    if "denial of service" in title_lower:
        return (
            "Evidence-supported candidate: "
            "Denial of Service"
        )

    if "hard-coded credential" in title_lower:
        return (
            "Evidence-supported candidate: "
            "Hard-coded Credentials"
        )

    # --------------------------------------------------------
    # Fallback
    # --------------------------------------------------------

    return (
        "Evidence-supported candidate: "
        f"{first_title}"
    )


# ============================================================
# Recommendation selection
# ============================================================

def _recommendation_for_evidence(
    evidence: tuple[EvidenceReference, ...],
) -> str:
    """
    Select an evidence-grounded recommendation.

    Expanded and legacy corpus identifiers are both supported.
    """

    for item in evidence:

        recommendation = _recommendation_for(
            item.evidence_id
        )

        if recommendation is not None:
            return recommendation

    return (
        "Review the cited guidance, preserve relevant evidence, "
        "monitor related activity, and apply least-privilege or "
        "appropriate containment controls after analyst confirmation."
    )


def _recommendation_for(
    evidence_id: str,
) -> str | None:

    evidence_id_lower = evidence_id.lower()

    # --------------------------------------------------------
    # MITRE T1046
    # --------------------------------------------------------

    if (
        "t1046" in evidence_id_lower
        or "network-service-scanning" in evidence_id_lower
    ):
        return (
            "Review source and destination port patterns, apply "
            "appropriate network filtering and segmentation, and "
            "monitor repeated probes."
        )

    # --------------------------------------------------------
    # MITRE T1110
    # --------------------------------------------------------

    if (
        "t1110" in evidence_id_lower
        or "brute-force" in evidence_id_lower
    ):
        return (
            "Apply authentication rate limiting, review targeted "
            "accounts, rotate suspected credentials, and strengthen "
            "authentication controls."
        )

    # --------------------------------------------------------
    # MITRE T1210
    # --------------------------------------------------------

    if "t1210" in evidence_id_lower:
        return (
            "Review exposed remote services, apply appropriate "
            "security patches, restrict unnecessary remote access, "
            "and strengthen network segmentation."
        )

    # --------------------------------------------------------
    # MITRE T1557.001
    # --------------------------------------------------------

    if (
        "t1557-001" in evidence_id_lower
        or "t1557.001" in evidence_id_lower
    ):
        return (
            "Validate DNS responses and resolver behavior, review "
            "unexpected name-resolution changes, strengthen trusted "
            "DNS resolution controls, and monitor for related "
            "manipulation activity."
        )

    # --------------------------------------------------------
    # MITRE T1499.002
    # --------------------------------------------------------

    if "t1499-002" in evidence_id_lower:
        return (
            "Review request-rate and service-exhaustion indicators, "
            "apply rate limiting and filtering, monitor resource "
            "utilization, and protect exposed application services."
        )

    # --------------------------------------------------------
    # MITRE T1498
    # --------------------------------------------------------

    if (
        "t1498" in evidence_id_lower
        or "network-dos" in evidence_id_lower
    ):
        return (
            "Use traffic monitoring, filtering, rate limiting, "
            "resilient service capacity, and appropriate upstream "
            "coordination where necessary."
        )

    # --------------------------------------------------------
    # OWASP Command Injection
    # --------------------------------------------------------

    if (
        "command-injection" in evidence_id_lower
        or "command_injection" in evidence_id_lower
    ):
        return (
            "Replace shell construction with safe APIs, validate "
            "input, apply least privilege, and monitor attempted "
            "command execution."
        )

    # --------------------------------------------------------
    # OWASP SQL Injection
    # --------------------------------------------------------

    if (
        "sql-injection" in evidence_id_lower
        or "sql_injection" in evidence_id_lower
    ):
        return (
            "Use parameterized queries, validate input, restrict "
            "database account privileges, and review database access logs."
        )

    # --------------------------------------------------------
    # CISA DDoS
    # --------------------------------------------------------

    if (
        "cisa-ddos" in evidence_id_lower
        or "ddos-guidance" in evidence_id_lower
    ):
        return (
            "Use traffic monitoring, rate limiting, filtering, "
            "resilient service capacity, and provider coordination "
            "where appropriate."
        )

    # --------------------------------------------------------
    # NIST IoT baseline
    # --------------------------------------------------------

    if (
        "nist-iot-8259a" in evidence_id_lower
        or "iot-baseline" in evidence_id_lower
    ):
        return (
            "Verify device identity, access control, secure updates, "
            "configuration hardening, and visibility into the device "
            "security state."
        )

    # --------------------------------------------------------
    # NIST incident response
    # --------------------------------------------------------

    if (
        "nist-sp-800-61" in evidence_id_lower
        or "incident-response" in evidence_id_lower
    ):
        return (
            "Preserve relevant evidence, analyze the incident, and "
            "select containment and recovery actions according to "
            "operational impact."
        )

    # --------------------------------------------------------
    # CWE hard-coded credentials
    # --------------------------------------------------------

    if (
        "cwe-798" in evidence_id_lower
        or "hardcoded-credentials" in evidence_id_lower
        or "hard-coded-credentials" in evidence_id_lower
    ):
        return (
            "Remove hard-coded credentials, rotate potentially exposed "
            "secrets, and use protected secrets-management controls."
        )

    return None