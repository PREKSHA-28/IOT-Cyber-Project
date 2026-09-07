"""Reproducible retrieval and provenance evaluation for the RAG baseline."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from .rag_contract import RAGRequest
from .rag_pipeline import (
    DeterministicGroundedGenerator,
    GroundedRAGPipeline,
)
from .rag_retriever import KnowledgeBase


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

RESULTS_FILE = (
    PROJECT_ROOT
    / "results"
    / "rag_evaluation_results.csv"
)

SUMMARY_FILE = (
    PROJECT_ROOT
    / "results"
    / "rag_evaluation_summary.json"
)


# ============================================================
# Evaluation case definition
# ============================================================

@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    description: str
    context: str
    expected_evidence_id: str


CASES = (
    EvaluationCase(
        "scanning",
        "Network service scanning",
        "network service scanning repeated port probes",
        "mitre-attack-t1046-network-service-scanning",
    ),
    EvaluationCase(
        "brute-force",
        "Repeated authentication attempts",
        "brute force repeated login failures against an IoT service",
        "mitre-attack-t1110-brute-force",
    ),
    EvaluationCase(
        "command-injection",
        "Unsafe operating-system command input",
        "command injection unsafe input passed to an operating system command",
        "owasp-command-injection",
    ),
    EvaluationCase(
        "sql-injection",
        "Unsafe database query input",
        "SQL injection untrusted input incorporated into a database query",
        "owasp-sql-injection",
    ),
    EvaluationCase(
        "ddos",
        "Resource exhaustion",
        "DDoS denial of service traffic exhausting service bandwidth and resources",
        "cisa-ddos-guidance",
    ),
    EvaluationCase(
        "iot-baseline",
        "IoT device hardening",
        "IoT device identification configuration updates access control and visibility",
        "nist-ir-8259a-iot-baseline",
    ),
)


# ============================================================
# Evaluate one case
# ============================================================

def evaluate_case(
    pipeline: GroundedRAGPipeline,
    case: EvaluationCase,
    top_k: int,
) -> dict[str, Any]:

    request = RAGRequest(
        prediction="ATTACK",
        confidence=0.41,
        drift_detected=True,
        attack_probability=0.79,
        context={
            "behavior": case.context
        },
        escalation_reason="LOW_CONFIDENCE_AND_DRIFT",
        request_id=case.case_id,
    )

    result = pipeline.run(request)

    ranked_ids = [
        item.evidence_id
        for item in result.retrieved.evidence
    ]

    expected_rank = (
        ranked_ids.index(case.expected_evidence_id) + 1
        if case.expected_evidence_id in ranked_ids
        else None
    )

    cited_ids = [
        item.evidence_id
        for item in result.response.evidence
    ]

    return {
        "case_id": case.case_id,
        "description": case.description,
        "expected_evidence_id": case.expected_evidence_id,
        "retrieved_evidence_ids": "|".join(ranked_ids),
        "cited_evidence_ids": "|".join(cited_ids),

        "hit_at_k": int(
            case.expected_evidence_id
            in ranked_ids[:top_k]
        ),

        "reciprocal_rank": (
            1.0 / expected_rank
            if expected_rank
            else 0.0
        ),

        "evidence_coverage": int(
            case.expected_evidence_id
            in cited_ids
        ),

        "citation_validity": int(
            set(cited_ids).issubset(
                set(ranked_ids)
            )
        ),

        "retrieval_latency_ms": (
            result.retrieved.latency_ms
        ),

        "generation_latency_ms": (
            result.response.generation_latency_ms
        ),

        "total_latency_ms": (
            result.response.total_latency_ms
        ),

        "prompt_size_bytes": (
            result.prompt_size_bytes
        ),

        "output_size_bytes": (
            result.output_size_bytes
        ),

        "rag_invoked": 1,
    }


# ============================================================
# Write evaluation results
# ============================================================

def write_results(
    rows: list[dict[str, Any]]
) -> None:

    if not rows:
        raise ValueError(
            "No evaluation results were generated."
        )

    RESULTS_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with RESULTS_FILE.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:

        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
        )

        writer.writeheader()
        writer.writerows(rows)


    # --------------------------------------------------------
    # Aggregate summary
    # --------------------------------------------------------

    summary = {
        "cases": len(rows),

        "rag_invocation_count": sum(
            row["rag_invoked"]
            for row in rows
        ),

        "hit_at_k": mean(
            row["hit_at_k"]
            for row in rows
        ),

        "mean_reciprocal_rank": mean(
            row["reciprocal_rank"]
            for row in rows
        ),

        "evidence_coverage": mean(
            row["evidence_coverage"]
            for row in rows
        ),

        "citation_validity": mean(
            row["citation_validity"]
            for row in rows
        ),

        "mean_retrieval_latency_ms": mean(
            row["retrieval_latency_ms"]
            for row in rows
        ),

        "mean_generation_latency_ms": mean(
            row["generation_latency_ms"]
            for row in rows
        ),

        "mean_total_latency_ms": mean(
            row["total_latency_ms"]
            for row in rows
        ),

        "mean_prompt_size_bytes": mean(
            row["prompt_size_bytes"]
            for row in rows
        ),

        "mean_output_size_bytes": mean(
            row["output_size_bytes"]
            for row in rows
        ),

        "evaluation_scope": (
            "Hand-authored threat queries against the "
            "versioned corpus; not a CIC-IoT-2023 "
            "test-set result."
        ),
    }


    SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )


    print("\n" + "=" * 80)
    print("RAG EVALUATION SUMMARY")
    print("=" * 80)

    print(
        json.dumps(
            summary,
            indent=2,
        )
    )

    print("=" * 80)
    print(
        f"Detailed results: {RESULTS_FILE}"
    )


# ============================================================
# Main
# ============================================================

def main() -> None:

    top_k = 4

    pipeline = GroundedRAGPipeline(
        KnowledgeBase.from_json(),
        DeterministicGroundedGenerator(),
        top_k=top_k,
        min_score=0.05,
    )

    rows = [
        evaluate_case(
            pipeline,
            case,
            top_k,
        )
        for case in CASES
    ]

    write_results(rows)


if __name__ == "__main__":
    main()