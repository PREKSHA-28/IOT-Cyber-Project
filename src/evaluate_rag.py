"""Research-grade retrieval evaluation for the cybersecurity RAG corpus.

Evaluates both:
1. lexical token-overlap retrieval
2. semantic sentence-embedding retrieval

Relevant evidence is defined as a set of records belonging to the
appropriate cybersecurity topic/technique, rather than requiring one
specific chunk to be the only correct answer.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from math import log2
from statistics import mean
from typing import Any

from .rag_contract import RAGRequest
from .rag_retriever import KnowledgeBase


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CORPUS_FILE = (
    PROJECT_ROOT
    / "knowledge_base"
    / "cybersecurity_sources_expanded.json"
)

RESULTS_FILE = (
    PROJECT_ROOT
    / "results"
    / "rag_retrieval_comparison.csv"
)

SUMMARY_FILE = (
    PROJECT_ROOT
    / "results"
    / "rag_retrieval_comparison_summary.json"
)


# ============================================================
# Evaluation case
# ============================================================

@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    description: str
    context: str
    relevant_evidence_ids: frozenset[str]


# ============================================================
# Relevance groups
#
# A query can have several valid evidence items:
# overview, indicators, mitigation, investigation, etc.
# ============================================================

CASES = (
    EvaluationCase(
        "vulnerability-scanning",
        "Vulnerability and network service scanning",
        (
            "port scanning, service enumeration, host discovery, "
            "remote service probing, vulnerability scanning"
        ),
        frozenset(
            {
                "mitre-t1046-overview",
                "mitre-t1046-indicators",
                "mitre-t1046-mitigation",
                "mitre-t1046-interpretation",
            }
        ),
    ),

    EvaluationCase(
        "dns-spoofing",
        "DNS spoofing and name resolution poisoning",
        (
            "DNS spoofing, name resolution poisoning, cache poisoning, "
            "unexpected DNS responses, resolver anomalies"
        ),
        frozenset(
            {
                "mitre-t1557-001-overview",
                "mitre-t1557-001-indicators",
                "mitre-t1557-001-mitigation",
                "mitre-t1557-001-caveat",
            }
        ),
    ),

    EvaluationCase(
        "http-flood",
        "HTTP application-layer flooding",
        (
            "HTTP flood, application-layer denial of service, "
            "high request rate, service exhaustion, resource exhaustion"
        ),
        frozenset(
            {
                "mitre-t1499-002-overview",
                "mitre-t1499-002-indicators",
                "mitre-t1499-002-mitigation",
                "mitre-t1499-002-http",
                "owasp-dos-overview",
                "owasp-dos-controls",
            }
        ),
    ),

    EvaluationCase(
        "brute-force",
        "Repeated authentication attempts",
        (
            "brute force, repeated login failures, authentication "
            "attempts, account targeting"
        ),
        frozenset(
            {
                "mitre-t1110-overview",
                "mitre-t1110-indicators",
                "mitre-t1110-mitigation",
                "cwe-798-overview",
            }
        ),
    ),

    EvaluationCase(
        "command-injection",
        "Unsafe operating-system command input",
        (
            "command injection, unsafe input, operating system "
            "command execution, shell injection"
        ),
        frozenset(
            {
                "owasp-command-injection-overview",
                "owasp-command-injection-defense",
                "owasp-command-injection-context",
            }
        ),
    ),

    EvaluationCase(
        "sql-injection",
        "Unsafe database query input",
        (
            "SQL injection, untrusted database input, malicious "
            "query construction, database query manipulation"
        ),
        frozenset(
            {
                "owasp-sql-injection-overview",
                "owasp-sql-injection-defense",
                "owasp-sql-injection-investigation",
            }
        ),
    ),

    EvaluationCase(
        "network-dos",
        "Network denial of service",
        (
            "network denial of service, traffic flooding, "
            "bandwidth exhaustion, resource exhaustion"
        ),
        frozenset(
            {
                "mitre-t1498-overview",
                "mitre-t1498-detection",
                "mitre-t1498-mitigation",
                "cisa-ddos-overview",
                "owasp-dos-overview",
                "owasp-dos-controls",
            }
        ),
    ),

    EvaluationCase(
        "iot-security",
        "IoT device cybersecurity",
        (
            "IoT device identification, access control, secure "
            "updates, configuration, device security state"
        ),
        frozenset(
            {
                "nist-iot-8259a-capabilities",
                "nist-ir-8259r1-foundational",
                "nist-iot-state-awareness",
            }
        ),
    ),
)


# ============================================================
# Build an evaluation request
# ============================================================

def build_request(
    case: EvaluationCase,
) -> RAGRequest:

    return RAGRequest(
        prediction="ATTACK",
        confidence=0.41,
        drift_detected=True,
        attack_probability=0.79,
        context={
            "behavior": case.context,
        },
        escalation_reason="LOW_CONFIDENCE_AND_DRIFT",
        request_id=case.case_id,
    )


# ============================================================
# Ranking metrics
# ============================================================

def precision_at_k(ranked_ids: list[str], relevant: frozenset[str], k: int) -> float:
    if k <= 0:
        return 0.0
    retrieved = ranked_ids[:k]
    if not retrieved:
        return 0.0
    return sum(evidence_id in relevant for evidence_id in retrieved) / len(retrieved)


def recall_at_k(ranked_ids: list[str], relevant: frozenset[str], k: int) -> float:
    if not relevant:
        return 0.0
    retrieved_relevant = sum(
        evidence_id in relevant
        for evidence_id in ranked_ids[:k]
    )
    return retrieved_relevant / len(relevant)


def ndcg_at_k(ranked_ids: list[str], relevant: frozenset[str], k: int) -> float:
    """Binary-relevance NDCG@k for the predefined relevance group."""
    retrieved = ranked_ids[:k]
    dcg = sum(
        (1.0 if evidence_id in relevant else 0.0) / log2(rank + 1)
        for rank, evidence_id in enumerate(retrieved, start=1)
    )
    ideal_relevant = min(len(relevant), k)
    idcg = sum(
        1.0 / log2(rank + 1)
        for rank in range(1, ideal_relevant + 1)
    )
    return dcg / idcg if idcg > 0.0 else 0.0


# ============================================================
# Evaluate one retrieval mode
# ============================================================

def evaluate_case(
    knowledge_base: KnowledgeBase,
    case: EvaluationCase,
    mode: str,
    top_k: int,
) -> dict[str, Any]:

    request = build_request(case)

    retrieved = knowledge_base.search(
        request,
        top_k=top_k,
        min_score=0.0,
        mode=mode,
    )

    ranked_ids = [
        item.evidence_id
        for item in retrieved.evidence
    ]

    relevant = case.relevant_evidence_ids

    relevant_ranks = [
        rank
        for rank, evidence_id in enumerate(
            ranked_ids,
            start=1,
        )
        if evidence_id in relevant
    ]

    first_relevant_rank = (
        min(relevant_ranks)
        if relevant_ranks
        else None
    )

    return {
        "case_id": case.case_id,
        "description": case.description,
        "mode": mode,
        "top_k": top_k,
        "relevant_evidence_count": len(
            relevant
        ),
        "retrieved_evidence_ids": "|".join(
            ranked_ids
        ),

        "hit_at_1": int(
            any(
                evidence_id in relevant
                for evidence_id in ranked_ids[:1]
            )
        ),

        "hit_at_3": int(
            any(
                evidence_id in relevant
                for evidence_id in ranked_ids[:3]
            )
        ),

        "hit_at_4": int(
            any(
                evidence_id in relevant
                for evidence_id in ranked_ids[:4]
            )
        ),

        "precision_at_1": precision_at_k(ranked_ids, relevant, 1),
        "precision_at_3": precision_at_k(ranked_ids, relevant, 3),
        "precision_at_4": precision_at_k(ranked_ids, relevant, 4),
        "recall_at_4": recall_at_k(ranked_ids, relevant, 4),
        "ndcg_at_4": ndcg_at_k(ranked_ids, relevant, 4),

        "first_relevant_rank": (
            first_relevant_rank
            if first_relevant_rank is not None
            else 0
        ),

        "reciprocal_rank": (
            1.0 / first_relevant_rank
            if first_relevant_rank
            else 0.0
        ),

        "relevant_items_retrieved": sum(
            evidence_id in relevant
            for evidence_id in ranked_ids
        ),

        "retrieved_items": len(
            ranked_ids
        ),

        "retrieval_latency_ms": (
            retrieved.latency_ms
        ),

        "candidates_considered": (
            retrieved.candidates_considered
        ),

        "rag_invoked": 1,
    }


# ============================================================
# Evaluate complete corpus
# ============================================================

def evaluate_corpus(
    knowledge_base: KnowledgeBase,
) -> list[dict[str, Any]]:

    rows: list[dict[str, Any]] = []

    for mode in (
        "lexical",
        "semantic",
    ):

        for case in CASES:

            rows.append(
                evaluate_case(
                    knowledge_base,
                    case,
                    mode,
                    top_k=4,
                )
            )

    return rows


# ============================================================
# Write results
# ============================================================

def write_results(
    rows: list[dict[str, Any]],
    knowledge_base: KnowledgeBase,
) -> None:

    if not rows:
        raise ValueError(
            "No retrieval evaluation results were generated."
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
    # Mode-level summaries
    # --------------------------------------------------------

    mode_summaries: dict[str, dict[str, Any]] = {}

    for mode in (
        "lexical",
        "semantic",
    ):

        mode_rows = [
            row
            for row in rows
            if row["mode"] == mode
        ]

        mode_summaries[mode] = {
            "cases": len(mode_rows),

            "hit_at_1": mean(
                row["hit_at_1"]
                for row in mode_rows
            ),

            "hit_at_3": mean(
                row["hit_at_3"]
                for row in mode_rows
            ),

            "hit_at_4": mean(
                row["hit_at_4"]
                for row in mode_rows
            ),

            "mean_reciprocal_rank": mean(
                row["reciprocal_rank"]
                for row in mode_rows
            ),

            "mean_precision_at_1": mean(
                row["precision_at_1"]
                for row in mode_rows
            ),

            "mean_precision_at_3": mean(
                row["precision_at_3"]
                for row in mode_rows
            ),

            "mean_precision_at_4": mean(
                row["precision_at_4"]
                for row in mode_rows
            ),

            "mean_recall_at_4": mean(
                row["recall_at_4"]
                for row in mode_rows
            ),

            "mean_ndcg_at_4": mean(
                row["ndcg_at_4"]
                for row in mode_rows
            ),

            "mean_retrieval_latency_ms": mean(
                row["retrieval_latency_ms"]
                for row in mode_rows
            ),

            "cold_start_latency_ms": mode_rows[0]["retrieval_latency_ms"],

            "warm_mean_retrieval_latency_ms": mean(
                row["retrieval_latency_ms"]
                for row in mode_rows[1:]
            ) if len(mode_rows) > 1 else mode_rows[0]["retrieval_latency_ms"],

            "mean_relevant_items_retrieved": mean(
                row["relevant_items_retrieved"]
                for row in mode_rows
            ),
        }


    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    summary = {
        "corpus_file": str(
            CORPUS_FILE
        ),

        "corpus_size": knowledge_base.size,

        "evaluation_cases": len(
            CASES
        ),

        "retrieval_top_k": 4,

        "mode_comparison": mode_summaries,

        "evaluation_scope": (
            "Eight hand-authored threat-behavior queries "
            "evaluated against relevance groups in the "
            "expanded versioned cybersecurity corpus. "
            "A retrieved item is considered relevant when "
            "it belongs to the predefined evidence group for "
            "that threat/topic. This is a retrieval benchmark, "
            "not a CIC-IoT-2023 attack-classification result."
        ),
    }


    SUMMARY_FILE.write_text(
        json.dumps(
            summary,
            indent=2,
        ),
        encoding="utf-8",
    )


    # --------------------------------------------------------
    # Print
    # --------------------------------------------------------

    print("\n" + "=" * 90)
    print("RAG RETRIEVAL METHOD COMPARISON")
    print("=" * 90)

    print(
        f"Corpus size : {knowledge_base.size}"
    )

    print(
        f"Cases       : {len(CASES)}"
    )

    print(
        "Top-k       : 4"
    )

    print()

    for mode in (
        "lexical",
        "semantic",
    ):

        values = mode_summaries[mode]

        print(
            f"{mode.upper()}"
        )

        print(
            f"  Hit@1                 : "
            f"{values['hit_at_1']:.4f}"
        )

        print(
            f"  Hit@3                 : "
            f"{values['hit_at_3']:.4f}"
        )

        print(
            f"  Hit@4                 : "
            f"{values['hit_at_4']:.4f}"
        )

        print(
            f"  MRR                   : "
            f"{values['mean_reciprocal_rank']:.4f}"
        )

        print(
            f"  Precision@1           : "
            f"{values['mean_precision_at_1']:.4f}"
        )

        print(
            f"  Precision@3           : "
            f"{values['mean_precision_at_3']:.4f}"
        )

        print(
            f"  Precision@4           : "
            f"{values['mean_precision_at_4']:.4f}"
        )

        print(
            f"  Recall@4              : "
            f"{values['mean_recall_at_4']:.4f}"
        )

        print(
            f"  NDCG@4                : "
            f"{values['mean_ndcg_at_4']:.4f}"
        )

        print(
            f"  Mean retrieval latency: "
            f"{values['mean_retrieval_latency_ms']:.6f} ms"
        )

        print(
            f"  Cold-start latency    : "
            f"{values['cold_start_latency_ms']:.6f} ms"
        )

        print(
            f"  Warm mean latency     : "
            f"{values['warm_mean_retrieval_latency_ms']:.6f} ms"
        )

        print(
            f"  Mean relevant items   : "
            f"{values['mean_relevant_items_retrieved']:.2f}"
        )

        print()


    print("=" * 90)

    print(
        f"Detailed results: {RESULTS_FILE}"
    )

    print(
        f"Summary: {SUMMARY_FILE}"
    )

    print("=" * 90)


# ============================================================
# Main
# ============================================================

def main() -> None:

    if not CORPUS_FILE.exists():
        raise FileNotFoundError(
            "Expanded corpus not found:\n"
            f"{CORPUS_FILE}"
        )

    knowledge_base = KnowledgeBase.from_json(
        CORPUS_FILE
    )

    rows = evaluate_corpus(
        knowledge_base
    )

    write_results(
        rows,
        knowledge_base,
    )


if __name__ == "__main__":
    main()