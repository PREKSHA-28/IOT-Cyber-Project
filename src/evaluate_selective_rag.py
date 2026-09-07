"""Measure selective RAG invocation against an always-on baseline."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from statistics import mean
from typing import Any

from orchestrator import EdgeOrchestrator
from rag_integration import RAGIntegration
from rag_pipeline import DeterministicGroundedGenerator


PROJECT_ROOT = Path(__file__).resolve().parent.parent
EDGE_RESULTS_FILE = PROJECT_ROOT / "results" / "edge_routing_results.csv"
OUTPUT_FILE = PROJECT_ROOT / "results" / "selective_rag_evaluation.json"
BENCHMARK_SAMPLE_SIZE = 100


def _as_bool(value: str) -> bool:
    return value.strip().lower() == "true"


def load_edge_results() -> list[dict[str, Any]]:
    if not EDGE_RESULTS_FILE.exists():
        raise FileNotFoundError(
            f"Edge results not found: {EDGE_RESULTS_FILE}. "
            "Run src/edge_decision.py after restoring local model/data files."
        )
    with EDGE_RESULTS_FILE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        row["drift_detected"] = _as_bool(row["drift_detected"])
    return rows


def main() -> None:
    rows = load_edge_results()
    rag_rows = [row for row in rows if row["routing_decision"] == "RAG"]
    local_rows = [row for row in rows if row["routing_decision"] == "LOCAL"]
    total_rows = len(rows)
    if total_rows == 0:
        raise ValueError("edge results are empty")

    benchmark_rows = rag_rows[:BENCHMARK_SAMPLE_SIZE]
    orchestrator = EdgeOrchestrator(confidence_threshold=0.70)
    rag_layer = RAGIntegration.create(DeterministicGroundedGenerator())
    latencies = []
    retrievable_count = 0
    for row in benchmark_rows:
        orchestration_result = orchestrator.decide(
            prediction=row["prediction"],
            raw_probability=float(row["raw_attack_probability"]),
            calibrated_probability=float(row["calibrated_attack_probability"]),
            confidence=float(row["confidence"]),
            drift_detected=row["drift_detected"],
        )
        request = orchestrator.create_rag_request(orchestration_result)
        if request is None:
            continue
        result = rag_layer.run(request)
        latencies.append(result.response.retrieval_latency_ms or 0.0)
        retrievable_count += int(bool(result.response.evidence))

    selective_count = len(rag_rows)
    always_on_count = total_rows
    summary = {
        "total_edge_samples": total_rows,
        "local_count": len(local_rows),
        "selective_rag_invocation_count": selective_count,
        "selective_rag_invocation_rate": selective_count / total_rows,
        "always_on_llm_invocation_count": always_on_count,
        "avoided_llm_invocations": always_on_count - selective_count,
        "invocation_reduction_rate": (always_on_count - selective_count) / always_on_count,
        "benchmark_rag_sample_size": len(benchmark_rows),
        "benchmark_rows_with_retrievable_evidence": retrievable_count,
        "benchmark_evidence_retrieval_rate": (
            retrievable_count / len(benchmark_rows)
            if benchmark_rows
            else None
        ),
        "benchmark_mean_retrieval_latency_ms": mean(latencies) if latencies else None,
        "evaluation_scope": "Structured edge-routing output compared with an always-on invocation baseline.",
        "limitation": "The finalized six-field request has no traffic-feature context; the adapter therefore uses generic incident-analysis retrieval context and does not claim a specific attack type.",
    }
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Results: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
