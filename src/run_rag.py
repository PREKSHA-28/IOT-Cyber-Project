"""Run the offline RAG pipeline for a JSON request fixture."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .rag_integration import RAGIntegration
from .rag_pipeline import DeterministicGroundedGenerator
from .rag_request import RAGRequest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__
    )

    parser.add_argument(
        "request_file",
        type=Path,
        help="Path to a JSON RAG request",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    with args.request_file.open(
        "r",
        encoding="utf-8",
    ) as handle:
        payload = json.load(handle)

    request = RAGRequest(
        **payload
    )

    result = (
        RAGIntegration
        .create(
            DeterministicGroundedGenerator()
        )
        .run(request)
    )

    print(
        json.dumps(
            result.response.to_dict(),
            indent=2,
        )
    )

    print(
        json.dumps(
            {
                "prompt_size_bytes": result.prompt_size_bytes,
                "output_size_bytes": result.output_size_bytes,
                "retrieval_latency_ms": (
                    result.response.retrieval_latency_ms
                ),
                "generation_latency_ms": (
                    result.response.generation_latency_ms
                ),
                "total_latency_ms": (
                    result.response.total_latency_ms
                ),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()