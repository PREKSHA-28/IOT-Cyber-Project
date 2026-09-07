"""Deterministic provenance-aware retrieval over the project knowledge corpus."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .rag_contract import EvidenceReference, RAGRequest


PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CORPUS_FILE = (
    PROJECT_ROOT
    / "knowledge_base"
    / "cybersecurity_sources.json"
)

_TOKEN_PATTERN = re.compile(
    r"[a-z0-9][a-z0-9_-]*"
)


@dataclass(frozen=True)
class RetrievedEvidence:
    """Ranked evidence plus retrieval timing for experiment logging."""

    evidence: tuple[EvidenceReference, ...]
    latency_ms: float
    candidates_considered: int


class KnowledgeBase:
    """Load source-linked knowledge items from a JSON corpus."""

    def __init__(
        self,
        records: Iterable[Mapping[str, Any]],
    ) -> None:

        self._records = tuple(records)

        if not self._records:
            raise ValueError(
                "knowledge corpus must contain at least one record"
            )

    @classmethod
    def from_json(
        cls,
        corpus_file: Path = DEFAULT_CORPUS_FILE,
    ) -> "KnowledgeBase":

        with corpus_file.open(
            "r",
            encoding="utf-8",
        ) as handle:

            records = json.load(handle)

        if not isinstance(records, list):
            raise ValueError(
                "knowledge corpus JSON must contain a list"
            )

        return cls(records)

    @property
    def size(self) -> int:
        return len(self._records)

    def search(
        self,
        request: RAGRequest,
        top_k: int = 4,
        min_score: float = 0.05,
    ) -> RetrievedEvidence:
        """Return relevant source excerpts using transparent token overlap."""

        if top_k < 1:
            raise ValueError(
                "top_k must be at least 1"
            )

        if not 0.0 <= min_score <= 1.0:
            raise ValueError(
                "min_score must be between 0.0 and 1.0"
            )

        started = time.perf_counter()

        query_tokens = self._query_tokens(
            request
        )

        ranked: list[
            tuple[
                float,
                Mapping[str, Any],
            ]
        ] = []

        for record in self._records:

            searchable_text = " ".join(
                str(
                    record.get(
                        field,
                        "",
                    )
                )
                for field in (
                    "document_title",
                    "document_type",
                    "content",
                )
            )

            document_tokens = set(
                _TOKEN_PATTERN.findall(
                    searchable_text.lower()
                )
            )

            overlap = (
                query_tokens
                & document_tokens
            )

            score = (
                len(overlap)
                / len(query_tokens)
                if query_tokens
                else 0.0
            )

            if score >= min_score:
                ranked.append(
                    (
                        score,
                        record,
                    )
                )

        ranked.sort(
            key=lambda item: (
                -item[0],
                str(
                    item[1].get(
                        "document_id",
                        "",
                    )
                ),
            )
        )

        selected = tuple(
            self._to_evidence(
                record,
                score,
            )
            for score, record in ranked[:top_k]
        )

        latency_ms = (
            time.perf_counter()
            - started
        ) * 1000

        return RetrievedEvidence(
            evidence=selected,
            latency_ms=latency_ms,
            candidates_considered=len(
                self._records
            ),
        )

    @staticmethod
    def _query_tokens(
        request: RAGRequest,
    ) -> set[str]:

        context_text = json.dumps(
            request.context,
            sort_keys=True,
            default=str,
        )

        query_text = " ".join(
            (
                request.prediction,
                request.escalation_reason
                or "",
                (
                    "drift"
                    if request.drift_detected
                    else ""
                ),
                context_text,
            )
        )

        return set(
            _TOKEN_PATTERN.findall(
                query_text.lower()
            )
        )

    @staticmethod
    def _to_evidence(
        record: Mapping[str, Any],
        score: float,
    ) -> EvidenceReference:

        return EvidenceReference(
            evidence_id=str(
                record["document_id"]
            ),
            source_name=str(
                record["source_name"]
            ),
            document_title=str(
                record["document_title"]
            ),
            source_url=str(
                record["source_url"]
            ),
            document_type=str(
                record["document_type"]
            ),
            excerpt=str(
                record["content"]
            ),
            relevance_score=score,
            chunk_id=str(
                record["document_id"]
            ),
            publication_date=record.get(
                "publication_date"
            ),
        )