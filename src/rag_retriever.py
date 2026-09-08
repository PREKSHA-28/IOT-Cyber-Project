"""Provenance-aware retrieval over the project cybersecurity corpus.

The retriever supports two modes:

1. lexical  - transparent token-overlap baseline
2. semantic - sentence-embedding similarity using SentenceTransformers

Keeping both modes allows direct research comparison between the
existing lightweight baseline and embedding-based semantic retrieval.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from sentence_transformers import SentenceTransformer

from .rag_contract import EvidenceReference, RAGRequest


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CORPUS_FILE = (
    PROJECT_ROOT
    / "knowledge_base"
    / "cybersecurity_sources.json"
)


# ============================================================
# Retrieval configuration
# ============================================================

DEFAULT_EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_TOKEN_PATTERN = re.compile(
    r"[a-z0-9][a-z0-9_-]*"
)


# ============================================================
# Retrieved evidence
# ============================================================

@dataclass(frozen=True)
class RetrievedEvidence:
    """Ranked evidence plus retrieval timing for experiment logging."""

    evidence: tuple[EvidenceReference, ...]
    latency_ms: float
    candidates_considered: int


# ============================================================
# Knowledge base
# ============================================================

class KnowledgeBase:
    """Load source-linked cybersecurity knowledge items."""

    def __init__(
        self,
        records: Iterable[Mapping[str, Any]],
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
    ) -> None:

        self._records = tuple(records)

        if not self._records:
            raise ValueError(
                "knowledge corpus must contain at least one record"
            )

        if not embedding_model_name:
            raise ValueError(
                "embedding_model_name must not be empty"
            )

        self._embedding_model_name = embedding_model_name

        # Semantic retrieval uses lazy loading so the embedding model
        # is NOT loaded when only the lexical baseline is used.
        self._embedding_model: SentenceTransformer | None = None

        # Cache searchable text and token sets for lexical retrieval.
        self._searchable_texts = tuple(
            self._build_searchable_text(record)
            for record in self._records
        )

        self._document_token_sets = tuple(
            set(
                _TOKEN_PATTERN.findall(
                    searchable_text.lower()
                )
            )
            for searchable_text in self._searchable_texts
        )

        # Semantic document embeddings are also created lazily.
        self._document_embeddings = None


    # ========================================================
    # Construction
    # ========================================================

    @classmethod
    def from_json(
        cls,
        corpus_file: Path = DEFAULT_CORPUS_FILE,
        embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
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

        return cls(
            records,
            embedding_model_name=embedding_model_name,
        )


    # ========================================================
    # Properties
    # ========================================================

    @property
    def size(self) -> int:
        return len(self._records)

    @property
    def embedding_model_name(self) -> str:
        return self._embedding_model_name


    # ========================================================
    # Main search interface
    # ========================================================

    def search(
        self,
        request: RAGRequest,
        top_k: int = 4,
        min_score: float = 0.05,
        mode: str = "lexical",
    ) -> RetrievedEvidence:
        """
        Return relevant source evidence.

        Parameters
        ----------
        request:
            RAG request containing prediction and context.

        top_k:
            Number of evidence items to return.

        min_score:
            Minimum similarity score.

            Lexical mode:
                token-overlap score in [0, 1]

            Semantic mode:
                cosine similarity after normalized embeddings.
                Values are expected to be in approximately [0, 1]
                for the current corpus/model.

        mode:
            "lexical" or "semantic"
        """

        if top_k < 1:
            raise ValueError(
                "top_k must be at least 1"
            )

        if not 0.0 <= min_score <= 1.0:
            raise ValueError(
                "min_score must be between 0.0 and 1.0"
            )

        normalized_mode = mode.strip().lower()

        if normalized_mode not in {
            "lexical",
            "semantic",
        }:
            raise ValueError(
                "mode must be either 'lexical' or 'semantic'"
            )

        if normalized_mode == "lexical":
            return self._search_lexical(
                request,
                top_k,
                min_score,
            )

        return self._search_semantic(
            request,
            top_k,
            min_score,
        )


    # ========================================================
    # Lexical retrieval
    # ========================================================

    def _search_lexical(
        self,
        request: RAGRequest,
        top_k: int,
        min_score: float,
    ) -> RetrievedEvidence:
        """Transparent token-overlap baseline."""

        started = time.perf_counter()

        query_tokens = self._query_tokens(
            request
        )

        ranked: list[
            tuple[
                float,
                int,
                Mapping[str, Any],
            ]
        ] = []

        for index, record in enumerate(
            self._records
        ):

            document_tokens = (
                self._document_token_sets[index]
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
                        index,
                        record,
                    )
                )

        ranked.sort(
            key=lambda item: (
                -item[0],
                str(
                    item[2].get(
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
            for score, _, record in ranked[:top_k]
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


    # ========================================================
    # Semantic retrieval
    # ========================================================

    def _search_semantic(
        self,
        request: RAGRequest,
        top_k: int,
        min_score: float,
    ) -> RetrievedEvidence:
        """Embedding-based semantic retrieval."""

        started = time.perf_counter()

        model = self._get_embedding_model()

        query_text = self._build_query_text(
            request
        )

        # Encode normalized embeddings so dot product is cosine
        # similarity. This avoids a separate sklearn dependency.
        query_embedding = model.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]

        document_embeddings = (
            self._get_document_embeddings(
                model
            )
        )

        scores = document_embeddings @ query_embedding

        ranked: list[
            tuple[
                float,
                int,
                Mapping[str, Any],
            ]
        ] = []

        for index, raw_score in enumerate(
            scores
        ):

            # Floating-point noise can produce tiny values outside
            # the theoretical cosine range.
            score = float(
                max(
                    0.0,
                    min(
                        1.0,
                        raw_score,
                    ),
                )
            )

            if score >= min_score:
                ranked.append(
                    (
                        score,
                        index,
                        self._records[index],
                    )
                )

        ranked.sort(
            key=lambda item: (
                -item[0],
                str(
                    item[2].get(
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
            for score, _, record in ranked[:top_k]
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


    # ========================================================
    # Embedding model management
    # ========================================================

    def _get_embedding_model(
        self,
    ) -> SentenceTransformer:

        if self._embedding_model is None:

            self._embedding_model = (
                SentenceTransformer(
                    self._embedding_model_name
                )
            )

        return self._embedding_model


    def _get_document_embeddings(
        self,
        model: SentenceTransformer,
    ):
        """
        Generate document embeddings once and cache them.

        Caching is important because the knowledge corpus is static
        during an evaluation run.
        """

        if self._document_embeddings is None:

            self._document_embeddings = (
                model.encode(
                    list(
                        self._searchable_texts
                    ),
                    convert_to_numpy=True,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                )
            )

        return self._document_embeddings


    # ========================================================
    # Text construction
    # ========================================================

    @staticmethod
    def _build_searchable_text(
        record: Mapping[str, Any],
    ) -> str:
        """
        Build searchable text from the knowledge record.

        Additional metadata fields are included when present so
        expanded corpora can provide richer semantic context.
        """

        fields = [
            "document_title",
            "document_type",
            "topic",
            "attack_family",
            "content",
        ]

        parts = []

        for field in fields:

            value = record.get(
                field,
                "",
            )

            if value:
                parts.append(
                    str(value)
                )

        return " ".join(parts)


    @staticmethod
    def _get_behavior_query(
        request: RAGRequest,
    ) -> str:
        """
        Extract the explicit observable-behavior retrieval query.

        The end-to-end pipeline may provide a compact label-free
        network-behavior description through request.context.
        This query is intentionally given priority over generic
        incident-response text because attack-specific behavior
        should drive evidence retrieval.
        """

        context = request.context

        if not isinstance(context, Mapping):
            return ""

        behavior_context = context.get(
            "behavior_context"
        )

        if not isinstance(
            behavior_context,
            Mapping,
        ):
            return ""

        retrieval_query = behavior_context.get(
            "retrieval_query",
            "",
        )

        if retrieval_query is None:
            return ""

        return str(
            retrieval_query
        ).strip()


    @staticmethod
    def _build_query_text(
        request: RAGRequest,
    ) -> str:
        """
        Build the semantic retrieval query.

        The explicit behavior query is placed first so that
        observable network behavior has priority. Generic request
        metadata is retained as supporting context.
        """

        context_text = json.dumps(
            request.context,
            sort_keys=True,
            default=str,
        )

        behavior_query = KnowledgeBase._get_behavior_query(
            request
        )

        query_parts = [
            behavior_query,
            request.prediction,
            request.escalation_reason or "",
            "drift" if request.drift_detected else "",
            context_text,
        ]

        return " ".join(
            part
            for part in query_parts
            if part
        )


    # ========================================================
    # Lexical query tokens
    # ========================================================

    @staticmethod
    def _query_tokens(
        request: RAGRequest,
    ) -> set[str]:
        """
        Build lexical query tokens with explicit behavioral terms
        prioritized in the query construction.

        The full context is retained as supporting information.
        """

        context_text = json.dumps(
            request.context,
            sort_keys=True,
            default=str,
        )

        behavior_query = KnowledgeBase._get_behavior_query(
            request
        )

        query_text = " ".join(
            (
                behavior_query,
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


    # ========================================================
    # Evidence conversion
    # ========================================================

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

            # Prefer an explicit chunk ID when available.
            # Fall back to document_id for older records.
            chunk_id=str(
                record.get(
                    "chunk_id",
                    record["document_id"],
                )
            ),

            publication_date=record.get(
                "publication_date"
            ),
        )