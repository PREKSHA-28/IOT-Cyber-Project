"""Optional OpenAI-compatible adapter for grounded RAG generation."""

from __future__ import annotations

import json
import os
from typing import Any, Mapping
from urllib import error, request

from rag_contract import EvidenceReference, RAGRequest


class OpenAICompatibleGenerator:
    """Call a local or hosted OpenAI-compatible chat-completions endpoint.

    The adapter does not decide whether a citation is valid. The RAG pipeline
    validates every returned evidence ID against the retrieved evidence set.
    """

    def __init__(
        self,
        endpoint: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.endpoint = endpoint or os.environ.get(
            "RAG_LLM_ENDPOINT",
            "http://localhost:11434/v1/chat/completions",
        )
        self.api_key = api_key or os.environ.get("RAG_LLM_API_KEY")
        self.model = model or os.environ.get("RAG_LLM_MODEL", "llama3.1")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.timeout_seconds = timeout_seconds

    def generate(
        self,
        request_context: RAGRequest,
        evidence: tuple[EvidenceReference, ...],
        prompt_context: str,
    ) -> Mapping[str, Any]:
        """Generate structured fields from bounded evidence context."""

        del request_context
        system_prompt = (
            "You are a cybersecurity analysis assistant. Return only a JSON object "
            "with exactly these fields: threat, detection_interpretation, "
            "uncertainty_reason, likely_attack_behavior, recommendation, "
            "containment_action, caveat, evidence_ids. The evidence_ids field "
            "must contain only IDs supplied in the evidence context. Do not invent "
            "sources or citations. Clearly label inference and uncertainty. "
            "Containment must be hypothetical and operator-approved."
        )
        payload = {
            "model": self.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt_context},
            ],
        }
        encoded_payload = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
        }
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        http_request = request.Request(
            self.endpoint,
            data=encoded_payload,
            headers=headers,
            method="POST",
        )
        try:
            with request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                response_payload = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            raise RuntimeError(f"LLM endpoint returned HTTP {exc.code}") from exc
        except error.URLError as exc:
            raise RuntimeError(f"LLM endpoint could not be reached: {exc.reason}") from exc

        content = self._extract_content(response_payload)
        try:
            generated = json.loads(content)
        except json.JSONDecodeError as exc:
            raise ValueError("LLM response content was not valid JSON") from exc
        if not isinstance(generated, Mapping):
            raise ValueError("LLM response JSON must be an object")
        return generated

    @staticmethod
    def _extract_content(payload: Mapping[str, Any]) -> str:
        try:
            content = payload["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("LLM response did not contain chat completion content") from exc
        if not isinstance(content, str) or not content.strip():
            raise ValueError("LLM response content was empty")
        return content
