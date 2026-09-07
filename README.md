# IoT Cybersecurity Project

IEEE-oriented IoT cybersecurity research project combining edge ML detection, selective escalation, provenance-aware retrieval, and grounded mitigation guidance.

## Current Architecture

```text
IoT/network traffic
        |
        v
Edge ML IDS
        |
        v
Prediction + calibrated confidence + drift status
        |
        +-------------------------------+
        |                               |
High confidence and no drift     Low confidence or drift
        |                               |
        v                               v
      LOCAL                             RAG
                                        |
                                        v
                              Evidence retrieval
                                        |
                                        v
                              Grounded response
```

The current edge routing rule is:

- `LOCAL`: confidence >= `0.70` and no detected drift
- `RAG`: confidence < `0.70` or drift detected

The RAG component must be invoked selectively. It must not receive every traffic sample.

## Repository Structure

```text
knowledge_base/
  cybersecurity_sources.json       Source-linked retrieval corpus

examples/
  rag_request.json                  Portable six-field Person 2 request

results/
  drift_detection_results.csv
  edge_routing_results.csv
  routing_evaluation_results.csv
  rag_evaluation_results.csv        Per-case retrieval evaluation
  rag_evaluation_summary.json       Aggregate RAG baseline metrics
  selective_rag_evaluation.json    Selective versus always-on invocation metrics

src/
  edge_decision.py                 Integrated edge prediction and routing
  evaluate_routing.py              Routing evaluation
  detect_drift.py                  ADWIN drift experiment
  rag_contract.py                  RAG request/response data contracts
  rag_retriever.py                 Provenance-aware deterministic retrieval
  rag_pipeline.py                  Selective retrieval and grounded response orchestration
  llm_adapter.py                   Optional OpenAI-compatible LLM adapter
  evaluate_rag.py                  Corpus retrieval and provenance evaluation
  evaluate_selective_rag.py        Selective versus always-on invocation evaluation
  run_rag.py                       Offline CLI for a JSON RAG request
  rag_integration.py               Person 2 request to Person 3 RAG adapter
  ...                              Existing ML and evaluation scripts

tests/
  test_rag.py                      RAG contract, provenance, and routing tests
  test_person2_rag_integration.py  Person 2 to Person 3 integration tests
```

The raw CIC-IoT-2023 data and trained model files are intentionally excluded from Git because of their size. The RAG layer consumes the orchestrator request and does not require those files.

## RAG Interface

### Person 2 Integration Contract

Person 2's request is defined in [src/rag_request.py](src/rag_request.py) and contains exactly these fields:

```json
{
  "prediction": "ATTACK",
  "raw_probability": 0.62,
  "calibrated_probability": 0.58,
  "confidence": 0.58,
  "drift_detected": true,
  "routing_reason": "LOW_CONFIDENCE_AND_DRIFT"
}
```

Person 2's `EdgeOrchestrator.create_rag_request(...)` returns this object only for RAG decisions. Person 3's [src/rag_integration.py](src/rag_integration.py) accepts that object unchanged and returns the existing `RAGResponse` through the retrieval, LLM, provenance, and mitigation pipeline. The adapter does not modify Person 2's request class or orchestrator logic.

For a complete orchestration handoff, use `RAGIntegration.run_orchestration_result(orchestrator, result)`. It returns a `PipelineResult` for RAG decisions and `None` for LOCAL decisions.

Run the integration test with:

```powershell
python -m unittest tests.test_person2_rag_integration -v
```

Because the finalized request intentionally contains no traffic-feature context, the adapter uses a generic incident-analysis retrieval hint. It does not infer a specific attack from the routing reason, and it still requires retrieved source evidence before producing a response.

The Person 2 request boundary is defined in `src/rag_request.py`. The adapter's internal projection is private to Person 3 and is not an orchestrator request.

Person 2 request fields:

```json
{
  "prediction": "ATTACK",
  "raw_probability": 0.62,
  "calibrated_probability": 0.58,
  "confidence": 0.58,
  "drift_detected": true,
  "routing_reason": "LOW_CONFIDENCE_AND_DRIFT"
}
```

Do not add fields to this request. Person 2's orchestrator decides whether to create it; Person 3's adapter consumes it and returns a separate `RAG_RESPONSE`.

## Knowledge Corpus

The initial corpus is [knowledge_base/cybersecurity_sources.json](knowledge_base/cybersecurity_sources.json). Each item records:

- source name
- document title
- source URL or reference
- document type
- publication date when available
- stable document identifier
- source-linked content summary

Initial sources include NIST, MITRE ATT&CK, MITRE CWE, OWASP, and CISA guidance. Corpus entries are concise source-linked summaries, not untraceable web text.

## Retrieval

`src/rag_retriever.py` currently provides deterministic lexical retrieval with no external package dependency. It returns:

- ranked evidence excerpts
- stable evidence identifiers
- source metadata and URLs
- relevance scores
- candidate count
- retrieval latency

Example from the repository root:

```powershell
@'
import sys
sys.path.insert(0, 'src')
from rag_request import create_rag_request
from rag_integration import RAGIntegration
from rag_pipeline import DeterministicGroundedGenerator
from rag_retriever import KnowledgeBase

request = create_rag_request(
  prediction='ATTACK',
  raw_probability=0.62,
  calibrated_probability=0.58,
  confidence=0.58,
  drift_detected=True,
  routing_reason='LOW_CONFIDENCE_AND_DRIFT',
)

result = RAGIntegration.create(
  DeterministicGroundedGenerator(),
  KnowledgeBase.from_json(),
).run(request)
for evidence in result.retrieved.evidence:
    print(evidence.evidence_id, evidence.source_url)
print(f'Retrieval latency: {result.response.retrieval_latency_ms:.3f} ms')
'@ | python -
```

This retrieval implementation is an experiment baseline. It is not presented as a substitute for a validated embedding or hybrid retrieval system.

## Grounded Pipeline

`src/rag_pipeline.py` connects the internal projection, retriever, and a replaceable generator adapter. `src/rag_integration.py` is the public Person 2-to-Person 3 boundary. The pipeline passes selected evidence to the generator and rejects any generated citation that does not match a retrieved evidence identifier.

The repository includes `DeterministicGroundedGenerator` as an offline baseline. It is useful for reproducible tests and does not claim to be an LLM. A hosted or local LLM adapter should implement the `GroundedGenerator` protocol and return the required structured fields plus `evidence_ids`.

The offline baseline selects a mitigation template from the highest-ranked cited source, such as probe monitoring and segmentation for network scanning, authentication rate limiting for brute force, parameterized queries for SQL injection, and safe APIs for command injection. These are recommendations only; no containment action is executed.

Example:

```powershell
@'
import sys
sys.path.insert(0, 'src')
from rag_request import create_rag_request
from rag_integration import RAGIntegration
from rag_pipeline import DeterministicGroundedGenerator

request = create_rag_request(
  prediction='ATTACK',
  raw_probability=0.62,
  calibrated_probability=0.58,
  confidence=0.58,
  drift_detected=True,
  routing_reason='LOW_CONFIDENCE_AND_DRIFT',
)
result = RAGIntegration.create(
  DeterministicGroundedGenerator(),
).run(request)
print(result.response.to_dict())
print('Prompt bytes:', result.prompt_size_bytes)
print('Output bytes:', result.output_size_bytes)
'@ | python -
```

Each pipeline response records `retrieval_latency_ms`, `generation_latency_ms`, and `total_latency_ms`. The `PipelineResult` additionally records prompt and output byte counts for communication-overhead experiments.

For a portable handoff between IDEs or between Person 2 and Person 3, use the request fixture in [examples/rag_request.json](examples/rag_request.json) and run:

```powershell
python src/run_rag.py examples/rag_request.json
```

The fixture demonstrates the exact six-field request that Person 2's orchestrator creates. It intentionally contains no additional context fields.

### Optional LLM Provider

`src/llm_adapter.py` provides `OpenAICompatibleGenerator` for local or hosted endpoints that implement the `/chat/completions` contract. It uses only the Python standard library. Configuration is supplied through environment variables:

```powershell
$env:RAG_LLM_ENDPOINT = 'http://localhost:11434/v1/chat/completions'
$env:RAG_LLM_MODEL = 'llama3.1'
$env:RAG_LLM_API_KEY = 'provider-key-if-required'
```

The API key is optional for local endpoints and is never printed by the adapter. The default deterministic generator remains the recommended validation path. A provider call should be made only after the edge layer has produced a `RAG` request.

## Validation

Run the automated RAG boundary tests with:

```powershell
python -m unittest discover -s tests -v
```

These tests verify provenance preservation, structured responses, LOCAL/RAG routing behavior, edge CSV conversion, and rejection of invented evidence identifiers.

Run the current focused checks from the repository root:

```powershell
python src/run_rag.py examples/rag_request.json
```

The existing edge experiments require the omitted data and model artifacts. They can be run only after those local prerequisites are restored:

```powershell
python src/edge_decision.py
python src/evaluate_routing.py
python src/detect_drift.py
```

## RAG Evaluation

Run the corpus-level baseline evaluation with:

```powershell
python src/evaluate_rag.py
```

The script evaluates six hand-authored threat queries and writes per-case results to `results/rag_evaluation_results.csv` and aggregate metrics to `results/rag_evaluation_summary.json`. It measures top-k hit rate, reciprocal rank, evidence coverage, citation validity, retrieval latency, prompt size, output size, and RAG invocation count.

These metrics evaluate retrieval and provenance behavior against the versioned corpus. They are not CIC-IoT-2023 detection accuracy, emerging-attack detection accuracy, or evidence that an LLM is always correct.

## Selective Invocation Evaluation

Run the routing-efficiency comparison with:

```powershell
python src/evaluate_selective_rag.py
```

The evaluator reads the structured edge output and compares selective invocation with an always-on baseline. On the current 20,000-row edge result, 3,941 rows route to RAG and 16,059 remain LOCAL, avoiding 80.295% of hypothetical LLM invocations. Because the finalized request has no traffic-feature context, the adapter uses generic incident-analysis retrieval context and does not claim a specific attack type.

## Research Integrity Constraints

The RAG response must distinguish retrieved evidence from model inference and uncertainty. It must not invent citations. Mitigation output must separate:

- `RECOMMENDATION`: contextual defensive guidance
- `CONTAINMENT ACTION`: hypothetical or operator-approved action only

The system must not automatically perform destructive actions on real systems. RAG does not guarantee truth, eliminate hallucinations, or ensure detection of every emerging attack. Results remain dependent on corpus coverage, retrieval quality, edge-model behavior, LLM behavior, and the controlled experimental environment.

## Next Development Steps

1. Add a real local or hosted LLM adapter that accepts selected evidence only.
2. Add retrieval relevance, evidence coverage, citation correctness, groundedness, usefulness, and hallucination checks.
3. Record retrieval latency, generation latency, prompt/output size, retrieved-document count, and selective invocation count.
4. Add an integration adapter for Person 2 without modifying the existing edge ML implementation.

## Working Conventions

- Do not add the raw CIC-IoT-2023 dataset or trained model files to Git.
- Keep RAG interfaces independent of Person 2's internal orchestration code.
- Preserve source metadata whenever corpus content is edited.
- Prefer standard-library components until a measured need for external infrastructure exists.
- Update this README whenever the RAG contract, corpus, retrieval behavior, evaluation process, or integration status changes.
