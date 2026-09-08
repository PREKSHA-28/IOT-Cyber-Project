<div align="center">

🛡️ IoT Edge-AI Cybersecurity Framework

Resource-Efficient • Uncertainty-Aware • Continually Adaptive

An end-to-end research prototype for detecting known and emerging IoT threats at the edge — and selectively escalating uncertain cases to provenance-aware cybersecurity RAG.

<br>







</div>

⚡ The idea in one picture

┌─────────────────────── IoT Network Traffic ───────────────────────┐
│                                                                  │
│                     Preprocessing / Features                     │
│                              │                                   │
│                              ▼                                   │
│                    🌲 Edge ML IDS                                │
│                       Random Forest                              │
│                              │                                   │
│                ┌─────────────┴─────────────┐                    │
│                ▼                           ▼                    │
│       Calibrated Probability        Confidence + Drift           │
│                │                           │                    │
│                └─────────────┬─────────────┘                    │
│                              ▼                                   │
│                  🧠 Edge Orchestrator                            │
│                              │                                   │
│              ┌───────────────┴───────────────┐                  │
│              ▼                               ▼                  │
│      ✅ HIGH CONFIDENCE                 ⚠️ LOW CONFIDENCE        │
│         + NO DRIFT                       or DRIFT                │
│              │                               │                  │
│              ▼                               ▼                  │
│        LOCAL DECISION              🔎 PROVENANCE-AWARE RAG       │
│                                              │                  │
│                              ┌───────────────┴──────────────┐   │
│                              ▼                              ▼   │
│                      Cybersecurity KB                Grounded   │
│                  MITRE • NIST • OWASP              Explanation  │
│                  CISA • CWE                                      │
│                                                              │
└──────────────────────────────────────────────────────────────────┘

Core research idea: use lightweight Edge AI first, then invoke additional cybersecurity reasoning only when the model is uncertain or the stream exhibits drift.

The contribution is not “Random Forest + RAG” as a collection of technologies. The contribution is the selective orchestration policy that connects edge intrusion detection, calibrated uncertainty, drift awareness, and evidence-grounded cybersecurity knowledge.

🎯 Problem Statement

Design and develop a resource-efficient, uncertainty-aware, continually adaptive Edge-AI cybersecurity framework for IoT networks that detects known and emerging cyber threats under evolving network conditions.

The framework is designed to:

perform lightweight intrusion and anomaly detection at the edge;

monitor prediction uncertainty and network-stream changes;

detect concept drift;

decide whether a case can remain local or needs additional cybersecurity knowledge;

selectively invoke a provenance-aware RAG pipeline for low-confidence, anomalous, or emerging cases;

generate evidence-grounded threat explanations and context-specific mitigation guidance;

minimize unnecessary RAG/LLM interactions while maintaining useful detection and low response latency.

The evaluation considers detection performance, false positives, adaptation behavior, latency, edge resource usage, communication/workload overhead, RAG invocation rate, retrieval quality, and evidence/recommendation reliability.

🌟 Why this project is interesting

Traditional IDS pipelines are usually built around a single question:

“Is this traffic malicious?”

This project adds a second question:

“How much additional reasoning does this case need?”

That creates a two-level security architecture:

Layer

Role

⚡ Edge ML

Fast first-stage detection

🧭 Orchestrator

Decides whether additional reasoning is necessary

📚 RAG

Retrieves trusted cybersecurity knowledge

🧾 Grounded response

Produces a traceable candidate interpretation and mitigation guidance

The design aims to avoid sending every network event through an expensive reasoning layer.

🏗️ System Architecture

End-to-end pipeline

IoT Traffic
    │
    ▼
Preprocessing
    │
    ▼
Random Forest IDS
    │
    ├──────────────► Raw attack probability
    │
    ▼
Isotonic Calibration
    │
    ▼
Calibrated probability
    │
    ▼
Confidence estimation
    │
    ▼
ADWIN Drift Detection
    │
    ▼
Edge Orchestrator
    │
    ├── High confidence + no drift ─────► LOCAL
    │
    └── Low confidence OR drift ─────────► RAG
                                            │
                                            ▼
                                  Behavior Context
                                            │
                                   ┌────────┴────────┐
                                   ▼                 ▼
                              Lexical           Semantic
                             Retrieval         Retrieval
                                   │                 │
                                   └────────┬────────┘
                                            ▼
                                   Provenance Evidence
                                            │
                                            ▼
                                  Grounded Response
                                            │
                                            ▼
                         Threat + Uncertainty + Behavior
                         Recommendation + Containment
                         + Evidence + Caveat

GitHub-friendly Mermaid architecture

flowchart TD
    A[IoT Network Traffic] --> B[Preprocessing]
    B --> C[Random Forest IDS]
    C --> D[Attack Probability]
    D --> E[Isotonic Calibration]
    E --> F[Confidence]
    F --> G[ADWIN Drift Detection]
    F --> H[Edge Orchestrator]
    G --> H

    H -->|High confidence + no drift| I[LOCAL]
    H -->|Low confidence or drift| J[RAG]

    J --> K[Label-free Behavior Context]
    K --> L[Retrieval]
    L --> M[Lexical / Semantic Search]
    M --> N[Provenance-aware Knowledge Base]
    N --> O[Grounded Response]
    O --> P[Candidate Threat]
    O --> Q[Uncertainty Reason]
    O --> R[Mitigation / Containment]

🧠 Research Contributions

01 — Lightweight Edge-AI IDS

A Random Forest classifier performs the first-stage binary intrusion detection directly on IoT network-flow features.

02 — Calibrated uncertainty

Raw Random Forest attack probabilities are calibrated using Isotonic Regression so that the routing layer can use a more meaningful probability estimate.

03 — Concept-drift awareness

The streaming layer uses ADWIN from River to monitor changes in the calibrated attack-probability stream.

04 — Selective security escalation

The Edge Orchestrator chooses between local processing and RAG escalation using confidence and drift signals.

05 — Provenance-aware cybersecurity RAG

The RAG layer retrieves source-linked evidence from a curated cybersecurity knowledge corpus containing material from:

MITRE ATT&CK • NIST • OWASP • CISA • CWE

06 — Evidence-grounded explanations

The response contains:

candidate threat interpretation;

detection interpretation;

uncertainty reason;

likely behavior;

recommendation;

hypothetical containment guidance;

caveat;

retrieved evidence.

07 — Research-oriented evaluation

The project includes dedicated experiments for:

known-attack detection;

emerging-attack detection;

probability calibration;

concept drift;

threshold trade-offs;

routing-policy comparison;

three-system ablation;

Selective-RAG vs Always-RAG;

emerging false-negative evidence alignment;

retrieval quality;

end-to-end latency;

orchestrator CPU/memory;

provenance tracking.

📊 Dataset

CIC-IoT-2023

The project uses CIC-IoT-2023.

The downloaded dataset is organized into five merged CSV partitions:

Merged01.csv
Merged02.csv
Merged03.csv
Merged04.csv
Merged05.csv

These are partitions of the same dataset, not five independent datasets.

Processed dataset

Property

Value

Rows

~2.21 million

Columns

41

Network-flow features

39

Label columns

Label, Label_Binary

ATTACK

2,123,497

BENIGN

83,841

Feature groups include:

packet/header statistics;

protocol indicators;

TCP flags;

application protocol indicators;

rate/count features;

packet-size statistics;

inter-arrival time;

variance/statistical features.

Emerging-attack protocol

Three attack classes were withheld from model development:

DNS_SPOOFING
VULNERABILITYSCAN
DOS-HTTP_FLOOD

The evaluation therefore distinguishes:

Known attacks    → used for model development/evaluation
Emerging attacks → withheld from model training

This is a controlled known/emerging experiment. The split is not claimed to represent a chronological deployment stream.

🤖 Machine Learning Layer

Stage 1 — Binary IDS

The first Random Forest predicts:

BENIGN
ATTACK

Configuration

n_estimators = 100
max_depth = 20
class_weight = balanced
random_state = 42
n_jobs = -1

Model artifact:

models/binary_ids_random_forest.joblib

Model artifacts are excluded from GitHub because they are large/generated files.

Known-test performance

Metric

Result

Precision

99.99%

Recall

98.90%

F1

99.44%

Emerging baseline

Metric

Result

Precision

100.00%

Recall

34.67%

F1

51.49%

That large recall gap is one of the motivations for selective escalation.

🎯 Attack-Type Classification

A second Random Forest is trained on known attack classes to estimate the attack family/type after malicious traffic has been identified.

Known attack classes: 30
Held-out emerging classes: 3

Artifact:

models/attack_type_random_forest.joblib

Known-test performance

Metric

Result

Accuracy

~78%

Macro F1

0.626

Weighted F1

0.775

The lower macro F1 reflects the difficulty of distinguishing rare and imbalanced attack classes.

📏 Probability Calibration

Random Forest probabilities are not automatically perfect confidence estimates.

Therefore:

Random Forest probability
          ↓
   Isotonic Regression
          ↓
Calibrated attack probability

The calibrator is trained using the validation set only.

Artifact:

models/attack_probability_calibrator.joblib

Brier score improvement

Split

Raw

Calibrated

Validation

0.009203

0.006140

Known test

0.009128

0.006095

Emerging

0.561658

0.259323

Calibration improves the measured probability quality, but it remains imperfect under distribution shift.

🌊 Concept Drift Detection

The streaming layer uses:

ADWIN — Adaptive Windowing

from the River library.

Configuration:

delta = 0.002

ADWIN monitors the calibrated attack-probability stream for distribution changes.

Important research interpretation

The drift experiment intentionally uses a controlled transition from:

KNOWN
  ↓
EMERGING

It is not presented as a real chronological ordering of CIC-IoT-2023 deployment traffic.

The subsequent drift-adaptation experiment found that the detected drift event produced no additional unique RAG escalations beyond low-confidence routing in this stream.

That negative result is retained rather than hidden.

🧭 Edge Orchestrator

The Edge Orchestrator is the decision layer connecting Edge ML and RAG.

Inputs

prediction
raw_probability
calibrated_probability
confidence
drift_detected

Outputs

prediction
raw_probability
calibrated_probability
confidence
drift_detected
routing_decision
routing_reason

Routing policy

Selected default confidence threshold:

0.70

Decision table:

Confidence

Drift

Decision

High

No

✅ LOCAL

Low

No

🔎 RAG

High

Yes

🔎 RAG

Low

Yes

🔎 RAG

The threshold is treated as a resource-aware research/engineering trade-off, not as a globally optimal mathematical threshold.

⚖️ Adaptive Threshold Analysis

The routing threshold was evaluated across multiple operating points.

Threshold

RAG Invocation

FN Escalation

0.60

6.63%

19.50%

0.65

9.86%

29.01%

0.70

19.84%

58.38%

0.75

22.34%

65.58%

0.80

24.31%

71.22%

0.85

26.82%

78.12%

0.90

30.73%

89.16%

The selected 0.70 operating point provides a practical middle ground between lower RAG usage and broader false-negative escalation.

🔎 Provenance-Aware RAG

The RAG subsystem is a separate cybersecurity-knowledge layer.

RAG Request
     │
     ▼
Behavior Query Construction
     │
     ├───────────────┐
     ▼               ▼
Lexical Search   Semantic Search
     │               │
     └───────┬───────┘
             ▼
     Provenance Evidence
             │
             ▼
     Grounded Generator
             │
             ▼
 Threat + Explanation
 + Recommendation
 + Containment
 + Evidence

RAG request contract

The request can contain:

prediction
raw_probability
calibrated_probability
confidence
drift_detected
routing_reason
behavior_context

behavior_context is constructed from observable network-flow information so that the RAG layer receives useful security context without receiving the ground-truth attack label.

📚 Cybersecurity Knowledge Base

Active corpus:

knowledge_base/cybersecurity_sources_expanded.json

Current size:

37 source-linked records

Coverage includes:

Source family

Examples

MITRE ATT&CK

Network Service Scanning, Brute Force, Remote Service Exploitation, DoS, Name Resolution Poisoning

NIST

IoT security capabilities, incident response, analysis, containment

OWASP

Command Injection, SQL Injection, DoS

CISA

DDoS guidance

CWE

Hard-coded Credentials

Each evidence record carries provenance-related metadata such as source identity, document/chunk identifiers, and source URLs.

Why provenance matters

The system is intended to avoid unexplained cybersecurity output.

Retrieved evidence is carried forward so that the generated interpretation and recommendation can be traced back to the source material selected by the retrieval stage.

🔬 Retrieval Methods

Two retrieval strategies are implemented.

1. Lexical retrieval

A transparent token-overlap baseline across fields such as:

document_title
document_type
topic
attack_family
content

Advantages:

lightweight;

easy to inspect;

deterministic;

low retrieval overhead.

2. Semantic retrieval

Uses:

Sentence Transformers
all-MiniLM-L6-v2

Embedding dimension:

384

Document embeddings are cached during a run to avoid repeatedly encoding the static knowledge corpus.

📈 RAG Retrieval Benchmark

The retrieval benchmark contains eight hand-authored cybersecurity cases covering:

vulnerability scanning;

DNS-related behavior;

HTTP flooding;

brute force;

command injection;

SQL injection;

network DoS;

IoT security.

Measured results

Metric

Lexical

Semantic

Hit@1

1.00

1.00

Hit@3

1.00

1.00

Hit@4

1.00

1.00

MRR

1.00

1.00

Precision@1

1.00

1.00

Precision@3

0.917

0.833

Precision@4

0.813

0.719

Recall@4

0.813

0.750

NDCG@4

0.925

0.860

These figures apply only to the eight-case retrieval benchmark. They are not universal RAG-performance claims and are not CIC-IoT-2023 classification results.

🧾 Grounded Response Generation

The current repository uses a deterministic grounded generator for reproducible offline evaluation.

It:

receives the RAG request;

receives retrieved evidence;

validates evidence identifiers;

derives a candidate threat interpretation from retrieved evidence;

produces explanation and recommendation fields based on the selected evidence;

retains provenance;

includes a caveat about the offline baseline.

Important

This component is not a production-grade autonomous cybersecurity LLM.

The architecture exposes a clean interface for a future real LLM adapter.

🔄 End-to-End Evaluation

Implementation:

src/run_end_to_end.py

Controlled stream:

10,000 KNOWN
10,000 EMERGING
────────────────
20,000 TOTAL

Routing observed

Decision

Samples

Percentage

✅ LOCAL

16,059

80.30%

🔎 RAG

3,941

19.71%

By phase

Phase

LOCAL

RAG

Known

98.58%

1.42%

Emerging

62.01%

37.99%

Deterministic offline RAG timing

Component

Mean

Retrieval

0.074 ms

Generation

0.005 ms

Total RAG

0.140 ms

These timings describe the current deterministic local implementation, not production LLM/API latency.

🚀 Selective-RAG vs Always-RAG

A dedicated experiment compares:

Always-RAG

Every sample is passed to the RAG layer.

20,000 / 20,000 = 100%

Selective-RAG

Only samples selected by the edge routing policy invoke RAG.

3,941 / 20,000 = 19.71%

Result

RAG calls avoided     = 16,059
RAG workload reduction = 80.30%

Measured cumulative deterministic RAG latency reduction:

≈ 80.53%

This demonstrates reduced RAG workload under the evaluated deterministic offline implementation. It should not be generalized directly into claims such as “80% lower real-world LLM cost.”

🧪 Three-System Ablation

The ablation isolates the value of the routing mechanism.

System

Description

A — EDGE_ML_ONLY

Edge ML prediction used directly; no RAG

B — ML + CONFIDENCE_RAG

Low-confidence cases escalate

C — PROPOSED

Low-confidence or drift cases escalate

All three systems use the same:

ML predictions;

probabilities;

samples.

Therefore the ablation isolates routing allocation, not model retraining.

Important observation

Because routing does not change the Random Forest prediction:

attack recall is unchanged;

FPR is unchanged;

raw ML false negatives are unchanged.

The meaningful differences are:

RAG invocation;

emerging false-negative escalation;

incremental effect of drift-aware routing.

In the current controlled stream, drift awareness produced no additional unique escalations beyond confidence-based routing.

🧩 Emerging False-Negative → RAG Evidence Evaluation

The final evidence experiment asks:

When the edge ML system misses an emerging attack and routes it to RAG, does the retrieved evidence actually correspond to that attack family?

Expected evidence groups:

VULNERABILITYSCAN → MITRE T1046
DNS_SPOOFING      → MITRE T1557.001
DOS-HTTP_FLOOD    → MITRE T1499.002

Important methodological rule

Ground-truth labels are used only by this evaluation script to judge evidence relevance.

They are never passed into the RAG request, retriever, or generator.

Observed limitation

Across the 3,799 emerging ML false negatives routed to RAG, relevant evidence appeared in the top four results for approximately:

20.40%

This is not RAG attack-classification accuracy.

It exposes an important representation problem:

Aggregate network-flow features do not always provide enough cybersecurity vocabulary to map directly onto threat-intelligence concepts such as scanning, spoofing, or specific attack techniques.

This limitation is retained intentionally.

🧠 A Key Research Insight

The experiments reveal a useful distinction:

Detecting malicious traffic
          ≠
Knowing what cybersecurity knowledge is relevant

The Edge ML detector can identify suspiciousness while remaining uncertain about the underlying cybersecurity interpretation.

The RAG layer can provide trusted external context — but retrieval quality depends on whether the observable traffic representation contains enough semantic information to retrieve the right threat knowledge.

That gap is an important direction for future work.

🧰 Technology Stack

Technology

Purpose

Python

Main implementation language

pandas

Dataset processing and experiment analysis

NumPy

Numerical operations

scikit-learn

Random Forest, calibration, metrics

Random Forest

Binary IDS + attack-type classification

Isotonic Regression

Probability calibration

River / ADWIN

Streaming concept-drift detection

Sentence Transformers

Semantic retrieval

all-MiniLM-L6-v2

Embedding model

PyTorch

Semantic-retrieval backend

Transformers

NLP/model ecosystem dependency

JSON

Provenance-aware cybersecurity knowledge base

psutil

CPU and memory measurement

unittest

Automated tests

Git

Version control

GitHub

Collaboration and source management

Architecture note

The current implementation does not require:

MySQL;

MongoDB;

a separately hosted vector database.

The research prototype uses a source-linked JSON knowledge base and performs lexical/semantic retrieval in-process for reproducibility and lightweight evaluation.

📁 Repository Structure

IOT_Cyber_Project/
│
├── data/
│   ├── raw/
│   │   ├── Merged01.csv
│   │   ├── Merged02.csv
│   │   ├── Merged03.csv
│   │   ├── Merged04.csv
│   │   └── Merged05.csv
│   │
│   └── processed/
│       ├── ciciot2023_processed.csv
│       └── splits/
│           ├── train.csv
│           ├── validation.csv
│           ├── known_test.csv
│           └── emerging_test.csv
│
├── knowledge_base/
│   └── cybersecurity_sources_expanded.json
│
├── models/
│   ├── binary_ids_random_forest.joblib
│   ├── attack_type_random_forest.joblib
│   └── attack_probability_calibrator.joblib
│
├── results/
│   ├── paper_results_summary.csv
│   ├── ablation_results.csv
│   ├── adaptive_threshold_results.csv
│   ├── drift_adaptation_evaluation.csv
│   ├── end_to_end_results.csv
│   ├── end_to_end_rag_results.csv
│   ├── emerging_rag_evidence_evaluation.csv
│   ├── emerging_rag_evidence_summary.csv
│   ├── rag_retrieval_comparison.csv
│   ├── routing_policy_comparison.csv
│   ├── threshold_tradeoff.csv
│   └── ...
│
├── src/
│   ├── preprocess_data.py
│   ├── split_dataset.py
│   ├── train_ids.py
│   ├── evaluate_ids.py
│   ├── evaluate_emerging.py
│   ├── analyze_confidence.py
│   ├── calibrate_model.py
│   ├── train_attack_classifier.py
│   ├── evaluate_attack_classifier.py
│   ├── detect_drift.py
│   ├── edge_decision.py
│   ├── orchestrator.py
│   ├── rag_request.py
│   ├── rag_contract.py
│   ├── rag_retriever.py
│   ├── rag_pipeline.py
│   ├── rag_integration.py
│   ├── run_rag.py
│   ├── run_end_to_end.py
│   ├── evaluate_rag.py
│   ├── evaluate_ablation.py
│   ├── evaluate_always_vs_selective_rag.py
│   ├── evaluate_drift_adaptation.py
│   └── evaluate_emerging_rag_evidence.py
│
├── tests/
│   ├── test_rag.py
│   └── test_person2_rag_integration.py
│
├── examples/
│   └── rag_request.json
│
├── .gitignore
└── README.md

Raw dataset files and generated model artifacts are excluded from GitHub through .gitignore.

▶️ Running the Project

1. Preprocess the dataset

python src/preprocess_data.py

2. Create the known/emerging splits

python src/split_dataset.py

3. Train the binary IDS

python src/train_ids.py

4. Evaluate the binary IDS

python src/evaluate_ids.py

5. Evaluate emerging attacks

python src/evaluate_emerging.py

6. Calibrate probabilities

python src/calibrate_model.py

7. Train/evaluate attack classification

python src/train_attack_classifier.py
python src/evaluate_attack_classifier.py

8. Run drift detection

python src/detect_drift.py

9. Run the end-to-end architecture

python src/run_end_to_end.py

10. Run RAG retrieval evaluation

python src/evaluate_rag.py

11. Run the research evaluations

python src/evaluate_ablation.py
python src/evaluate_always_vs_selective_rag.py
python src/evaluate_drift_adaptation.py
python src/evaluate_emerging_rag_evidence.py

✅ Tests

The project includes automated integration/unit tests for the orchestration → RAG interface and RAG provenance behavior.

Run:

python -m unittest discover -s tests -p "test_*.py" -v

Current suite:

Ran 9 tests

OK

🔐 Research Integrity & Safety

This repository deliberately separates:

Evaluation truth

Ground-truth labels are used to calculate metrics.

Runtime evidence

The RAG layer receives observable behavior context, not the ground-truth answer.

Generated interpretation

The deterministic generator describes retrieved findings as candidate/evidence-supported hypotheses, rather than confirmed diagnoses.

Containment

Containment guidance is presented as hypothetical/operator-reviewed guidance, not blind autonomous blocking.

This separation is important for avoiding evaluation leakage and overclaiming.

⚠️ Limitations

This project is a research prototype, not a production SOC platform.

1. Dataset scope

The experiments use CIC-IoT-2023 only.

Cross-dataset generalization has not been established.

2. Controlled emerging evaluation

The emerging classes are held out by class, but the stream is not presented as a natural chronological deployment trace.

3. Imperfect uncertainty under shift

Calibration improves Brier scores but does not eliminate uncertainty errors on emerging distributions.

4. Drift did not add unique routing decisions in this stream

ADWIN detected the controlled transition, but the detected event did not generate additional RAG escalations beyond confidence-based routing.

5. Retrieval/evidence alignment is limited for actual emerging false negatives

Only about 20.40% of emerging false-negative RAG cases had relevant evidence within the top four results in the evaluated experiment.

6. Deterministic RAG generator

The current grounded generator is a reproducible offline baseline and is not a production LLM.

7. Small retrieval benchmark

The retrieval comparison uses eight hand-authored cybersecurity cases, so its results should not be interpreted as universal retrieval performance.

🔭 Future Work

The current architecture provides several natural research extensions:

richer behavior representations for retrieval;

stronger traffic-to-cybersecurity semantic mapping;

larger and independently annotated retrieval benchmarks;

evaluation across additional IoT datasets;

online/temporal deployment streams;

stronger adaptive thresholding;

real LLM adapter evaluation;

deeper recommendation-quality and analyst-utility studies;

deployment on constrained edge hardware.

👥 Team Responsibilities

The project was developed as a three-person research effort.

Role

Main responsibilities

👤 Person 1

Dataset processing, preprocessing, ML IDS, initial detection, calibration/drift integration, end-to-end integration

👤 Person 2

Edge orchestration, adaptive routing, threshold analysis, ablation/evaluation, resource/latency analysis

👤 Person 3

Provenance-aware RAG, cybersecurity knowledge base, retrieval, grounded response generation, RAG evaluation

📝 Research Positioning

The project is intended to answer a focused systems-research question:

Can an IoT security pipeline use lightweight edge detection as the first line of defense while selectively allocating additional cybersecurity reasoning to cases where uncertainty or changing traffic conditions indicate that local inference may be insufficient?

The experiments do not assume that every component improves every metric.

Instead, they evaluate:

Detection
   +
Calibration
   +
Drift Awareness
   +
Selective Routing
   +
Evidence Retrieval
   +
Grounded Explanation
   +
Resource Efficiency

The resulting system is therefore best understood as an adaptive security orchestration framework, rather than simply an IDS or simply a RAG chatbot.

📌 Reproducibility Notes

For reproducibility, the repository keeps:

source code;

research evaluation scripts;

processed research results;

knowledge-base records;

retrieval benchmarks;

routing/ablation outputs;

automated tests.

Large raw datasets and generated model binaries are intentionally excluded from GitHub.

Random seed used in the controlled experiments:

42

<div align="center">

🛡️ Edge ML first. Evidence when needed.

Resource-efficient detection • uncertainty-aware routing • drift monitoring • provenance-aware cybersecurity RAG

</div>