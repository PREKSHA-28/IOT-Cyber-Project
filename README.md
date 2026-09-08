<div align="center">

# 🛡️ IoT Edge-AI Cybersecurity Framework

### Resource-Efficient • Uncertainty-Aware • Continually Adaptive

**An end-to-end research prototype for detecting known and emerging IoT cyber threats at the edge, monitoring uncertainty and concept drift, and selectively escalating difficult cases to provenance-aware cybersecurity RAG.**

<p>
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/Random%20Forest-Edge%20IDS-F7931E?style=for-the-badge">
  <img src="https://img.shields.io/badge/ADWIN-Concept%20Drift-6C3483?style=for-the-badge">
  <img src="https://img.shields.io/badge/RAG-Provenance--Aware-00A67E?style=for-the-badge">
  <img src="https://img.shields.io/badge/MITRE%20ATT%26CK-Knowledge-111111?style=for-the-badge">
  <img src="https://img.shields.io/badge/IEEE-Research%20Prototype-00629B?style=for-the-badge">
</p>

</div>

---

# 🚀 What did we build?

Instead of sending every IoT network event to an expensive cybersecurity reasoning layer, this project uses a **two-level security architecture**:

> **Edge AI first. Additional cybersecurity reasoning only when it is needed.**

The system first performs lightweight intrusion detection locally.

It then asks:

- How confident is the model?
- Has the traffic stream changed?
- Does this case require additional cybersecurity knowledge?

Only selected cases are escalated to a **provenance-aware Retrieval-Augmented Generation (RAG) layer**.

```mermaid
flowchart LR
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

    J --> K[Observable Behavior Context]
    K --> L[Retrieval]

    L --> M[Lexical Retrieval]
    L --> N[Semantic Retrieval]

    M --> O[Provenance-aware Cybersecurity KB]
    N --> O

    O --> P[Grounded Response]
    P --> Q[Candidate Threat]
    P --> R[Uncertainty]
    P --> S[Recommendation]
    P --> T[Containment Guidance]
    P --> U[Evidence]
🎯 Core Research Idea

The project is not simply "Random Forest + RAG."

The central research contribution is the selective orchestration mechanism connecting:

Edge intrusion detection
↓
Probability calibration
↓
Prediction confidence
↓
Concept-drift monitoring
↓
Dynamic routing
↓
Provenance-aware cybersecurity retrieval
↓
Evidence-grounded explanation and mitigation guidance

The goal is to reduce unnecessary RAG/LLM interactions while still giving difficult or uncertain cases access to broader cybersecurity knowledge.

📌 Problem Statement

Design and develop a resource-efficient, uncertainty-aware, continually adaptive Edge-AI cybersecurity framework for IoT networks that detects known and emerging cyber threats under evolving network conditions.

The framework is designed to:

perform lightweight intrusion and anomaly detection at the edge;
continuously monitor prediction uncertainty;
detect changes in the traffic stream using concept-drift detection;
dynamically decide whether a case can be handled locally or requires additional cybersecurity knowledge;
selectively invoke a provenance-aware RAG pipeline for low-confidence, anomalous, or emerging cases;
retrieve relevant evidence from trusted cybersecurity knowledge sources;
generate evidence-grounded threat interpretations and context-specific mitigation guidance;
minimize unnecessary RAG/LLM interactions while maintaining useful detection performance and low response latency.
⭐ Key Results at a Glance
Metric	Result
Processed CIC-IoT-2023 samples	~2.21M
Network-flow features	39
Held-out emerging attack classes	3
Controlled end-to-end evaluation	20,000 samples
Local decisions	80.30%
RAG decisions	19.71%
RAG workload avoided vs Always-RAG	80.30%
Measured cumulative RAG latency reduction	~80.53%
Known-test binary IDS recall	98.90%
Known-test binary IDS F1	99.44%
Emerging-test binary IDS recall	34.67%
Retrieval benchmark cases	8
Retrieval benchmark Hit@1	1.00 lexical / 1.00 semantic
Automated tests	9/9 passing

The results above are from the current controlled research evaluation and should not be interpreted as universal production performance.

🧠 Why this architecture?

A conventional IDS can answer:

"Is this traffic malicious?"

But real cybersecurity operations often need additional answers:

"How confident are we?"
"Has the traffic behavior changed?"
"What cybersecurity knowledge is relevant?"
"What should an analyst investigate next?"

This project separates those responsibilities.

Layer	Responsibility
⚡ Edge ML	Fast first-stage intrusion detection
📏 Calibration	Produce better-behaved attack probabilities
🌊 Drift Detector	Monitor changing prediction behavior
🧭 Orchestrator	Decide LOCAL vs RAG
📚 Knowledge Base	Store source-linked cybersecurity knowledge
🔎 Retriever	Find relevant evidence
🧾 Grounded Generator	Produce evidence-supported interpretation
🛠️ Recommendation Layer	Provide context-specific mitigation guidance
🏗️ Complete System Architecture
🧩 System Components
1. Data and Preprocessing

The system starts from CIC-IoT-2023 network-flow data.

The preprocessing stage produces:

cleaned numerical features;
binary attack labels;
multiclass attack labels;
training/validation/test splits;
a dedicated held-out emerging-attack evaluation set.
2. Edge Intrusion Detection

The first-stage detector is a Random Forest binary classifier.

It predicts:

BENIGN
ATTACK

Configuration:

n_estimators = 100
max_depth = 20
class_weight = balanced
random_state = 42
n_jobs = -1

Model artifact:

models/binary_ids_random_forest.joblib
Known-test performance
Metric	Result
Precision	99.99%
Recall	98.90%
F1-score	99.44%
Emerging-attack performance
Metric	Result
Precision	100.00%
Recall	34.67%
F1-score	51.49%

The large drop in emerging-threat recall motivates the additional reasoning layer.

🎯 3. Attack-Type Classification

A second Random Forest is trained on the known attack classes to classify attack family/type after malicious traffic is identified.

Known attack classes   = 30
Held-out emerging      = 3

Model artifact:

models/attack_type_random_forest.joblib
Known-test performance
Metric	Result
Accuracy	~78%
Macro F1	0.626
Weighted F1	0.775

The lower macro F1 reflects the difficulty of classifying rare and imbalanced attack classes.

📏 4. Prediction Probability Calibration

Random Forest output probabilities are used by the routing layer as a signal of uncertainty.

To improve their interpretation, the project applies:

Isotonic Regression

The calibrator is trained on the validation set only.

Artifact:

models/attack_probability_calibrator.joblib
Brier score comparison
Split	Raw	Calibrated
Validation	0.009203	0.006140
Known test	0.009128	0.006095
Emerging	0.561658	0.259323

Calibration improves the measured probability quality, although uncertainty remains imperfect under distribution shift.

🌊 5. Concept Drift Detection

The streaming component uses:

ADWIN — Adaptive Windowing

from the River library.

Configuration:

delta = 0.002

The detector monitors the calibrated attack-probability stream.

Important research interpretation

The drift experiment uses a controlled transition from known traffic to held-out emerging traffic.

It is not claimed to be a naturally occurring chronological deployment stream.

The drift event was detected near the transition.

However, the subsequent routing experiment found that the detected drift produced no additional unique RAG escalations beyond those already triggered by low confidence in this stream.

That negative result is intentionally retained.

🧭 6. Edge Orchestrator

The Edge Orchestrator is the central routing component.

It receives:

prediction
raw_probability
calibrated_probability
confidence
drift_detected

and produces:

prediction
raw_probability
calibrated_probability
confidence
drift_detected
routing_decision
routing_reason
Default routing threshold
confidence_threshold = 0.70
Routing policy
Confidence	Drift	Routing
High	No	✅ LOCAL
Low	No	🔎 RAG
High	Yes	🔎 RAG
Low	Yes	🔎 RAG

The threshold is treated as a resource-aware engineering/research trade-off, not as a globally optimal threshold.

⚖️ 7. Adaptive Threshold Evaluation

Several confidence thresholds were evaluated.

Threshold	RAG Invocation	False-Negative Escalation
0.60	6.63%	19.50%
0.65	9.86%	29.01%
0.70	19.84%	58.38%
0.75	22.34%	65.58%
0.80	24.31%	71.22%
0.85	26.82%	78.12%
0.90	30.73%	89.16%

The selected operating point is:

0.70

because it provides a practical trade-off between RAG workload and false-negative escalation coverage.

🔎 8. Provenance-Aware RAG

The RAG subsystem is intentionally separated from the edge detector.

The RAG request can contain:

prediction
raw_probability
calibrated_probability
confidence
drift_detected
routing_reason
behavior_context

The important part is:

The ground-truth attack label is not passed into the RAG request.

📚 9. Cybersecurity Knowledge Base

Active knowledge base:

knowledge_base/cybersecurity_sources_expanded.json

Current size:

37 source-linked records

Sources represented include:

MITRE ATT&CK
NIST
OWASP
CISA
CWE

Topics include:

Network Service Scanning
Name Resolution Poisoning
HTTP flooding
Network Denial of Service
Brute Force
Exploitation of Remote Services
Command Injection
SQL Injection
IoT security capabilities
Incident response
Containment
Hard-coded credentials

Each knowledge record stores provenance-related information such as:

source identity
document identifier
chunk identifier
document title
document type
topic / attack family
source URL
content
Why provenance?

Cybersecurity recommendations should not appear from nowhere.

The system carries retrieved evidence forward so that an analyst can trace the response back to the cybersecurity source material used by the retrieval stage.

🔬 10. Retrieval Engine

Two retrieval strategies are implemented.

Lexical Retrieval

A transparent token-overlap baseline is used across fields such as:

document_title
document_type
topic
attack_family
content

Advantages:

lightweight;
transparent;
deterministic;
easy to inspect.
Semantic Retrieval

Semantic search uses:

Sentence Transformers
all-MiniLM-L6-v2

Embedding dimension:

384

The embedding model and document embeddings are loaded/cached lazily so lexical-only experiments do not need to initialize the semantic model.

📈 11. Retrieval Benchmark

The retrieval evaluation contains 8 hand-authored cybersecurity query cases covering:

vulnerability scanning;
DNS-related behavior;
HTTP flooding;
brute force;
command injection;
SQL injection;
network DoS;
IoT security.
Results
Metric	Lexical	Semantic
Hit@1	1.000	1.000
Hit@3	1.000	1.000
Hit@4	1.000	1.000
MRR	1.000	1.000
Precision@1	1.000	1.000
Precision@3	0.917	0.833
Precision@4	0.813	0.719
Recall@4	0.813	0.750
NDCG@4	0.925	0.860

These results apply specifically to the eight-case retrieval benchmark. They are not universal retrieval-performance claims.

🧾 12. Grounded Response Generation

The current repository uses a:

Deterministic Grounded Generator

This was intentionally used for reproducible offline evaluation.

The generator:

receives a structured RAG request;
receives retrieved evidence;
validates cited evidence identifiers;
derives a candidate threat interpretation from retrieved evidence;
generates structured explanation fields;
selects evidence-grounded recommendations;
preserves provenance;
includes a caveat about the deterministic baseline.

The output contains:

Threat
Detection Interpretation
Uncertainty Reason
Likely Attack Behavior
Recommendation
Containment Action
Caveat
Evidence
Important

The deterministic generator is not a production autonomous cybersecurity LLM.

The architecture exposes an adapter-style interface that can be connected to a real LLM in future work.

🔄 13. Label-Free Behavior Context

A particularly important part of the implementation is the behavior context passed to RAG.

The RAG layer receives observable traffic information such as:

active_protocols
active_application_protocols
active_tcp_flags
traffic_statistics
retrieval_query

The context is deliberately constructed without:

Label
Label_Binary

This prevents the evaluation answer from being directly passed into retrieval or generation.

This distinction is important for research integrity.

🧪 14. End-to-End Evaluation

The complete pipeline is implemented in:

src/run_end_to_end.py

Controlled evaluation stream:

10,000 KNOWN
10,000 EMERGING
----------------
20,000 TOTAL
Routing results
Decision	Samples	Percentage
LOCAL	16,059	80.30%
RAG	3,941	19.71%
Phase-wise routing
Phase	LOCAL	RAG
KNOWN	98.58%	1.42%
EMERGING	62.01%	37.99%

The RAG stage logs the evidence identifiers retrieved for every escalated sample.

Measured deterministic offline RAG timing
Measurement	Mean
Retrieval latency	0.074 ms
Generation latency	0.005 ms
Total RAG latency	0.140 ms

These are measurements of the current deterministic local implementation and should not be interpreted as production LLM API latency.

♻️ 15. Selective-RAG vs Always-RAG

A dedicated experiment compares the proposed selective strategy with an Always-RAG baseline.

Selective-RAG

Only routed cases invoke RAG:

3,941 / 20,000
= 19.71%
Always-RAG

Every sample invokes RAG:

20,000 / 20,000
= 100%
Workload reduction
20,000 - 3,941
= 16,059 RAG calls avoided

Therefore:

RAG workload reduction
= 80.30%

Measured cumulative deterministic RAG latency reduction:

≈ 80.53%

This demonstrates reduced RAG workload under the evaluated deterministic offline implementation. It should not be directly generalized into claims such as "80% lower real-world LLM cost."

🧪 16. Three-System Ablation

The project includes a dedicated ablation study.

System A — Edge ML Only
Random Forest
      ↓
LOCAL

No RAG escalation.

System B — ML + Confidence RAG
confidence < 0.70
        ↓
      RAG
System C — Proposed
confidence < 0.70
        OR
drift_detected
        ↓
      RAG
Why this ablation matters

All systems use the same:

model predictions;
probabilities;
evaluated samples.

Therefore routing does not change the underlying Random Forest prediction.

The ablation isolates:

RAG allocation;
false-negative escalation;
effect of drift-aware routing.
Important observed result

In the current controlled stream:

Drift awareness produced no additional unique RAG escalations beyond confidence-based routing.

This is kept as a negative research finding rather than being hidden.

🌊 17. Drift Adaptation Evaluation

The drift experiment compares:

CONFIDENCE_ONLY

against:

CONFIDENCE_PLUS_DRIFT

using a symmetric 5,000-sample window before and after the detected transition.

The experiment measures:

RAG invocation rate;
emerging false-negative escalation;
local false negatives;
drift-only escalations;
additional RAG invocations produced by drift.

The measured stream showed:

ADWIN detected the transition, but the affected samples were already being escalated because of low confidence.

This means the incremental routing contribution of drift was zero in this particular stream.

🚨 18. Emerging False-Negative → RAG Evidence Evaluation

One of the most important research experiments asks:

When the edge model misses an emerging attack and escalates it to RAG, does the retrieved evidence actually correspond to the attack family?

Expected evidence mappings:

VULNERABILITYSCAN
        ↓
MITRE T1046

DNS_SPOOFING
        ↓
MITRE T1557.001

DOS-HTTP_FLOOD
        ↓
MITRE T1499.002
Evaluation rule

Ground-truth labels are used only inside this evaluation script to determine whether retrieved evidence is relevant.

The label is never passed into the RAG pipeline.

Observed result

Across:

3,799

emerging false-negative cases routed to RAG:

≈ 20.40%

had relevant evidence within the top-four retrieved items.

This should not be described as:

"20.40% RAG attack detection accuracy."

It is an:

evidence-alignment measurement for emerging false-negative cases routed to RAG.

💡 19. Important Research Insight

One of the strongest observations from the experiments is that:

Detecting malicious traffic
            ≠
Identifying the right cybersecurity knowledge

The edge model works on aggregate network-flow features.

Cybersecurity knowledge bases, however, are written in concepts such as:

Network Service Scanning
Name Resolution Poisoning
Brute Force
Remote Service Exploitation
Command Injection

This creates a representation/vocabulary gap.

The emerging-evidence experiment exposed that gap directly.

This is an important limitation, but also a natural future research direction.

📊 20. Resource and Latency Evaluation

The project also measures the overhead of the orchestration layer.

Orchestrator latency

Measured on a 20,000-sample evaluation:

Metric	Result
Mean	13.625 μs
Median	11.6 μs
P95	21.4 μs
P99	35 μs
Throughput	~22,166 samples/s
Resource measurement

Measured on the project execution environment:

Metric	Result
CPU measurement	101.8%
Memory at start	72.94 MB
Memory at end	78.14 MB
Memory change	+5.20 MB

These are measurements of the experimental environment, not universal hardware-independent performance guarantees.

🧠 21. Why selective RAG matters

The project's central resource-efficiency argument is:

          All traffic
              │
              ▼
        Edge ML first
              │
       ┌──────┴──────┐
       │             │
       ▼             ▼
    Easy cases    Difficult cases
       │             │
       ▼             ▼
     LOCAL          RAG

Instead of:

All traffic
    │
    ▼
  RAG
    │
    ▼
Extra reasoning everywhere

The experiment showed that the selective policy reduced the number of RAG invocations by:

80.30%

in the controlled 20,000-sample evaluation.

🛠️ Technology Stack
Technology	Role
Python	Main implementation language
pandas	Data processing and experiment analysis
NumPy	Numerical operations
scikit-learn	Machine learning and calibration
Random Forest	Binary IDS + attack-type classification
Isotonic Regression	Probability calibration
River	Streaming analytics
ADWIN	Concept-drift detection
Sentence Transformers	Semantic retrieval
all-MiniLM-L6-v2	384-dimensional semantic embeddings
PyTorch	Semantic model backend
Transformers	NLP/model stack
JSON	Cybersecurity knowledge base
psutil	CPU/memory measurement
unittest	Automated testing
Git	Version control
GitHub	Collaboration and project hosting
🧱 Architecture Choices

The current prototype deliberately avoids unnecessary infrastructure.

It does not require:

MySQL
MongoDB
Elasticsearch
A separately hosted vector database
A dedicated backend server

The current research implementation uses:

Source-linked JSON Knowledge Base
+
In-process lexical retrieval
+
In-process semantic retrieval

This keeps the prototype lightweight and reproducible.

📂 Repository Structure
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
│   ├── adaptive_threshold_results.csv
│   ├── ablation_results.csv
│   ├── drift_adaptation_evaluation.csv
│   ├── drift_detection_results.csv
│   ├── routing_policy_comparison.csv
│   ├── threshold_tradeoff.csv
│   ├── always_vs_selective_rag.csv
│   ├── always_vs_selective_rag_phase.csv
│   ├── always_vs_selective_rag_savings.csv
│   ├── always_vs_selective_rag_summary.csv
│   ├── emerging_false_negative_escalation.csv
│   ├── emerging_rag_evidence_details.csv
│   ├── emerging_rag_evidence_evaluation.csv
│   ├── emerging_rag_evidence_summary.csv
│   ├── rag_evaluation_results.csv
│   ├── rag_evaluation_summary.json
│   ├── rag_retrieval_comparison.csv
│   ├── rag_retrieval_comparison_summary.json
│   ├── end_to_end_results.csv
│   ├── end_to_end_rag_results.csv
│   ├── end_to_end_drift_events.csv
│   ├── orchestrator_latency_results.csv
│   └── orchestrator_resource_results.csv
│
├── src/
│   ├── inspect_dataset.py
│   ├── preprocess_data.py
│   ├── check_processed_data.py
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
│   ├── evaluate_routing.py
│   ├── adaptive_threshold_evaluation.py
│   ├── compare_routing_policies.py
│   ├── create_threshold_tradeoff.py
│   ├── evaluate_emerging_fn_escalation.py
│   ├── measure_orchestrator_latency.py
│   ├── measure_orchestrator_resources.py
│   ├── create_paper_results.py
│   ├── orchestrator.py
│   ├── rag_request.py
│   ├── rag_contract.py
│   ├── rag_retriever.py
│   ├── rag_pipeline.py
│   ├── rag_integration.py
│   ├── rag_eval...
│   ├── run_rag.py
│   ├── llm_adapter.py
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

Large raw datasets and generated model binaries are intentionally excluded from GitHub.

📦 Dataset Details
CIC-IoT-2023

The dataset is organized locally into five merged CSV partitions:

Merged01.csv
Merged02.csv
Merged03.csv
Merged04.csv
Merged05.csv

These are partitions of the same CIC-IoT-2023 dataset.

Processed dataset
Rows    ≈ 2,207,338
Columns = 41
Features = 39
Labels   = 2

Binary distribution:

ATTACK = 2,123,497
BENIGN =    83,841

Feature groups include:

Header / packet statistics
Protocol indicators
TCP flags
Application protocol indicators
Traffic rate/count features
Packet-size statistics
Inter-arrival time
Variance/statistical features
🚨 Emerging-Attack Evaluation Design

The following attack classes are held out from model training:

DNS_SPOOFING
VULNERABILITYSCAN
DOS-HTTP_FLOOD

The experiment therefore distinguishes:

Known attacks
      ↓
Model development / evaluation

Emerging attacks
      ↓
Held out from model training

This setup evaluates generalization to withheld attack classes.

It does not claim chronological concept drift in the original dataset.

🔐 Research Integrity

The implementation deliberately separates:

Ground truth

Used for:

evaluation
metrics
error analysis
Runtime observation

Passed to the RAG layer through:

observable behavior context
Ground truth is not passed to RAG

The RAG system does not receive:

Label
Label_Binary

during the end-to-end inference path.

Generated output

The system uses terms such as:

candidate threat
evidence-supported inference
uncertainty
analyst review

rather than claiming that retrieved evidence itself proves an attack occurred.

🛡️ Safety-Oriented Response Design

The system does not blindly perform autonomous blocking.

Instead, generated containment guidance is framed as:

Hypothetical operator-reviewed containment

Examples include:

reviewing suspicious traffic patterns;
applying network filtering;
validating authentication activity;
checking exposed services;
strengthening segmentation;
preserving evidence;
monitoring related activity.

This makes the architecture more suitable for analyst-assisted cybersecurity workflows.

🧪 Automated Testing

The project includes integration and unit tests covering:

RAG request validation;
orchestration-to-RAG integration;
LOCAL routing protection;
provenance handling;
grounded response generation;
latency recording;
required response fields.

Run:

python -m unittest discover -s tests -p "test_*.py" -v

Current result:

Ran 9 tests in 0.006s

OK

✅ 9/9 tests passing

▶️ Running the Project
Install core dependencies
pip install pandas numpy scikit-learn river sentence-transformers torch transformers psutil
1. Inspect the dataset
python src/inspect_dataset.py
2. Preprocess data
python src/preprocess_data.py
3. Verify processed data
python src/check_processed_data.py
4. Create known/emerging splits
python src/split_dataset.py
5. Train binary IDS
python src/train_ids.py
6. Evaluate binary IDS
python src/evaluate_ids.py
7. Evaluate emerging attacks
python src/evaluate_emerging.py
8. Analyze prediction confidence
python src/analyze_confidence.py
9. Calibrate attack probabilities
python src/calibrate_model.py
10. Train attack classifier
python src/train_attack_classifier.py
11. Evaluate attack classifier
python src/evaluate_attack_classifier.py
12. Run drift detection
python src/detect_drift.py
13. Run routing evaluation
python src/evaluate_routing.py
14. Run the complete end-to-end framework
python src/run_end_to_end.py
15. Evaluate RAG retrieval
python src/evaluate_rag.py
16. Run ablation
python src/evaluate_ablation.py
17. Compare Always-RAG vs Selective-RAG
python src/evaluate_always_vs_selective_rag.py
18. Evaluate drift adaptation
python src/evaluate_drift_adaptation.py
19. Evaluate emerging false-negative evidence alignment
python src/evaluate_emerging_rag_evidence.py
20. Run automated tests
python -m unittest discover -s tests -p "test_*.py" -v
📊 Research Experiments Included

The repository contains dedicated experiments for:

Experiment	Purpose
Binary IDS evaluation	Measure known-attack detection
Emerging attack evaluation	Measure generalization to withheld classes
Confidence analysis	Study model uncertainty
Probability calibration	Improve probability interpretation
Drift detection	Detect distribution changes
Threshold evaluation	Study confidence/RAG trade-off
Routing-policy comparison	Compare routing strategies
Three-system ablation	Isolate orchestration contribution
Always-RAG comparison	Measure RAG workload savings
Drift adaptation	Measure incremental value of drift routing
Emerging FN evidence evaluation	Test evidence alignment for missed emerging attacks
Retrieval comparison	Compare lexical and semantic search
Orchestrator latency	Measure routing overhead
Resource evaluation	Measure CPU and memory behavior
End-to-end evaluation	Validate integrated pipeline
📌 Reproducibility

Controlled research experiments use:

random_state = 42

The end-to-end controlled evaluation uses:

10,000 known
10,000 emerging
20,000 total

The emerging classes are fixed as:

DNS_SPOOFING
VULNERABILITYSCAN
DOS-HTTP_FLOOD

The selected routing threshold is:

0.70

The ADWIN configuration is:

delta = 0.002
⚠️ Limitations

This repository is a research prototype, not a production SOC platform.

1. Dataset scope

The evaluation uses CIC-IoT-2023.

Cross-dataset generalization has not been established.

2. Emerging-test design

Emerging classes are held out by class.

The evaluation is not claimed to represent natural chronological deployment.

3. Calibration under distribution shift

Calibration improves the measured Brier score but remains imperfect on emerging traffic.

4. Drift contribution

ADWIN detected the controlled transition, but drift generated no additional unique RAG escalations beyond confidence routing in the evaluated stream.

5. Evidence alignment

Only approximately 20.40% of the emerging ML false-negative RAG cases had relevant evidence within the top four retrieved results in the evaluated experiment.

6. Representation gap

Aggregate traffic features do not always contain enough cybersecurity-specific semantic information to retrieve the most appropriate threat-intelligence concepts.

7. Deterministic generator

The current generator is an offline reproducible baseline and is not a production LLM.

8. Retrieval benchmark size

The lexical/semantic retrieval comparison uses eight hand-authored benchmark cases.

Therefore those results should not be generalized to arbitrary cybersecurity queries.

🔭 Future Work

Natural extensions of the current framework include:

stronger traffic-to-cybersecurity semantic representations;
richer behavior descriptions for retrieval;
larger independently annotated retrieval benchmarks;
additional IoT datasets;
true chronological or live-stream evaluation;
stronger adaptive threshold optimization;
real LLM integration and evaluation;
analyst-utility studies;
recommendation-quality evaluation;
deployment on constrained edge hardware;
real-time streaming ingestion;
production-grade vector indexing for much larger knowledge bases.
👥 Team Contributions

This project was developed as a three-person research collaboration.

Role	Main Responsibilities
👤 Person 1	Dataset processing, preprocessing, binary IDS, confidence/calibration, drift integration, final integration and validation
👤 Person 2	Edge orchestration, routing policy, threshold analysis, ablation studies, latency/resource evaluation
👤 Person 3	Cybersecurity knowledge base, provenance-aware RAG, retrieval, grounded responses, mitigation guidance, RAG evaluation
🧠 Final Research Positioning

This project can be summarized as:

Fast detection at the edge
            +
Better uncertainty estimation
            +
Streaming drift awareness
            +
Selective security escalation
            +
Trusted cybersecurity evidence
            +
Grounded analyst guidance

The framework does not assume that every component improves every metric.

Instead, it investigates whether adaptive allocation of cybersecurity reasoning can make an IoT security pipeline more resource-aware while still providing broader context for difficult cases.

The experiments show:

Strong known-attack detection
        ↓
Weakness on unseen attack classes
        ↓
Need for additional context
        ↓
Selective RAG instead of RAG everywhere
        ↓
Reduced RAG workload
        ↓
But also a measurable traffic-to-cybersecurity
representation gap

That combination of performance, efficiency, ablation, and honest failure analysis is the central research story of the project.

📎 Important Project Files
Core Edge-AI
src/train_ids.py
src/evaluate_ids.py
src/calibrate_model.py
src/detect_drift.py
src/orchestrator.py
src/run_end_to_end.py
RAG
src/rag_request.py
src/rag_contract.py
src/rag_retriever.py
src/rag_pipeline.py
src/rag_integration.py
src/run_rag.py
knowledge_base/cybersecurity_sources_expanded.json
Research Evaluation
src/evaluate_rag.py
src/evaluate_ablation.py
src/evaluate_always_vs_selective_rag.py
src/evaluate_drift_adaptation.py
src/evaluate_emerging_rag_evidence.py
Testing
tests/test_rag.py
tests/test_person2_rag_integration.py
<div align="center">
🛡️ Edge ML First. Evidence When Needed.
Detect locally. Escalate selectively. Retrieve responsibly.

IoT Security • Edge AI • Uncertainty • Concept Drift • RAG • Provenance • Cybersecurity

</div> ```
