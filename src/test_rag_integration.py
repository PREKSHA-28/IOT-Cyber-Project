from orchestrator import EdgeOrchestrator


orchestrator = EdgeOrchestrator(confidence_threshold=0.70)


# Simulate a sample that should be routed to RAG
result = orchestrator.decide(
    prediction="ATTACK",
    raw_probability=0.62,
    calibrated_probability=0.58,
    confidence=0.95,
    drift_detected=False,
)


rag_request = orchestrator.create_rag_request(result)


print("\n" + "=" * 60)
print("              ORCHESTRATOR → RAG")
print("=" * 60)

print(f"Routing Decision        : {result.routing_decision}")
print(f"Routing Reason          : {result.routing_reason}")

print("-" * 60)

if rag_request:
    print(f"Prediction              : {rag_request.prediction}")
    print(f"Raw Probability        : {rag_request.raw_probability:.4f}")
    print(f"Calibrated Probability : {rag_request.calibrated_probability:.4f}")
    print(f"Confidence             : {rag_request.confidence:.4f}")
    print(f"Drift Detected         : {rag_request.drift_detected}")
    print(f"Routing Reason         : {rag_request.routing_reason}")
else:
    print("No RAG request created.")

print("=" * 60)