from rag_request import create_rag_request


request = create_rag_request(
    prediction="ATTACK",
    raw_probability=0.62,
    calibrated_probability=0.58,
    confidence=0.58,
    drift_detected=True,
    routing_reason="LOW_CONFIDENCE_AND_DRIFT",
)


print("\n" + "=" * 50)
print("              RAG REQUEST")
print("=" * 50)

print(f"Prediction              : {request.prediction}")
print(f"Raw Probability        : {request.raw_probability:.4f}")
print(f"Calibrated Probability : {request.calibrated_probability:.4f}")
print(f"Confidence             : {request.confidence:.4f}")
print(f"Drift Detected         : {request.drift_detected}")
print(f"Routing Reason         : {request.routing_reason}")

print("=" * 50)