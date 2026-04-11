def score_risk(action: str) -> float:
    high_risk_keywords = ["transfer", "delete", "shutdown", "funds"]
    if any(word in action.lower() for word in high_risk_keywords):
        return 0.9
    return 0.2
