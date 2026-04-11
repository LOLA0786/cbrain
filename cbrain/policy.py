def apply_policy(risk: float) -> str:
    if risk > 0.8:
        return "BLOCK"
    elif risk > 0.5:
        return "REVIEW"
    return "ALLOW"
