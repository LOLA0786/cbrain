from .policy import apply_policy
from .risk import score_risk
from .types import Decision


def evaluate_action(action: str) -> Decision:
    risk = score_risk(action)
    decision = apply_policy(risk)
    return Decision(action=action, risk=risk, decision=decision)
