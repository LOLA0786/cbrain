import sys
import hashlib

action = sys.argv[1] if len(sys.argv) > 1 else "unknown_action"

def classify(action):
    a = action.lower()

    if any(x in a for x in ["transfer", "payment", "fund"]):
        return "HIGH", "financial_irreversible"

    if any(x in a for x in ["delete", "rm", "drop"]):
        return "HIGH", "destructive_irreversible"

    if any(x in a for x in ["update", "modify"]):
        return "MEDIUM", "state_change"

    if any(x in a for x in ["read", "view", "logs"]):
        return "LOW", "read_only"

    return "UNKNOWN", "unclassified"

def evaluate(action):
    level, category = classify(action)

    if level == "HIGH":
        decision = "BLOCK"
        risk = 0.9
    elif level == "MEDIUM":
        decision = "REVIEW"
        risk = 0.6
    elif level == "LOW":
        decision = "ALLOW"
        risk = 0.2
    else:
        decision = "REVIEW"
        risk = 0.5

    decision_id = "0x" + hashlib.sha256(action.encode()).hexdigest()[:8]

    return {
        "id": decision_id,
        "action": action,
        "level": level,
        "category": category,
        "risk": risk,
        "decision": decision
    }

result = evaluate(action)

print(f"[cbrain] Action: {result['action']}")
print(f"Decision ID: {result['id']}")
print(f"Risk Score: {result['risk']}")
print(f"Level: {result['level']}")
print(f"Category: {result['category']}")
print(f"Decision: {result['decision']}")
