from cbrain import evaluate_action
import sys

action = sys.argv[1] if len(sys.argv) > 1 else "read_logs"

print(f"[gbrain] Planned: {action}")

result = evaluate_action(action)

print(f"[cbrain] Risk: {result.risk}")
print(f"[cbrain] Decision: {result.decision}")
print(f"[privatevault] Enforcement: {result.decision}")
