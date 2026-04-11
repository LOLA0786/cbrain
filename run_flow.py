import sys
import subprocess

action = sys.argv[1] if len(sys.argv) > 1 else "transfer_funds"

print(f"[gbrain] Planned action: {action}")

# Step 1: cbrain evaluation
cbrain_output = subprocess.check_output(
    ["python", "cbrain.py", action],
    text=True
)

print("\n" + cbrain_output)

# Extract decision
decision_line = [line for line in cbrain_output.split("\n") if "Decision:" in line]
decision = decision_line[0].split(":")[1].strip() if decision_line else "REVIEW"

# Step 2: enforce decision
if decision == "BLOCK":
    print("[PrivateVault] Enforcement:\n")
    print("RESULT: BLOCK")
    print("REASON: Blocked upstream by cbrain")
    print("POLICY: pre-execution-risk-control")
    print("EVIDENCE: Verified")
else:
    pv_output = subprocess.check_output(
        ["bash", "-c", f"cd ~/PrivateVault.ai && python replay_demo.py {action}"],
        text=True
    )
    print("[PrivateVault] Enforcement:\n")
    print(pv_output)
