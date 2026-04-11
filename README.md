 CBrain — Execution Control Layer for AI Agents
The missing decision enforcement layer between planning and execution.


GBrain makes agents intelligent.
CBrain makes them safe, governed, and controllable.
Powered by PrivateVault

Philosophy
textgbrain → Plans & Reasons
cbrain → Evaluates Risk & Consequences
PrivateVault → Enforces & Audits

The Problem
Modern AI agents can reason and plan actions, but most lack a dedicated judgment layer before execution. This leads to:

Destructive commands being executed
Irreversible financial or system operations
No audit trail or policy enforcement


The Solution
CBrain is a lightweight, real-time control layer that sits between agent planning and execution. It performs:

Risk scoring
Consequence modeling
Policy-based decision making (ALLOW | REVIEW | BLOCK)
Signed evidence logging + deterministic replay


Quick Demo
Bashpython run_flow.py transfer_funds
Output:
Bash[gbrain] Planned action: transfer_funds
[cbrain] Action: transfer_funds
[cbrain] Risk Score: 0.9 | Level: HIGH | Category: financial_irreversible
[cbrain] Decision: BLOCK

[PrivateVault] RESULT: BLOCK | POLICY: pre-execution-risk-control
Bashpython run_flow.py read_system_logs
Output:
Bash[gbrain] Planned action: read_system_logs
[cbrain] Action: read_system_logs
[cbrain] Risk Score: 0.2 | Level: LOW | Category: read_only
[cbrain] Decision: ALLOW

[PrivateVault] RESULT: ALLOW | POLICY: safe-observability

Architecture
textAgent (Hermes + GBrain)
        ↓ (Planned Action)
   CBrain (Consequence & Decision Layer)
        ↓ (Risk Score + Policy)
   PrivateVault (Sandbox + Enforcement)
        ↓
Execution (or BLOCK + Audit Log)

Key Features

Real-time risk scoring and action classification
Three-tier decisions: ALLOW | REVIEW | BLOCK
Pre-execution consequence modeling
Integration with PrivateVault sandbox + auto-cleanup
Deterministic replay capability (coming soon)
Clean separation of concerns (thinking vs judgment vs enforcement)


Installation & Quick Start
Bashgit clone https://github.com/LOLA0786/cbrain.git
cd cbrain

# Run examples
python run_flow.py transfer_funds
python run_flow.py read_system_logs

Project Structure
textcbrain/
├── cbrain.py          # Core decision engine
├── run_flow.py        # Demo flows
├── README.md
└── requirements.txt   # (add when needed)

Roadmap

 Full deterministic replay system with signed evidence
 Dynamic authority & multi-step chain evaluation
 Pluggable policy engine
 Formal integration with GBrain / Hermes / OpenClaw
 Web dashboard for decision visualization


Positioning

GBrain = Memory + Skills + Reasoning
CBrain = Judgment + Risk + Control
PrivateVault = Enforcement + Sandbox + Audit


Made by Chandan Galani
Building the safety & control layer for the next generation of AI agents.
