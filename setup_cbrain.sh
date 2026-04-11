#!/bin/bash

mkdir -p cbrain examples tests

# __init__.py
cat << 'EOPY' > cbrain/__init__.py
from .engine import evaluate_action
EOPY

# types.py
cat << 'EOPY' > cbrain/types.py
from dataclasses import dataclass

@dataclass
class Decision:
    action: str
    risk: float
    decision: str
EOPY

# risk.py
cat << 'EOPY' > cbrain/risk.py
def score_risk(action: str) -> float:
    high_risk_keywords = ["transfer", "delete", "shutdown", "funds"]
    if any(word in action.lower() for word in high_risk_keywords):
        return 0.9
    return 0.2
EOPY

# policy.py
cat << 'EOPY' > cbrain/policy.py
def apply_policy(risk: float) -> str:
    if risk > 0.8:
        return "BLOCK"
    elif risk > 0.5:
        return "REVIEW"
    return "ALLOW"
EOPY

# engine.py
cat << 'EOPY' > cbrain/engine.py
from .risk import score_risk
from .policy import apply_policy
from .types import Decision

def evaluate_action(action: str) -> Decision:
    risk = score_risk(action)
    decision = apply_policy(risk)
    return Decision(action=action, risk=risk, decision=decision)
EOPY

# example runner
cat << 'EOPY' > examples/run_flow.py
from cbrain import evaluate_action
import sys

action = sys.argv[1] if len(sys.argv) > 1 else "read_logs"

print(f"[gbrain] Planned: {action}")

result = evaluate_action(action)

print(f"[cbrain] Risk: {result.risk}")
print(f"[cbrain] Decision: {result.decision}")
print(f"[privatevault] Enforcement: {result.decision}")
EOPY

# test file
cat << 'EOPY' > tests/test_engine.py
from cbrain import evaluate_action

def test_high_risk():
    result = evaluate_action("transfer_funds")
    assert result.decision == "BLOCK"

def test_low_risk():
    result = evaluate_action("read_logs")
    assert result.decision == "ALLOW"
EOPY

# requirements.txt
cat << 'EOPY' > requirements.txt
pytest
EOPY

# README.md
cat << 'EOPY' > README.md
# CBrain — Execution Control Layer for AI Agents

CBrain is a decision engine that evaluates and enforces actions before execution.

## System Role

GBrain → plans  
CBrain → evaluates  
PrivateVault → enforces  

## Current Capabilities

- Rule-based risk scoring
- Decision engine: ALLOW | REVIEW | BLOCK
- CLI simulation

## Quick Demo

python examples/run_flow.py transfer_funds

## Architecture

Agent → CBrain → PrivateVault → Execution

## Setup

pip install -r requirements.txt

## Why It Matters

Adds a decision boundary before execution.
EOPY

chmod +x setup_cbrain.sh

echo "✅ Setup script created. Run: ./setup_cbrain.sh"
