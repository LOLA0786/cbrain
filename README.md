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
