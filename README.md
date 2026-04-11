# cbrain
CBrain: Execution Control Layer for AI Agents. Real-time risk scoring, policy enforcement (Allow/Review/Block), signed evidence, and deterministic replay
CBrain — The Control Brain for Agentic Systems
Runtime enforcement, risk-aware decisions, and auditability for production-grade AI agents.
While GBrain gives your agent memory, skills, and intelligence — CBrain ensures it makes safe, governed, and auditable decisions before any action is executed.
The Missing Layer
Most agent frameworks today have:

LLM + Memory (reasoning)
Tools / Code Execution (action)

What's missing: A hardened decision enforcement layer in between.
CBrain fills that gap.

✨ Features
Core Decision Engine

Real-time policy evaluation and risk scoring for every planned action
Three-tier decisions: ALLOW | REVIEW | BLOCK
Context-aware, intent-aware, and policy-driven rulings
Adaptive risk control that adjusts based on behavior and sensitivity

Safety & Protection

Simulation mode (fast dry-run before execution)
Irreversible action protection (delete, transfer, rm -rf, etc.)
PrivateVault sandbox integration with automatic cleanup
Authority & permission modeling at runtime

Audit & Transparency

Deterministic Replay System — Replay any decision exactly as it happened
Cryptographically signed evidence + verification hashes
Tamper-proof decision receipts
Full audit trail (intent → risk → policy → outcome)

Integration & DX

Native integration with GBrain, Hermes, OpenClaw, and LangGraph-style agents
Seamless gbrain → CBrain → Execution pipeline
Clean CLI tools for policy testing, replays, and monitoring
Multi-tenant & policy scoping support
Pluggable policy architecture (TypeScript + Python)

Advanced Capabilities

Sovereign & finance-grade policies
Safe observability policies
Dream-cycle compatible (works with GBrain’s nightly maintenance)
High-performance decision engine (<50ms typical latency)


Quick Start
Bash# Install
npm install @cbrain/core

# Or via npx
npx cbrain init
TypeScriptimport { CBrain } from '@cbrain/core';

const cbrain = new CBrain({
  policies: ['adaptive-risk-control', 'safe-observability', 'sovereign-v1-finance'],
  vault: 'privatevault.ai'
});

// Agent plans an action
const decision = await cbrain.evaluate({
  intent: "transfer_funds",
  context: { amount: 50000, user: "admin" },
  agentId: "hermes_agent"
});

console.log(decision.result); // ALLOW | REVIEW | BLOCK

Example: Real Decision Replay
Bash# Replay any past decision
npx cbrain replay 0x20688bc5
Output:

Result: REVIEW
Reason: Unknown or medium-risk action
Policy: adaptive-risk-control
Evidence: 0xfacec35828 (Verified)


Why CBrain?



































LayerWithout CBrainWith CBrainReasoningGBrain / LLMGBrain / LLMDecision MakingPrompt-basedPolicy + Risk EngineExecutionDirectGated + AuditedSafetyHope & promptsDeterministic enforcementAuditabilityNone / logsSigned evidence + replays

Use Cases

Financial agents
Autonomous research agents
Enterprise & sovereign AI systems
High-stakes agent deployments
Safety-critical agent testing


Roadmap

 Advanced authority modeling (OTANIS compatible)
 Web dashboard + visualization
 Policy marketplace
 Formal verification support


Built For
People who want agents that are both powerful and trustworthy.

Made with ❤️ by Chandan Galani
PrivateVault.ai | Intent-Engine
