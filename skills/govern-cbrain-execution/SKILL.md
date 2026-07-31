---
name: govern-cbrain-execution
description: Build, review, or operate CBrain-governed AI agent execution with PrivateVault authorization, credential-free model tools, exact-byte sidecar dispatch, evidence closure, and model-invariance evaluation. Use for CBrain integrations, consequential agent tool calls, provider/model routing, Hermes or framework adapters, CRM/payment scenarios, PrivateVault decision wiring, dispatcher configuration, or five-model safety evaluations.
---

# Govern CBrain Execution

Treat model inference as proposal generation only. Never grant a model, framework
adapter, memory system, or ordinary tool callback execution authority.

## Workflow

1. Inspect `AGENTS.md`, `upstreams.lock.json`, and the affected CBrain contracts.
2. Translate the framework tool proposal into an immutable `ActionIntent`.
3. Keep provider selection, endpoints, credentials, and routes outside model output.
4. Expose credential-free tool schemas. Reject credential-shaped schema fields or
   returned arguments.
5. Send consequential actions to PrivateVault. Treat BLOCK, REVIEW, malformed
   responses, timeouts, or unavailable controls as non-executable.
6. On ALLOW, require signed authorization, atomic single-use consumption, exact-byte
   sidecar dispatch, witness verification, and closure verification.
7. Classify uncertainty after sending may have begun as non-retryable INDETERMINATE.
8. Evaluate model proposals separately from gate decisions. Compare identical
   `ActionIntent` bytes across models and report proposal frequency independently.
9. Run the repository's formatting, strict typing, full tests, compile, lock, and
   diff checks before handoff.

## Guardrails

- Never place secrets in messages, tool schemas, tool arguments, memory, logs, or
  evaluation artifacts.
- Never let model arguments choose a URL, provider, credential audience, TLS peer,
  artifact digest, retry policy, or transport.
- Never add policy logic to model or framework adapters.
- Never claim a callback proves bytes on the wire. Only the independent sidecar can
  witness the bytes and peer it actually observed.
- Never automatically retry INDETERMINATE execution.
- Never report “model agnostic” without separating decision invariance from model
  proposal behavior.

## Repository surfaces

- `cbrain/models/`: neutral contracts, provider adapters, HTTPS transport, routing.
- `cbrain/evaluation/`: scenarios, model generation matrix, decision matrix.
- `cbrain/simulators/`: mutable CRM and payment targets plus dispatch planner.
- `cbrain/execution/`: authorization, transport, independent sidecar, closure.
- `integrations/`: framework-specific translation only.

Read [references/contracts.md](references/contracts.md) for failure semantics,
provider settings, evaluation claims, and required verification commands.
