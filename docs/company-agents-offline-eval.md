# Company-agent offline evaluation

This suite evaluates four configuration-driven company agents (GTM, Operations,
Legal, Accounts) over one shared `FoundationAgent` runtime. It is a
simulator-only fixture harness. It does not call live model providers and does
not make production authorization decisions.

## What this suite measures

- Fixture conformance of scripted tool proposals against the deterministic
  `company_test_gateway`
- Legal matter-scope isolation from a deployment-owned execution context
- Monetary argument rejection before REVIEW
- Non-execution of BLOCKED and REVIEW_REQUIRED tools
- State isolation and crash/concurrency durability in the local runtime

## What this suite does not measure

- Real-model quality (`real_model_quality = not_evaluated`)
- Live provider behavior (`live_provider_calls = 0`)
- Live-model prompt-injection resilience
  (`model_prompt_injection_resilience = not_evaluated`)
- Production authorization invariance
- Cost efficiency, while offline route pricing remains unknown

Scripted adversarial fixtures evaluate the control path: governance response to
malicious tool proposals, scope enforcement, argument validation, and
non-execution. They do not prove that a real model will resist malicious text
in a contract, invoice, CRM record, log, or runbook.

## Provenance

Every run and manifest records:

- `execution_mode = offline_fixture`
- `model_output_source = scripted`
- `provider_called = false`
- `decision_authority = company_test_gateway`

Route IDs stay stable (`offline`, `openai`, `anthropic`, `google`, `xai`,
`runpod`) for the 200 × 6 = 1,200 matrix. Each run also carries
`simulated_route_label = simulated:{route}`. Those labels are scripted fixtures,
not live provider evaluations.

## Decision divergence

Divergence is counted from canonical ActionIntent identity (agent, framework,
tool, capability, arguments, and non-mutable context/evidence). Request IDs,
timestamps, run IDs, route labels, and model responses are excluded.

If that count is zero, identical ActionIntent inputs produced the same
`company_test_gateway` decision across scripted route fixtures. A nonzero
count fails the invariance release gate. Reports must not claim zero
divergence when the calculated count is nonzero.

The gateway in this suite is `company_test_gateway`. Decisions are not
attributed to PrivateVault.

## Cost reporting

Offline route pricing is typically unknown. Incomplete cost may coexist with
release-gate success, but unit-cost fields stay null and cost optimization is
never accepted. The report must state that cost evaluation was not evaluated.
