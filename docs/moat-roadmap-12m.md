# 12-month moat roadmap (sequential)

Treat as a sequence. Do not skip isolation for AuthZen theater.

Competitors in view: Zenity / Obsidian Security (enforcement coverage & posture),
Okta / Ping / Entra (IdP PDP the customer already bought), Galileo (eval /
control-plane narrative). We do not claim we already beat them on installed
hooks. We beat them on **meaning of ALLOW** and, later, on auditor-accepted
evidence.

## Months 0–3 — Bypass impossible

| Deliverable | Done when |
| --- | --- |
| In-process transport DEV_ONLY in type system + config | Production config cannot load `dispatch_mode=in_process` |
| Sidecar sole egress compose + CI isolation probe | `ci-isolation` claim pinned; engineer curl from worker fails |
| Consume-before-send adversary tests | Escape suite cannot send without consume |
| “ALLOW ≠ execute” CISO brief | One-pager in customer packet |

**Cut:** in-process as a production option. Domain agents as GTM story.

## Months 3–6 — Identity language enterprises bought

| Deliverable | Done when |
| --- | --- |
| Permit fields: human `sub`, agent `act`, audience, capability, exact-byte hash, expiry, one-use `jti`, optional DPoP/SPIFFE peer | Schema published; decide path emits them |
| AuthZen (or thin facade) in front of `/v1/decide` | Okta/Ping/Azure can sit as extra PDP; PrivateVault remains execution PEP + evidence |
| Five hooks depth, not fifty frameworks | Hermes (have), Claude Code / Anthropic tools, MCP gateway, Salesforce Agentforce, one bank payments/core channel |

**Cut:** zoo of adapters that do not sit on the real wire.

## Months 6–9 — Evidence lock-in + OSS kernel

| Deliverable | Done when |
| --- | --- |
| Publish evidence schema (decision + witness + closure, Merkle-linked) | External team can parse without private docs |
| One independent lab verifies replay | Written attestation that “who allowed this wire” reconstructs |
| Open-source CBrain kernel (contracts, runtime, ports, sidecar protocol) | Public repo; PrivateVault stays closed hosted authority |
| Decision-invariance bench | Same corpus + policy pack: PrivateVault vs LLM-as-judge vs regex/NeMo |

**Win condition:** switching cost is a year of auditor-trusted receipts, not the sidecar binary.

## Months 9–12 — Vertical + confidential sidecar

| Deliverable | Done when |
| --- | --- |
| One regulated payments/lending design partner | Real blocked-transfer story in production path |
| Durable REVIEW + reconcile in that vertical | Parked ≠ “never happened”; 202 ≠ done |
| TEE / GPU-adjacent attested witness (if NVIDIA badge used) | Measured binary + peer attestation; not “Python + TLS” |
| Brand | Category language remains Decision Security; no overclaim vs MS/Zenity coverage |

## Also: personal computer / web for agents

Ship governed computer-use in parallel with months 0–6:

- Desktop/web observe + act so agents complete normal user tasks on webpages.
- Model never sees destinations, credentials, or raw policy.
- Consequential clicks/submits enter `GovernedRuntime`; browser dialer is not in
  the agent network namespace in production.

## Explicit non-goals this year

- Global “most moated agent” marketing.
- Claiming ERP exactly-once.
- Replacing the customer’s IdP.
- Fifty framework logos on the homepage.
