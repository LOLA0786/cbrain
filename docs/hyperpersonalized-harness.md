# Hyperpersonalized insights harness

CBrain's insights harness is the governed equivalent of:

`session history → /insights → reviewed rules/skills → personalized agent profile`

It is not model self-training and it is never an authorization mechanism.
Personalization may improve instructions and optional skills. PrivateVault
remains the sole authority for consequential actions.

## Boundary

| Layer | May | Must not |
| --- | --- | --- |
| Human feedback | Record sanitized corrections, preferences, workflows | Carry secrets, prompts, arguments, or outputs |
| Runtime observation | Record agent ID, run ID, typed status, tool name, rejection status | Store task text, reasons, credentials, or payloads |
| Insights engine | Propose findings from repeated evidence | Approve, activate, or apply guidance |
| Personalization manager | Accept allowlisted human approval bound to a profile fingerprint | Trust caller-supplied snapshots or fabricated evidence |
| Compiled profile | Append approved rules and explicitly activated skills | Change identity, route, tools, limits, or policy |
| Foundation agent | Execute the compiled instructions through `GovernedRuntime` | Read the learning store from inside the run loop |

The model, session history, Insights engine, and personalization layer are
proposal-only. Every consequential `ActionIntent` still enters
`GovernedRuntime` and the PrivateVault authorization/dispatch/evidence path.

## Learning signals

Signals are immutable, versioned, and content-addressed.

Human signals:

- correction
- preference
- workflow instruction

Runtime signals:

- structural run outcome
- governed tool rejection

Only explicit human feedback can become a rule or skill candidate. Runtime
failures and tool rejections may identify friction. They never become
instructions automatically.

Ingest rejects credential-shaped text (private keys, API keys, JWTs,
password/secret/token assignments, bearer tokens). Status fields accept only
defined `RunStatus` and `ExecutionStatus` values.

## Insights engine

The engine is deterministic and read-only. It uses a configurable window
(default 30 days) and requires at least two matching signals before it
proposes guidance.

It can emit:

- repeated-correction findings
- repeated-preference findings
- repeated-workflow findings
- run-friction findings
- tool-friction findings

Rule and skill candidates are produced only from repeated human evidence.
Each finding includes evidence IDs, an occurrence count, and a deterministic
hash. The engine never writes approvals or compiled profiles.

## Storage

`InMemoryLearningStore` and `SQLiteLearningStore` are append-only. SQLite
uses WAL mode. Records are immutable and content-addressed. Stored columns
are revalidated against the serialized payload. Corrupt or inconsistent
rows fail closed. Idempotent retries may differ only in the observation
timestamp. Evidence is loaded by exact IDs; missing IDs fail closed.

## Human approval

`PersonalizationManager` requires a non-empty allowlist. Only those humans
may approve. Approval:

1. Reconstructs a trusted snapshot from storage.
2. Reloads the cited human evidence by exact ID.
3. Recomputes the candidate and rejects fabricated, changed, or stale input.
4. Binds the approval to the exact base `profile_fingerprint`.

Approvals are immutable and retry-idempotent. Caller-supplied snapshot
objects are never trusted as source data.

## Profile compilation

Personalization may modify only:

- agent instructions
- non-authoritative metadata

It must not modify agent identity, model route, permitted tools, execution
limits, turn or tool-call limits, timeouts, PrivateVault policy, approval
requirements, or action/dispatch contracts.

Standing approved rules are appended to instructions. Approved skills still
require explicit activation through an `active_skills` allowlist. The model
cannot choose or activate skills. Unknown or unapproved skills fail closed.

The compiled profile receives a new fingerprint because its instructions
changed. An in-progress durable run cannot silently switch personalization.

## CLI

```text
cbrain-insights feedback --store PATH --kind correction --agent-id ID \
    --reviewer-id HUMAN --text TEXT --idempotency-key KEY

cbrain-insights report --store PATH [--agent-id ID] [--days 30] \
    [--min-occurrences 2] [--format json|html] [--output FILE]
```

`feedback` records sanitized human input. `report` is read-only and defaults
to the last 30 days. Day and occurrence values must be positive. HTML
escapes visible text and safely encodes embedded JSON. No command approves
or activates guidance.

## Foundation agent integration

Do not inject the learning store into `FoundationAgent.run()`. Observe
afterward:

```python
from cbrain.agent import LearningRecorder

result = agent.run(run_input)
LearningRecorder(store).record_run(
    agent_id=profile.agent_id,
    result=result,
)
```

If the learning store fails, `result` is unchanged.
