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
uses WAL mode on writable opens. Records are immutable and content-addressed.

Each record has two digests:

- a full record hash, including timestamps, used as `signal_id` /
  `approval_id` and `content_hash`
- a semantic idempotency digest that excludes only the permitted retry
  timestamp

Stores validate records on append and on load. Stored JSON must be the
canonical serialization of the exact schema-key allowlist. Missing keys,
unknown keys (including `prompt_text` and `Prompt`), nested extras,
duplicate JSON keys, and column/payload mismatches fail closed.

Idempotent retries compare semantic digests and return the originally
stored record. Timestamp edits to an existing row fail integrity checks.
SQLite appends are conflict-safe: a racing insert reloads the stored row
and accepts it only when the semantic digest matches.

The on-disk schema version is `2`. Opening a v1 store fails closed and
never silently reinterprets old hashes. Operators who need a rewrite must
call `migrate_learning_store_v1_to_v2` explicitly; v1 approvals are not
migrated and must be re-approved.

Evidence is loaded by exact IDs; missing IDs fail closed.

## Human approval

`PersonalizationManager` requires a deployment-owned `ReviewerVerifier`.
If no verifier is configured, construction fails closed. Callers present a
`ReviewerPrincipal` (reviewer id plus a non-secret attestation). The
verifier attests identity; the manager then re-checks that the principal
is still authorized. Test doubles belong only under `tests/`. Authentication
secrets never enter signals, stored records, logs, or model context.

Approval time comes from an injected trusted wall clock owned by the
manager. The production `approve()` API does not accept `approved_at`.
Non-finite clocks and regressions behind already-stored timestamps fail
closed. Candidate freshness is evaluated at that trusted time.

Approval:

1. Verifies the reviewer principal through the configured port.
2. Reconstructs a trusted snapshot from storage at the trusted clock.
3. Reloads the cited human evidence by exact ID.
4. Recomputes the candidate and rejects fabricated, changed, or stale input.
5. Binds the approval to the exact base `profile_fingerprint`.

`compile_profile()` never trusts an approval merely because it is present
and hash-consistent. Every stored approval for the agent is revalidated
(reviewer still authorized, evidence still present and human, candidate
reconstructs, agent id matches) before any instruction text is appended.
Invalid approval data fails closed instead of being skipped.

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
to the last 30 days. Reporting a missing store returns exit code 2 and
creates no file. An existing store is opened with a read-only / query-only
connection; report mode never creates schema, migrates, or sets WAL. Day
and occurrence values must be positive. HTML escapes visible text and
safely encodes embedded JSON. No command approves or activates guidance.

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
