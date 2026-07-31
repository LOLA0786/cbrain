 # CBrain — Governed Runtime for Production AI Agents

CBrain is the execution-control layer between an AI agent’s plan and the systems it can affect.

It normalizes tool calls from agent frameworks, sends consequential actions to PrivateVault for an authoritative decision, and prevents execution unless the complete control path is available.

> **Status:** Production-runtime alpha. The governed runtime, PrivateVault
> decision and evidence adapters, atomic authorization consumption, exact-byte
> gateway, and independent sidecar transport are implemented. Production still
> requires the sidecar composition root, container isolation, and network policy
> that makes the sidecar the agent's only egress route.

## Architecture

```mermaid
flowchart TD
    A["Hermes or framework agent"] --> H["Framework adapter"]
    H --> C["CBrain ActionIntent"]
    C --> P["PrivateVault decision"]
    P -->|BLOCK, REVIEW, failure| B["Blocked tool result"]
    P -->|ALLOW| E["Signed execution permit"]
    E --> D["Independent sole-egress sidecar"]
    D --> W["Exact-byte send, witness, closure"]
    G["GBrain memory and skills"] --> H
```

A PrivateVault `ALLOW` decision is necessary, but it is not treated as sufficient execution authority.

## Responsibilities

| Component | Responsibility | Authority |
| --- | --- | --- |
| Hermes Agent | Planning, reasoning, and tool orchestration | No execution authority |
| GBrain | Memory, retrieval, graph knowledge, and skills | No execution authority |
| CBrain | Normalization, runtime invariants, routing, and fail-closed control | Enforces control availability |
| PrivateVault Agent DNA | Policy, authority, approval, decisions, and evidence | Sole decision authority |
| Dispatcher sidecar | Permit consumption, credentials, exact-byte egress, witness, and closure | Execution authority only under a valid signed permit |
| RunPod/model providers | Model inference | No tool or secret authority |

## Implemented

Current integration branch:

```text
feature/dispatcher-sidecar-v0.1
```

### Governed Runtime Kernel

- Immutable `ActionIntent` capture
- Stable request and idempotency identities
- Immutable `GovernedExecution` results
- Runtime states:
  - `EXECUTED`
  - `BLOCKED`
  - `REVIEW_REQUIRED`
  - `CONTROL_FAILURE`
  - `INDETERMINATE`
- At-most-once handler entry
- No automatic retry after execution may have started
- Fail-closed control-plane behavior

### PrivateVault Decision Adapter

- Uses the real PrivateVault `/v1/decide` contract
- Supports:
  - `allow`
  - `require_approval`
  - `block`
- Strict status mapping:
  - `200` → `allow`
  - `202` → `require_approval`
  - `403` → `block`
- Rejects malformed, contradictory, redirected, oversized, and non-JSON responses
- HTTPS required except explicit localhost development
- No automatic retries
- Uses PrivateVault’s real `X-API-Key` authentication
- Requires a full-scope key
- PrivateVault validates key scope and registry identity server-side

### Exact-Byte Dispatch Contract

`PreparedDispatch` binds execution to:

- Request identity
- Transport and destination
- Operation
- Exact outbound bytes
- Peer identity bytes
- Content type and encoding
- Tool identity
- Tool schema and artifact digests
- Credential audience
- Idempotency digest
- Retry-policy digest

An arbitrary Python callback is not considered proof of the bytes transmitted over the wire.

### Independent Dispatcher Sidecar

- Runs outside the agent process and declares an independent witness identity
- Receives no model-provider or target credentials from the agent
- Resolves credentials from a signed `credential_audience` inside the sidecar
- Uses an explicit destination and operation allow-list; model-provided URLs are
  never dialed directly
- Opens and verifies TLS before consuming the one-use permit
- Compares the observed certificate digest with the peer identity in the permit
- Atomically consumes the authorization immediately before sending
- Sends the same immutable body bytes verified by Agent DNA
- Signs the dispatch witness with the bytes and peer identity it observed
- Signs closure at the dispatch boundary and returns the complete chain
- Classifies refusal before send as `CONTROL_FAILURE`
- Classifies any failure after send may have started as non-retryable
  `INDETERMINATE`
- Provides a strict HTTPS/mTLS client and TLS server adapter

The existing in-process transport remains available for local conformance tests
and explicitly cannot claim witness independence.

### Hermes Integration

Built against the real NousResearch Hermes plugin system.

- Official `pre_tool_call` hook
- Official `plugin.yaml` and `register(ctx)` packaging
- Unknown tools fail closed
- Missing session or tool-call identity fails closed
- PrivateVault failures become valid Hermes block directives
- Deterministic request IDs
- Mandatory `cbrain-hermes` launcher
- Startup refused unless:
  - `cbrain_guard` is loaded
  - CBrain owns the first pre-tool callback
  - The expected callback implementation is registered
- Rejects bypass flags including:
  - `--safe-mode`
  - `--ignore-rules`
  - `--ignore-user-config`
  - `--yolo`
- Plugin administration is blocked through the production launcher

Hermes documents hook and middleware failures as fail-open. CBrain catches control failures inside the hook and returns a valid blocking directive. The mandatory launcher prevents silent startup without enforcement.

### GBrain Integration

Built against the real Garry Tan GBrain repository and MCP server.

- Pinned version: `0.42.67.0`
- Real stdio MCP command: `gbrain serve`
- Hermes discovered 106 real GBrain tools
- Only 15 read/skill tools are exposed:
  - `get_page`
  - `list_pages`
  - `search`
  - `query`
  - `get_tags`
  - `get_links`
  - `get_backlinks`
  - `traverse_graph`
  - `get_timeline`
  - `get_stats`
  - `get_health`
  - `get_brain_identity`
  - `list_skills`
  - `get_skill`
  - `resolve_slugs`
- Writes, deletes, purges, uploads, jobs, source mutation, and admin operations are denied by default
- `GBRAIN_HOME` must be an absolute dedicated path
- Parent database environment variables are cleared for isolated PGLite operation
- API keys, authority grants, approvals, and signing material must never enter GBrain

Local verification:

```text
PGLite schema migrated through version 125
52/52 GBrain skills conformant
106 MCP tools discovered
15 tools selected
```

## Current Enforcement State

| PrivateVault result | Current behavior |
| --- | --- |
| `block` | Tool call blocked |
| `require_approval` | Tool call blocked with review reference |
| `allow` | Executed only when a configured gateway completes authorization, single-use consumption, dispatch, witness, and closure; the Hermes pre-tool hook remains blocked until that gateway composition is installed |
| Timeout or authentication failure | Tool call blocked |
| Unknown tool | Tool call blocked |
| Malformed response | Tool call blocked |
| Adapter failure | Tool call blocked |

A decision receipt is not treated as execution authorization.

## Upstream Pins

Exact upstream identities are stored in [`upstreams.lock.json`](upstreams.lock.json).

| Component | Repository | Pinned commit |
| --- | --- | --- |
| Hermes Agent | `NousResearch/hermes-agent` | `f3cda0ceb18d8ba7465a6d223098ef0e56c8fee1` |
| GBrain | `garrytan/gbrain` | `c6dc0adf26a2d20df1147d2ec87c8922ca86d410` |
| PrivateVault Agent DNA | `LOLA0786/privatevault-agent-dna` | `eabc02e806fe4804d7422556af3d1b376742ccfc` |

Upstream changes require review and conformance testing before these pins are updated.

## Framework Dependencies

The reproducible dependency graph includes optional packages for:

- LangChain
- LangChain OpenAI
- CrewAI
- Microsoft AutoGen Core
- Microsoft AutoGen AgentChat
- Microsoft AutoGen OpenAI extensions

These packages are locked in `uv.lock`.

The shared fail-closed framework guard is implemented. Native LangChain,
CrewAI, and AutoGen execution wrappers remain to be connected to the concrete
gateway; they will translate calls and never contain separate policy logic.

## Secrets

Production invariants:

- Model-provider keys never enter prompts
- Secrets never enter GBrain
- Secrets never enter tool arguments or decision reasons
- PrivateVault credentials are read from an absolute mounted secret file
- Credentials are loaded at request time to support rotation
- Raw credentials are never committed
- Production will use workload identity, a secret manager, or a credential broker
- Execution, witness, and closure signing keys remain outside the agent process

## Installation

Python 3.12 or newer is required.

```bash
git clone git@github.com:LOLA0786/cbrain.git
cd cbrain
git checkout main

uv sync --extra dev
```

Optional dependency groups:

```bash
uv sync --extra frameworks
uv sync --extra privatevault
```

Do not install the heavy framework group in Google Cloud Shell. Build it through the remote container pipeline.

## Verification

```bash
uv run ruff check cbrain integrations tests
uv run mypy cbrain
uv run pytest -q
uv lock --check
git diff --check
```

Current verification:

```text
131 default tests passed
7 additional real Agent DNA conformance tests passed in pinned CI
138 total tests with the PrivateVault dependency enabled
Ruff clean
Production-source mypy clean
Dependency lock consistent
```

## Project Layout

```text
cbrain/
├── adapters/
│   ├── framework.py
│   ├── gbrain.py
│   ├── hermes.py
│   ├── privatevault.py
│   ├── privatevault_claim.py
│   ├── privatevault_consumption.py
│   ├── privatevault_execution.py
│   └── privatevault_http.py
├── execution/
│   ├── gateway.py
│   ├── sidecar.py
│   └── transport.py
├── consumption.py
├── contracts.py
├── dispatch.py
├── hermes_launcher.py
├── ports.py
└── runtime.py

integrations/
└── hermes/
    └── cbrain_guard/

tests/
migrations/postgres/
upstreams.lock.json
uv.lock
pyproject.toml
AGENTS.md
```

## Security Invariants

1. Every consequential action is normalized before execution.
2. Framework adapters translate; they do not authorize.
3. PrivateVault is the sole decision authority.
4. BLOCK, REVIEW, unknown, malformed, and unavailable states do not execute.
5. Tool execution is at most once.
6. Indeterminate execution is never automatically retried.
7. Unknown tools are denied by default.
8. Secrets never enter model-visible memory or arguments.
9. Upstream implementations are pinned.
10. An `ALLOW` decision does not replace signed execution authorization.
11. Outbound bytes and peer identity must match the signed permit.
12. Execution requires authorization, a dispatch witness, and closure evidence.

## Remaining Work

1. Add the sidecar composition root and production configuration schema
2. Build separate agent, control-plane, sidecar, CRM simulator, and ledger
   simulator containers
3. Enforce network policy so the agent can reach only PrivateVault and the
   sidecar, while only the sidecar can reach target systems
4. Add workload identity, mTLS certificate rotation, and secret-manager-backed
   credential providers
5. Build the CRM and finance capability taxonomy and executable simulators
6. Add the benign, adversarial, multi-step, replay, and failure-injection
   scenario harness
7. Add CBrain `SKILL.md` packages and resolver policy; GBrain's external skills
   are not currently committed in this repository
8. Add native LangChain, CrewAI, and AutoGen execution wrappers
9. Add Anthropic, OpenAI, xAI, Google, and RunPod/OpenAI-compatible model routing
10. Run the cross-model matrix and report gate-decision divergence separately
    from model activation frequency
11. Add OpenTelemetry, metrics, structured audit export, signed containers,
    SBOMs, load tests, and deployment runbooks

## Positioning

- **Hermes:** reasoning and agent runtime
- **GBrain:** memory and skills
- **CBrain:** governed execution runtime
- **PrivateVault:** authorization, enforcement, and verifiable evidence

CBrain’s purpose is not to make an agent appear safe.

Its purpose is to make unauthorized or unverifiable execution impossible.

---

Built by Chandan Galani.
