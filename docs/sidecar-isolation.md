# Sidecar isolation design

**Status:** design + CI compose probe. Not a claim that every customer cluster
is isolated until `ci-isolation` (or live) evidence is recorded for that
cluster.

**Goal:** make bypass structurally impossible. If an engineer can `curl` the
CRM / ledger / mail target from the agent container, you do not have a product.

## 1. Three layers (all required)

```text
┌─────────────────────┐     no route to target IPs
│  Agent / planner    │──────────────────────────────────────┐
│  (model context)    │                                      │
└─────────┬───────────┘                                      │
          │ ActionIntent only                                │ deny
          ▼                                                  ▼
┌─────────────────────┐                           ┌──────────────────┐
│  CBrain runtime     │  decide / mint / claim    │  CRM / ledger /  │
│  (no target creds)  │──────────────────────────▶│  mail targets    │
└─────────┬───────────┘                           └────────▲─────────┘
          │ sole packet path (mTLS / UDS)                  │
          ▼                                                │
┌─────────────────────┐   consume-before-send + witness    │
│  Sidecar process    │────────────────────────────────────┘
│  (creds + dialer)   │
└─────────────────────┘
```

| Layer | What it enforces | Failure mode if missing |
| --- | --- | --- |
| Process | Sidecar is a separate process; agent has no target credentials | Co-located forge of witness |
| Network | Agent namespace cannot route to destination IPs (mesh / NetworkPolicy / eBPF) | Engineer curl = product failure |
| Protocol | Consume-before-send; exact-byte witness; peer pin | Replay / swap wire after ALLOW |

`InProcessDispatchTransport` is **DEV_ONLY**. `DeploymentConfig` with
`environment=production` refuses `dispatch_mode=in_process` at load time.
`require_production_dispatch_transport()` refuses any non-independent transport
at assembly.

## 2. Consume-before-send

Order is load-bearing:

1. PrivateVault `ALLOW` + signed authorization for exact bytes + peer.
2. Sidecar atomically **consumes** the one-use permit (`jti`).
3. Only then dial and transmit `PreparedDispatch.wire_bytes`.
4. Sign witness over observed bytes + peer + outcome.
5. Seal closure; gateway maps proven closure → `EXECUTED`.

If consume fails, bytes never leave. If send may have started without closure,
outcome is `INDETERMINATE` — never auto-retried.

## 3. Network composition (reference)

See `deploy/isolation/`:

- `worker` network: talk to PrivateVault + sidecar only.
- `egress` network: sidecar ↔ write target only.
- Probe: worker cannot reach write target; sidecar can.

CI job `isolation` runs the compose probe. Public claim “network isolated”
requires `ci-isolation` evidence in `docs/claims.md`.

## 4. Computer / desktop use

See `docs/computer-use.md`.

Browser and desktop acts use the same sole-egress rule:

- Model sees **page aliases**, never raw URLs or credentials.
- FoundationAgent tools go through `GovernedRuntime` once via `ComputerToolBridge`.
- `RemoteComputerBackend` talks only to the computer worker (not page URLs).
- Compose probe: `deploy/computer/` — agent ↛ web fixture; agent → worker.
- Screenshots: digest only (`screenshot_sha256`); raw PNG stays in the worker.
- `DevOnlyBrowserBackend` is DEV_ONLY; production must use the remote worker.

## 5. What this design does **not** claim

- Hosted mesh policies for every cloud SKU.
- That TEE / confidential computing is shipped (roadmap item 8).
- Exactly-once delivery to ERP.
