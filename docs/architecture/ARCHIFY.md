# Using Archify with CBrain

[Archify](https://github.com/tt-a1i/archify) is an **agent skill for verifiable
architecture diagrams**, not an authorization component. Use it to present the
moat topology; do not treat a diagram as isolation proof.

## Install (Cursor)

```bash
npx -y skills add tt-a1i/archify --skill archify --agent cursor --global --copy --yes
```

## CBrain artifact

Typed IR for the isolation story:

`docs/architecture/moat-isolation.architecture.json`

Ask the agent (with Archify installed) to render that IR to a self-contained
HTML map for reviews and customer pitches.

## What still proves the claim

| Claim | Proof |
|-------|--------|
| Worker cannot reach write | `deploy/isolation/probe.sh` under compose + CI `isolation` job |
| Credentials stay in sidecar | `tests/test_isolation_boundary.py` |
| Trust root is deployment-owned | `tests/test_independent_sidecar.py` |

A beautiful diagram without those pins is marketing, not a moat.
