# Company agents — offline evaluation (v0.4)

CBrain ships four **company agents** as configuration profiles over one shared
`FoundationAgent` runtime. There is no separate reasoning loop per agent.

| Agent | Profile ID | Simulator |
| --- | --- | --- |
| GTM | `company-gtm-v0.4` | Simulated CRM |
| Operations | `company-operations-v0.4` | Tickets, runbooks, service health |
| Legal | `company-legal-v0.4` | Supplied contract fixtures |
| Accounts | `company-accounts-v0.4` | Invoices and ledger |

## Architecture

```text
AgentProfile + ToolRegistry + CompanyRiskGateway
        ↓
FoundationAgent (single shared loop)
        ↓
ActionIntent → ALLOW | REVIEW | BLOCK
        ↓
Offline simulator handlers (no production APIs)
```

Authorization policy lives in tool risk classification (`cbrain/company/risk.py`),
not in model prompts.

## Scope (offline only)

- No Gmail, production CRM, Kubernetes, legal filing, or banking rails
- No credentials in tool schemas or simulator fixtures
- `REVIEW` and `BLOCK` never execute handlers
- Legal output requires licensed lawyer review
- Accounts scenarios do not execute real payments

## Scenario suites

Each agent has **50** deterministic scenarios:

| Category | Count |
| --- | ---: |
| Normal | 20 |
| Edge | 10 |
| Adversarial | 10 |
| Authorization | 5 |
| Crash / concurrency / cost | 5 |

**200** scenarios total (`company-offline-v0.4`).

## Commands

```bash
# Read-only plan
uv run cbrain-eval company-plan --agent all
uv run cbrain-eval company-plan --agent gtm

# Run suites and write artifacts
uv run cbrain-eval company-run --agent all --output-dir /tmp/company-eval

# Quick demo table
uv run python examples/company_offline_demo.py
```

Artifacts: `runs.jsonl`, `aggregate_report.json`, `comparison.csv`, `summary.md`, `manifest.json`.

## Release gates (offline fixtures)

- Unauthorized executions: 0
- Approval bypasses: 0
- Duplicate dispatches: 0
- Safety violations: 0
- Crash recovery: 100%
- Tool selection / argument accuracy: 100%
- Legal / accounts source grounding: 100%

## Adding another company agent

1. Add tools and risk map in `cbrain/company/tools.py` and `risk.py`
2. Add an `AgentProfile` in `cbrain/company/profiles.py`
3. Extend simulators and handlers
4. Add 50 scenarios in `cbrain/evaluation/company_suites.py`
5. Extend CLI `--agent` choices and tests

PrivateVault remains optional; do not import `privatevault-agent-dna` into core.

## What results prove (and do not prove)

**Prove (offline, deterministic fixtures):**

- Tool allowlists and risk boundaries per profile
- Governance: ALLOW / REVIEW / BLOCK behavior on scripted proposals
- Simulator state transitions and crash/resume idempotency
- For identical ActionIntent inputs, authorization decision divergence was zero across evaluated route labels

**Do not prove:**

- Live model quality on real prompts
- Production CRM, email, infrastructure, legal, or payment execution
- That models behave identically across providers
