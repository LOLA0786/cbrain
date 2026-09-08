# CBrain public claims matrix

Every public sentence about enforcement must cite a pinned test and an
evidence level. Competitors invent “exactly-once”; we refuse to claim it.

Evidence levels:

| Level | Meaning |
|-------|---------|
| `unit` | In-process pytest, no external network authority |
| `real-server` | Pinned PrivateVault `api.server` via TestClient |
| `independent-process` | Separate process / mTLS sidecar path |
| `ci-isolation` | Compose network probe (`CBRAIN_ISOLATION=1`) |
| `live` | Real provider/DB with explicit live gate |
| `reference-target` | Clearly labeled simulator / stub, not ERP |

| Claim | Evidence | Pin |
|-------|----------|-----|
| PrivateVault alone authorizes production dispatch | `real-server` | `tests/test_privatevault_http_contract.py` |
| First-mint action≠wire body cannot authorize | `real-server` | `tests/test_privatevault_http_contract.py::test_first_mint_refuses_action_wire_mismatch_via_direct_http` |
| Caller-selected trust bundle cannot reach target | `unit` | `tests/test_independent_sidecar.py::test_forged_request_trust_bundle_never_reaches_credentials_or_target` |
| Worker without target credentials cannot write | `unit` | `tests/test_isolation_boundary.py::test_worker_without_credentials_cannot_authorize_write` |
| Network isolation worker↛write is proven in CI | `ci-isolation` | `deploy/isolation/probe.sh` + `tests/test_isolation_boundary.py::test_compose_network_isolation_probe_passed` |
| REVIEW lease expiry ≠ “write never happened” | `unit` | `tests/test_durable_review.py::test_lease_expiry_returns_parked_and_is_not_proof_of_non_execution` |
| Competing workers cannot both hold a REVIEW lease | `unit` | `tests/test_durable_review.py::test_competing_workers_cannot_both_hold_lease` |
| Wrong-role approval cannot consume | `unit` | `tests/test_durable_review.py::test_wrong_role_approval_refused_and_consume_once` |
| HTTP 202 is pending acceptance, not completion | `reference-target` | `tests/test_governed_po_idempotency.py::test_http_202_is_pending_acceptance_not_confirmed` |
| Uncertain PO create reconciles by business key; no blind resubmit | `reference-target` | `tests/test_governed_po_idempotency.py` |
| Unapproved REVIEW never mints or executes | `real-server` | `tests/test_privatevault_http_contract.py::test_unapproved_review_cannot_mint_and_never_calls_authorize` |
| We do **not** claim exactly-once purchase delivery | — | Explicit non-claim; INDETERMINATE + reconcile |

## Forbidden language unless evidence exists

- “exactly-once purchase”
- “ERP / Oracle / SAP connected” without a live sandbox pin
- “network isolated” without `ci-isolation` evidence
- “production ready approvals” without durable REVIEW pins
