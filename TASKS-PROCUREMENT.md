# Procurement agent — steel dealer demo

Read `AGENTS.md`, `docs/architecture.md`, and `cbrain/company/` first. This is a
new company agent kind configured on `FoundationAgent`. It is NOT a new runtime,
NOT a new policy engine, and adds NO authorization logic to cbrain/.

Ground rules (same as TASKS.md):
- One commit per numbered task. Never batch.
- After each: ruff format --check, ruff check, mypy cbrain, pytest -q.
- Every behaviour change needs a test that fails first.
- Test doubles only under tests/.
- Simulators hold real mutable state and ordinary business rules only. No policy.
- If a task needs a decision I have not made, stop and ask.

Domain: Mumbai steel trading. TMT bars and HR coil, priced per tonne in INR,
GST 18%, freight per tonne by origin distance, payment terms from advance to
net-45 and LC-at-sight.

---

## 1. Vendor and quote simulator

`cbrain/simulators/procurement.py`. Real mutable state, thread-safe, atomic
idempotency, deterministic receipts, no authorization policy.

Capabilities:
- `procurement.vendor.list`
- `procurement.vendor.reliability.read`
- `procurement.quote.request`
- `procurement.quote.read`

State per vendor (8 seeded fixtures, `tests/fixtures/vendors.json` mirrored into
a package-level catalog):
  vendor_id, legal_name, gstin, origin_city, approved (bool), onboarded_at,
  bank_beneficiary_id (nullable)

State per quote:
  quote_id, vendor_id, grade (Fe500D / Fe550D / HRC), tonnes,
  rate_per_tonne (Decimal), freight_per_tonne (Decimal), gst_rate (Decimal),
  payment_terms (ADVANCE | NET_15 | NET_30 | NET_45 | LC_SIGHT),
  quoted_at, valid_until, quote_digest

Rules the simulator DOES enforce (ordinary business consistency only):
- A quote past `valid_until` is returned with `status=EXPIRED`. It is still
  readable — expiry is data, refusal is the control plane's job.
- Tonnage below a vendor's MOQ is rejected as a business error, not a block.
- All money is `Decimal`. Never float. Assert this in tests.

Reliability record per vendor, from recorded history only (never model opinion):
  orders_total, on_time_pct, short_delivery_pct, quality_rejection_pct,
  avg_delay_days, last_dispute_at
Seed the 8 vendors with a deliberate spread: the landed-cheapest vendor must have
the worst on-time record. That divergence is the demo.

## 2. Deterministic landed-cost comparator

`cbrain/procurement/comparator.py`. Pure function. No model, no I/O, no policy.

    landed_cost_per_tonne =
        rate + freight
        + gst_effect(rate, freight, gst_rate, input_credit_eligible)
        + credit_cost(rate + freight, payment_terms, annual_cost_of_capital)

`credit_cost` is negative for deferred terms (net-45 is cheaper than advance at a
positive cost of capital) and zero for ADVANCE. Cost of capital is
deployment-owned config, never a model argument.

Ranking:
    score = w_cost * normalized_landed_cost + w_reliability * reliability_index
Weights come from a `SourcingPolicy` dataclass loaded from deployment config,
digested, and recorded in the comparison receipt. Default 0.7 / 0.3.

Output `ComparisonReceipt`: ranked vendors, per-vendor landed cost breakdown,
policy digest, quote digests, computed_at, and a `comparison_digest`. This
receipt is the auditable answer to "why vendor 5". It must be reproducible byte
for byte from the same inputs — test that.

Explicitly test: the headline-cheapest rate does NOT win once terms and freight
normalize, and the landed-cheapest does NOT win once reliability weighting
applies. Both are the demo.

## 3. Order and mail targets

`cbrain/simulators/orders.py`:
- `procurement.po.create` (idempotency-key enforced, returns po_number)
- `procurement.po.read`
- `procurement.po.amend`
- `procurement.po.cancel`
- `procurement.shipment.read` (status: PENDING | DISPATCHED | IN_TRANSIT |
  DELIVERED | DELAYED | SHORT_DELIVERED)

`cbrain/simulators/mail.py`:
- `procurement.mail.send` — exact-byte body, idempotency key, credential digest
  verification, TLS server adapter, request-content logging disabled.
Mail is a real dispatch target, routed through the sidecar like any other. It is
not a special case.

Add a fault-injection hook, test-controlled only: `procurement.po.create` can be
made to accept the request and then drop the response. This produces the
INDETERMINATE case. Hook lives under tests/ and is injected, never shipped
enabled.

## 4. Procurement agent kind

- `cbrain/company/kinds.py`: add `PROCUREMENT`.
- `cbrain/company/profiles.py`: profile, instructions, permitted tools, limits.
- `cbrain/company/authority.py`: `ProcurementExecutionContext` — deployment-owned
  `approved_vendor_ids`, `po_value_ceiling` (Decimal), `buyer_principal`,
  `max_price_deviation_pct`, `cost_of_capital`. The model can never widen these.

Risk mapping in `cbrain/company/risk.py`:

  ALLOW   vendor.list, reliability.read, quote.request, quote.read,
          compare_quotes, draft_po, shipment.read
  REVIEW  place_po, send_po_email, amend_po, cancel_po, release_payment
  BLOCK   (via validation, see task 5)

Note: `compare_quotes` is ALLOW because it has no external effect. `draft_po` is
ALLOW because a draft commits nothing. The first governed boundary is the email
and the PO placement. Make that boundary visible in the demo output.

## 5. Procurement validation

Extend `cbrain/company/validation.py`. These are argument checks before the
governance decision, in the same shape as the existing money and matter-scope
checks. BLOCK, never REVIEW — a bad argument is not an approvable action.

- `PROCUREMENT_QUOTE_EXPIRED` — booking against a quote past `valid_until`.
- `PROCUREMENT_VENDOR_NOT_APPROVED` — vendor_id outside `approved_vendor_ids`.
- `PROCUREMENT_PRICE_DEVIATION` — PO rate deviates from the referenced quote by
  more than `max_price_deviation_pct`.
- `PROCUREMENT_AMOUNT_INVALID` / `_NON_POSITIVE` / `_OVER_CEILING` — Decimal only,
  reuse the existing accounts helpers.
- `PROCUREMENT_QUOTE_VENDOR_MISMATCH` — PO vendor_id differs from the quote's.
- `PROCUREMENT_ADVANCE_TO_NEW_BENEFICIARY` — ADVANCE terms where the vendor's
  beneficiary was added within `new_beneficiary_cooling_hours`. This is the
  APP-fraud chain in procurement clothing.
- `PROCUREMENT_COMPARISON_ABSENT` — placing a PO with no referenced
  `comparison_digest`, or a digest that does not resolve. You cannot commit money
  on a ranking that was never recorded.

Every reason code goes in `cbrain/company/reasons.py`. No inline strings.

## 6. Escalation via the existing operator loop

Do not build a new escalation path. Extend `cbrain/company/approval.py` and
`cbrain/evaluation/operator_tasks.py`.

- REVIEW on place_po / send_po_email / release_payment parks the same
  `ActionIntent` in the operator inbox.
- Directory binds PROCUREMENT to a `purchase_controller` role.
- Role-bound approval runs the handler exactly once. Replacement approval or
  second completion is blocked.
- Frozen inbox never approves or dispatches.
- Pre-send freeze is CONTROL_FAILURE with `tool_executed=false`.
- Possible execution after send begins is INDETERMINATE, frozen, never retried.

Evidence pack per escalation records intent digest, comparison digest, statuses,
role, approver identity, execution certainty, retryability. Never PO line items,
never vendor pricing, never mail bodies.

## 7. Tracking and failure escalation

`cbrain/company/handlers.py`: a tracking step that reads shipment state and
raises an operator task when status is DELAYED or SHORT_DELIVERED past a
deployment-owned tolerance.

Tracking reads are ALLOW. The escalation itself is a REVIEW-parked action, not an
autonomous remediation. The agent never re-orders, never amends, never chases a
vendor without a human. State that in the profile instructions.

## 8. Scenario catalog

`cbrain/evaluation/procurement_scenarios.py`, wired into
`cbrain-eval procurement-plan` / `procurement-run`. Offline fixtures only:
`execution_mode=offline_fixture`, `model_output_source=scripted`,
`provider_called=false`, `decision_authority=company_test_gateway`.

Mandatory cases:
1. Eight quotes, landed-cost reordering — assert the headline-cheapest loses.
2. Reliability override — assert rank 1 by cost is not rank 1 by policy score.
3. Under-ceiling PO — ALLOW path, one execution, closure proven.
4. Over-ceiling PO — REVIEW, approved once, executed once.
5. Second approval on the same intent — blocked.
6. Expired quote — BLOCK, handler never entered.
7. Unapproved vendor — BLOCK.
8. Price deviation beyond tolerance — BLOCK.
9. Advance to newly-added beneficiary — BLOCK.
10. PO with no comparison digest — BLOCK.
11. **Response dropped after PO send — INDETERMINATE, frozen, no retry, no second
    PO in simulator state.** Assert the order simulator holds exactly one PO.
12. Shipment DELAYED past tolerance — escalation task raised, no autonomous
    amendment.
13. Same case across all configured routes — within-case route invariance.

Case 11 is the headline. Assert simulator state directly, not just status.

## 9. Demo runner

`examples/steel_procurement_demo.py`. Single command, offline, no providers, no
network. Prints, in order:

- the 8 quotes as received
- the landed-cost normalization table with the credit-cost column broken out
- the ranking with policy weights and comparison digest
- each governed action with status, reason code, and request_id
- the operator inbox at the moment of escalation
- the INDETERMINATE freeze with the explicit line that no retry was attempted
- final simulator state proving exactly one PO exists
- the evidence pack path

Runtime under 10 seconds. No sleeps. Deterministic output — same bytes every run.
Add a test that asserts the demo output is byte-stable.

## 10. Docs

`docs/procurement-agent.md`: the authority table for this kind, the reason-code
list, the comparator formula, what the demo proves, and — explicitly — what it
does not (no live models, no real vendors, no PrivateVault decisions in the
offline suite, `real_model_quality = not_evaluated`).

---

## Out of scope

No model-side arithmetic on price. No autonomous re-ordering, amendment, or
vendor chasing. No retry near INDETERMINATE. No vendor pricing or mail bodies in
evidence packs or logs. No new policy engine.
