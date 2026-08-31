# Procurement agents

Independent procurement chat bots are **profiles on one `FoundationAgent`**.
Oracle, SAP, and SQL Server never plug into the model. They attach as
deployment-owned adapters on two CBrain paths.

| Path | Role | Must not |
| --- | --- | --- |
| Knowledge | Replica extracts (vendor master, catalog, open POs) quoted as untrusted context | Create a PO, send mail, or authorize spend |
| Governed tools | Named lookups, registered-vendor RFQs, award, PR/PO release | Run unless PrivateVault (production) or the company risk gateway (offline eval) authorizes them |

This slice does **not** ship live JDBC, BAPI/RFC, or SQL Server drivers. Deployment
ETL writes replica JSON; CBrain loads those files from constructor-owned paths.
DSNs, RFC destinations, and passwords never appear in tool schemas, prompts,
proofs, or knowledge metadata.

Replica `registered` flags accept only JSON booleans. Strings such as `"false"`,
integers, `null`, and missing fields fail closed and cannot create a registered
vendor.

## Independent bots

| Bot | Profile | May | Must not |
| --- | --- | --- | --- |
| Buyer | `BUYER_PROFILE` / `CompanyAgentKind.PROCUREMENT` | Lookup, simulated RFQ to **registered vendor IDs**, show quotes, propose award | Choose emails or DSNs; change bank details; post ERP payment; execute award without buyer-lead approval |
| Category manager | `CATEGORY_MANAGER_PROFILE` | Read catalog, vendors, quotation board | Send RFQs, award spend, or change bank/payment tools |
| Vendor onboarding | `VENDOR_ONBOARDING_PROFILE` | Look up registered vendors | Change bank details or post ERP payment (those tools are absent from the profile; the risk gateway still `BLOCK`s them) |

They share `GovernedRuntime`. Spend-sensitive bots set
`knowledge_required_for_tools=True`.

## RFQ mail and instant quotes

`send_rfq_email` is a **simulation-only** handler. It takes `vendor_ids`, never a
raw mailbox or SMTP host. Outputs include `delivery_mode="simulated"` and must
not claim live email delivery, ERP posting, or PrivateVault authorization. The
offline simulator deposits fixture quotations in the same call (`instant: true`).
`show_quotations` ranks by integer minor-unit amount.

The same `rfq_id` with an identical canonical payload returns the prior outcome.
The same `rfq_id` with different bytes is `PROCUREMENT_RFQ_IDEMPOTENCY_CONFLICT`
and is blocked before the handler.

An inbound vendor email that says “award us” is still untrusted context.

## Human-in-the-loop purchasing

`award_quote`, `create_purchase_requisition`, and `release_purchase_order` are
`REVIEW`. The operator loop parks the same `ActionIntent`; a deployment-owned
`buyer_lead` principal may consume that approval **once**.

Offline eval uses `build_procurement_proof`. It recomputes the ActionIntent
digest internally, binds RFQ/quote/vendor/amount/currency/quote-board digest to
the intent and `EXECUTED` output, and always reports
`decision_authority = company_test_gateway`. It never accepts caller-supplied
receipt digests or `action_intent_digest`.

A PrivateVault proof is built only by `build_privatevault_procurement_proof`
from a `VerifiedClosure` produced by the PrivateVault adapter. The builder
re-verifies signatures, trust bundle, expiry, `decision_id`, `request_id`, the
exact action payload, and closure. Digest-shaped strings are not enough.
`BLOCKED`, `REVIEW_REQUIRED`, `CONTROL_FAILURE`, and `INDETERMINATE` never
produce an award, PR, or PO success proof.

## Operator loop

Procurement is an operator-loop kind like Coding, not part of the 200-case
company matrix. Run:

```text
cbrain-eval operator-plan --agent procurement
cbrain-eval operator-run --agent procurement --output-dir operator-loop-output
```

## What this does not claim

- Live Oracle, SAP, or SQL Server connectivity
- Live SMTP delivery or ERP posting
- That a real model will only pick registered vendors
- PrivateVault invariance for `company_test_gateway` fixtures
