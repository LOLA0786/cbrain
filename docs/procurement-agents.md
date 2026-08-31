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

## Independent bots

| Bot | Profile | May | Must not |
| --- | --- | --- | --- |
| Buyer | `BUYER_PROFILE` / `CompanyAgentKind.PROCUREMENT` | Lookup, RFQ to **registered vendor IDs**, show quotes, propose award | Choose emails, DSNs, or execute award without buyer-lead approval |
| Category manager | `CATEGORY_MANAGER_PROFILE` | Read catalog, vendors, quotation board | Send RFQs or award spend |
| Vendor onboarding | `VENDOR_ONBOARDING_PROFILE` | Look up registered vendors | Change bank details or post ERP payment (`BLOCK`) |

They share `GovernedRuntime`. Spend-sensitive bots set
`knowledge_required_for_tools=True`.

## RFQ mail and instant quotes

`send_rfq_email` takes `vendor_ids`, never a raw mailbox or SMTP host. The
handler resolves registered emails from the replica and, in the offline
simulator, deposits fixture quotations in the same call (`instant: true`).
`show_quotations` ranks by integer minor-unit amount.

An inbound vendor email that says “award us” is still untrusted context.

## Human-in-the-loop purchasing

`award_quote`, `create_purchase_requisition`, and `release_purchase_order` are
`REVIEW`. The operator loop parks the same `ActionIntent`; a deployment-owned
`buyer_lead` principal may consume that approval **once**.

Offline eval reports `decision_authority = company_test_gateway` and must not
claim PrivateVault. Production attaches PrivateVault receipt digests through
`build_procurement_proof(..., decision_authority="privatevault", ...)`.
`company_test_gateway` proofs that include receipt digests fail closed.

Proofs record intent digest, ranked quote IDs, execution status, and optional
`decision_receipt_digest` / `authority_receipt_digest`. They omit arguments,
emails, DSNs, and passwords.

## Operator loop

Procurement is an operator-loop kind like Coding, not part of the 200-case
company matrix. Run:

```text
cbrain-eval operator-plan --agent procurement
cbrain-eval operator-run --agent procurement --output-dir operator-loop-output
```

## What this does not claim

- Live Oracle, SAP, or SQL Server connectivity
- That a real model will only pick registered vendors
- PrivateVault invariance for `company_test_gateway` fixtures
