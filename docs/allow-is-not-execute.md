# ALLOW is not execute

**Audience:** bank / payments CISO, control owner, internal audit  
**One page.** Decision Security for agents that can move money.

## The mistake most vendors sell

An LLM (or “guardrail”) says the tool call looks fine. The agent then dials the
payments API with a bearer token sitting in the planner process. That is
**advice**, not **authorization**. Advice is deniable. Unauthorized wires are not.

## What “ALLOW” means here

PrivateVault may return ALLOW. That is **necessary, not sufficient**.

Before any consequential byte leaves toward CRM, ledger, mail, or core banking:

1. **Human sub + agent act** — who is responsible, which agent is acting.
2. **Audience + capability** — this tool, this destination class, not “the model felt like it.”
3. **Exact-byte hash** — the wire body that will be sent is sealed; swap quantity → refuse.
4. **Expiry + one-use `jti`** — permits are not reusable souvenirs.
5. **Consume-before-send** — the permit is burned in the sidecar **before** dial.
6. **Witness + closure** — independent process attests what peer and bytes were observed.
7. **Network isolation** — the agent namespace cannot reach destination IPs. If an
   engineer can `curl` the target from the agent container, you do not have a product.

`BLOCKED`, `REVIEW_REQUIRED`, and `CONTROL_FAILURE` never invoke the tool.
Possible send without proven closure is `INDETERMINATE` and is never auto-retried.

## What we do **not** claim (yet)

- Broader installed-hook coverage than Microsoft / Zenity across every IDE and SaaS.
- Exactly-once purchase posting into your ERP.
- That our policy language replaces Okta / Ping / Azure AD.

We claim a sharper sentence: **ALLOW is not execute.** Execution integrity is a
separate plane — PEP + evidence — that IdP PDPs can sit beside, not replace.

## What you should demand in a pilot

| Ask | Pass criterion |
| --- | --- |
| Bypass | Agent pod cannot reach payment host; only sidecar can |
| Swap | Change wire after decide → authorize refuses |
| Review | Unapproved REVIEW never mints, never dials |
| Evidence | Replay “who allowed this wire” from sealed records without trusting the vendor’s UI |
| Computer use | Agent can drive a portal page; model never sees URL, password, or raw policy |

## The winnable category

Not “most autonomous agent.”  
**First execution-integrity plane a regulated payments or lending stack will put
in front of an agent that can move money.**

One design partner with a real blocked-transfer story beats ten procurement demos.
