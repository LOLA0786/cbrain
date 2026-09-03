# CBrain remediation + moat

Repo: cbrain. Read `AGENTS.md` and `docs/architecture.md` first. Non-negotiable
invariants there override anything below. Do NOT modify `cbrain/contracts.py`,
`cbrain/runtime.py`, or `cbrain/ports.py` unless a task says to.

Ground rules:
- One commit per numbered task. Never batch.
- After every task: `ruff format --check`, `ruff check cbrain integrations tests`,
  `mypy cbrain`, `pytest -q`. All must pass before the next task.
- Every behaviour change needs a test that fails before the change.
- Test doubles only under `tests/`.
- No new policy logic inside cbrain/. Policy lives in PrivateVault.
- If a task requires a decision I have not made, stop and ask. Do not guess.

---

## P0 — correctness

### 1. Unify TLS peer identity (BLOCKER)

`cbrain/execution/planner.py:112` encodes `route.peer_identity` verbatim
(`tls-spki:...` per `deploy/config.example.json`). `_PinnedHTTPSChannel`
(`cbrain/execution/sidecar.py:551`) computes `tls-cert-sha256:{sha256(DER leaf)}`.
`_same_bytes` compares them. They can never match. No test crosses planner ->
PinnedHTTPSConnector; each side tests its own format.

Adopt SPKI pinning (survives cert rotation; full-leaf hashing does not, and
rotation is on the roadmap):

- Canonical form: `tls-spki-sha256:<64 hex>`.
- `_PinnedHTTPSChannel` extracts SubjectPublicKeyInfo from the DER leaf and
  hashes that, not the whole cert.
- Add a strict validator for `ToolRoute.peer_identity`: reject anything not
  matching `^tls-spki-sha256:[0-9a-f]{64}$` at config load, in
  `cbrain/deploy/config.py`. Fail at startup, never at dispatch.
- Update `deploy/config.example.json` with a real-shaped digest and a comment
  block above `routes` explaining how an operator computes it.
- Add an operator helper: `cbrain-spki <host> <port>` printing the pin.
- New test `tests/test_sidecar_peer_identity.py`: spin a local TLS server with a
  self-signed cert, run `ToolRoute` -> `plan()` -> `PinnedHTTPSConnector.open()`
  -> peer compare, assert match; then assert mismatch on a second cert.
- Delete `tls-spki:` / `tls-cert-sha256:` literals from every other test.

### 2. Witness independence flag

`cbrain/adapters/privatevault_execution.py:333` hardcodes
`require_witness_independence=True`, so `InProcessDispatchTransport` can never
reach EXECUTED against real Agent DNA — but `gateway.py` still carries a
`closure_writer.seal()` branch for it.

Decision: production requires independence. So:
- Delete the in-process closure-sealing branch from
  `PrivateVaultExecutionGateway.decide_and_execute`. If `result.closure is None`,
  that is `closure_unproven`, unconditionally.
- Keep `InProcessDispatchTransport` for unit tests only; add a module docstring
  line saying it cannot produce EXECUTED against real Agent DNA.
- Update `README.md` and `docs/architecture.md` to stop implying otherwise.

### 3. Knowledge runtime must not default to fakes

`KnowledgeRuntime.__init__` defaults `embeddings=DeterministicEmbeddingProvider()`
and `extractor=RuleBasedExtractor()`. A deployment that forgets to inject real
providers gets silently fake retrieval.

- Make `embeddings` and `extractor` required keyword arguments.
- Move `DeterministicEmbeddingProvider` and `RuleBasedExtractor` out of
  `cbrain/knowledge/stores/memory.py` into `tests/knowledge_fakes.py`.
- Fix all call sites and tests.

### 4. Move the eval gateway out of the production package

`cbrain/company/governance.py::CompanyRiskGateway` is a second policy engine
inside cbrain/. Reports already stamp `decision_authority=company_test_gateway`;
make the import path carry the same claim.

- Move it to `cbrain/evaluation/company_gateway.py`.
- Add `tests/test_no_second_policy_engine.py`: assert no module under
  `cbrain/adapters/`, `cbrain/execution/`, or `cbrain/agent/` imports it.

### 5. Sidecar hardening

In `cbrain/execution/sidecar.py`:
- `_PinnedHTTPSChannel.__init__` leaks the socket when `getpeercert()` returns
  empty — it raises after `connect()` and the caller's `finally` never sees a
  channel. Wrap construction so the connection is closed on any failure.
- `_content_length()` is called outside `do_POST`'s try block; a missing or
  non-numeric `Content-Length` raises into `handle_error`, printing a traceback
  to stderr — the one place logging is deliberately disabled. Move it inside and
  `send_error(411)`.
- `serve_sidecar` requires `CERT_REQUIRED` but never checks *which* client. Add a
  required `allowed_client_principals: frozenset[str]` and reject any peer whose
  cert subject CN / SAN is not in it. Fail closed with 403 and no body.

### 6. Launcher deny-list -> allow-list

`hermes_launcher.validate_runtime_arguments` splits on `=` and checks a deny-list;
any unlisted or aliased bypass flag passes. Invert it: an explicit allow-list of
permitted argv options, everything else refused with exit 78. Update
`tests/test_hermes_launcher.py` with aliased and bundled bypass attempts.

### 7. Doc drift

- `README.md` pins Agent DNA at `eabc02e8...`; `upstreams.lock.json` and CI use
  `3789a216...`. Make README read from the lock file's value. For a repo whose
  pitch is pinned upstreams, this must never drift again — add
  `tests/test_upstream_pins.py` asserting every commit SHA in README matches
  `upstreams.lock.json`.
- README claims 197/204 tests. Replace with a statement that does not go stale.
- README "Project Layout" omits `agent/`, `company/`, `knowledge/`, and the
  `evaluation/company_*` + `operator_*` modules. Regenerate it.
- README "Verification" omits `ruff format --check`, which CI enforces. Add it.

### 8. Agent loop: refusals are observations, not run-enders

`FoundationAgent._run_ephemeral` terminates the run on BLOCKED / REVIEW_REQUIRED.
A five-step task blocked on step two dies. The reason string is already
secret-free (`privatevault_block:{triggered_by}`).

- Add `AgentProfile.on_refusal: Literal["terminate", "observe"]`, default
  `"terminate"` (no behaviour change without opt-in).
- Under `"observe"`, append a TOOL message with status + request_id + reason
  (never arguments, never output) and continue the loop. Count it against
  `max_tool_calls`. Add `max_consecutive_refusals` to `limits.py`, default 3,
  terminating as REJECTED when exceeded.
- INDETERMINATE and CONTROL_FAILURE always terminate. Never observable, never
  retried. Test that explicitly.
- Mirror in `_run_durable`.

---

## P1 — moat

Each of these is separately defensible and separately demoable. Build in order.

### 9. Offline-verifiable evidence pack + standalone verifier

The strongest lock-in: an auditor verifies a governed action months later with
neither CBrain nor PrivateVault running.

- New `cbrain/evidence/pack.py`: `EvidencePack` bundling, for one request_id —
  canonical ActionIntent bytes, PreparedDispatch document (bytes + peer identity
  as digests, never raw payloads), decision receipt digest, signed authorization,
  dispatch witness, closure record, trust bundle, config digest, upstream pins,
  and a schema version. Canonical JSON, `cbrain-evidence-pack/v1`.
- Emit a pack from `PrivateVaultExecutionGateway` on every terminal status,
  including BLOCKED and INDETERMINATE. A refusal is evidence.
- New console script `cbrain-verify <pack.json> --trust-bundle <path>`: verifies
  the full chain using only Agent DNA verifiers + the embedded trust bundle.
  Exit 0 verified, 1 rejected with a reason code, 2 unverifiable. No network.
- Redaction test: assert no tool arguments, no outputs, no credential-shaped keys
  ever appear in a pack. Reuse the `FORBIDDEN_SIGNAL_KEYS` discipline from
  `cbrain/agent/insights.py`.

### 10. Tamper-evident local decision journal

Evidence must survive control-plane loss.

- New `cbrain/evidence/journal.py`: append-only hash-chained journal
  (`entry_digest = sha256(prev_digest || canonical(entry))`), one entry per
  governed outcome, written before the gateway returns.
- Postgres migration `003_decision_journal.sql` with a unique constraint on
  `(organisation_id, sequence)` and a CHECK on digest shape.
- `cbrain-verify journal <path>` detects any insertion, deletion, or reorder and
  names the first broken link.
- Test: mutate one entry mid-chain, assert detection at the right index.

### 11. Deterministic replay / counterfactual

"Prove the control would have refused this on the day it happened."

- `cbrain/evidence/replay.py`: take recorded ActionIntent bytes from packs or the
  journal and re-submit them to a decision client against a named policy bundle
  digest.
- Report per action: original disposition, replayed disposition, and whether they
  diverge. Divergence is the product, not an error.
- CLI: `cbrain-eval replay --from <journal|packs> --policy-bundle <digest>`.
- Must not execute anything. Assert in a test that no handler and no transport is
  ever reachable from the replay path.

### 12. Published conformance suite

Owning the spec is the moat. Whoever else builds this has to pass your tests.

- Extract `tests/test_execution_gateway_conformance.py` into an installable
  `cbrain-conformance` package under `conformance/`: a set of black-box cases any
  candidate gateway must satisfy (at-most-once, no execution on BLOCK/REVIEW,
  INDETERMINATE never retryable, permit single-use, exact-byte match, peer
  mismatch refusal, closure required).
- Runner takes any object satisfying `PrivateVaultGateway` and emits a signed
  conformance report with pass/fail per case.
- Include at least three intentionally-broken reference gateways under
  `conformance/fixtures/` that must fail specific cases. A suite that nothing
  fails proves nothing.

### 13. Adversarial escape corpus

- `cbrain/evaluation/escape_catalog.py`: canonical bypass attempts — model-supplied
  destination, credential-shaped tool argument, permit replay, permit rebinding to
  different bytes, TLS peer substitution, idempotency-key collision, approval
  replacement after approval, cross-matter widening, argument mutation between
  decision and dispatch.
- Each case asserts a specific refusal reason code, not just "did not execute".
- Wire into `cbrain-eval escape-run`. Gate CI on 100% refusal.

### 14. Time-bound permits

- Add `not_before` / `expires_at` to the authorization binding and verify both at
  claim time in the sidecar, using the sidecar's own clock, never the agent's.
- Explicit clock-skew tolerance as deployment config, default 0.
- Expired permit is CONTROL_FAILURE pre-send, never INDETERMINATE. Test both
  boundaries and the skew window.

### 15. Regulatory control mapping

- `docs/control-mapping.md`: table mapping each of the 12 security invariants to
  the specific obligations an FS buyer's second line will ask about — EU AI Act
  Art. 14 human oversight and Art. 12 record-keeping, DORA ICT risk and incident
  evidence, SR 11-7 model risk, RBI IT governance. One row per invariant: control,
  obligation, the artifact that evidences it, the test that enforces it.
- Every row must cite a real file and test name. No aspirational rows.
- Add `tests/test_control_mapping.py` asserting every referenced path exists.

---

## Out of scope

Do not add: a second policy engine, retry logic anywhere near INDETERMINATE,
vendored Agent DNA, live provider calls without `--confirm-live-api`, or any
telemetry that could carry tool arguments.
