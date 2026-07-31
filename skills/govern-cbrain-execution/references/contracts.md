# CBrain governed-execution reference

## Authority boundary

| Component | Permitted role |
| --- | --- |
| Model provider | Produce text or one tool proposal |
| Framework adapter | Translate a proposal into `ActionIntent` |
| GBrain | Supply memory and skills without secrets or authority |
| CBrain | Enforce runtime invariants and fail-closed availability |
| PrivateVault | Make the authoritative decision and sign evidence |
| Dispatcher sidecar | Send exact permitted bytes using sidecar-held credentials |

## Failure mapping

- BLOCK or REQUIRE_APPROVAL: do not dispatch.
- Control failure before send: `CONTROL_FAILURE`, `tool_executed=False`.
- Successful verified closure: `EXECUTED`, `tool_executed=True`.
- Failure after send may have started: `INDETERMINATE`, `tool_executed=None`,
  `retryable=False`.

## Model configuration

Deployment-owned settings:

- `CBRAIN_ANTHROPIC_MODEL`
- `CBRAIN_OPENAI_MODEL`
- `CBRAIN_XAI_MODEL`
- `CBRAIN_GOOGLE_MODEL`
- `CBRAIN_RUNPOD_MODEL`
- `CBRAIN_RUNPOD_BASE_URL`

Transport-only credential variables:

- `ANTHROPIC_API_KEY`
- `OPENAI_API_KEY`
- `XAI_API_KEY`
- `GOOGLE_API_KEY`
- `RUNPOD_API_KEY`

Do not serialize credential values into any neutral model request or evaluation
report.

## Defensible evaluation claim

Use: “For identical ActionIntent inputs, PrivateVault decision divergence across
the evaluated model routes was zero.”

Also report proposal frequency, exact canonical proposals, text responses,
unmatched calls, and control failures by model. Do not imply that models behave
identically.

## Verification

Run from the repository root:

```bash
.venv/bin/ruff format cbrain integrations tests
.venv/bin/ruff check cbrain integrations tests
.venv/bin/mypy cbrain
.venv/bin/python -m pytest -q
.venv/bin/python -m compileall -q cbrain tests
uv lock --check
git diff --check
```
