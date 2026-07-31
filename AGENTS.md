# CBrain engineering contract

## Mission

CBrain is the framework-neutral governance harness between agent planning and
real tool execution. PrivateVault is the authorization and evidence authority.

## Non-negotiable invariants

- Every consequential tool call must enter `GovernedRuntime`.
- Framework adapters translate requests; they never decide policy.
- PrivateVault is the only component allowed to authorize dispatch.
- BLOCKED, REVIEW_REQUIRED, and CONTROL_FAILURE never execute tools.
- One request may invoke its tool handler at most once.
- Possible execution without proven closure becomes INDETERMINATE.
- INDETERMINATE outcomes are never automatically retried.
- Secrets never enter model context, GBrain memory, arguments, or logs.
- Production integrations use commits pinned in `upstreams.lock.json`.
- Test doubles are permitted only under `tests/`.

## Verification

Run before every commit:

    python -m pytest -q
    python -m compileall -q cbrain tests
    git diff --check
