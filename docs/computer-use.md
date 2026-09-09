# Governed computer / personal browser

**Status:** production seam + out-of-process worker protocol. Stub engine is
CI-default; Playwright is an optional worker engine (`computer` extra).

## Shape (Grok-like UX, CBrain invariants)

```text
FoundationAgent
   │  tools: computer_observe|navigate|click|type|submit
   │  model args: page_alias (+ selector/text) — never URL
   ▼
ComputerToolBridge          ← FA already owns GovernedRuntime
   │  catalog.resolve(alias) → operator URL
   ▼
RemoteComputerBackend       ← DEV_ONLY=False; dials worker only
   │  Bearer token; JSON schema cbrain-computer-worker/request-v1
   ▼
cbrain-computer-worker      ← separate process / pod
   │  StubBrowserEngine | PlaywrightBrowserEngine
   ▼
Allowlisted web destinations (worker network only)
```

Agent namespace **cannot** reach business web IPs. Isolation compose:
`deploy/computer/` — agent ↛ web-fixture; agent → computer-worker → (worker net).

## Screenshots

Workers may capture PNG bytes. The agent/model receive only
`screenshot_sha256` (+ media type). Raw pixels stay in the worker.

## Assembly

```python
from cbrain.computer_use import (
    ComputerToolBridge,
    PageAlias,
    PageCatalog,
    RemoteComputerBackend,
    computer_tools,
    require_production_computer_backend,
)

backend = RemoteComputerBackend(
    endpoint="https://computer-worker.internal/v1/computer",
    token=os.environ["CBRAIN_COMPUTER_TOKEN"],
)
require_production_computer_backend(backend)

catalog = PageCatalog([
    PageAlias(
        alias="banking.portal",
        url="https://banking.example/portal",
        capability="computer.navigate.banking",
        description="Retail banking portal",
    ),
])
bridge = ComputerToolBridge(catalog=catalog, backend=backend)
# Merge computer_tools() into ToolRegistry; bridge.handlers() into handlers.
```

Worker:

```bash
export CBRAIN_COMPUTER_TOKEN=...
export CBRAIN_COMPUTER_ENGINE=stub   # or playwright
cbrain-computer-worker --host 0.0.0.0 --port 8765
```

## Non-claims

- Playwright is not required for CI green.
- Open-web browsing from the agent pod.
- Model-supplied destinations or credentials.
