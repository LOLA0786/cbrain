# Foundation agent

CBrain's foundation agent is the reusable reasoning and tool loop that future
company-specific agents configure rather than reimplement.

## Components

| Piece | Module | Role |
| --- | --- | --- |
| Agent profile | `cbrain.agent.AgentProfile` | Identity, instructions, model route, tool allowlist, limits |
| Tool registry | `cbrain.agent.ToolRegistry` | Maps tool names to capabilities and model schemas |
| Foundation loop | `cbrain.agent.FoundationAgent` | Bounded model turns and governed tool execution |
| Governed runtime | `cbrain.runtime.GovernedRuntime` | Existing execution boundary (at-most-once, fail-closed) |
| Model routing | `cbrain.models.ModelRouter` | Deployment-owned provider selection |

PrivateVault remains optional. Wire any object that satisfies
`PrivateVaultGateway` into `GovernedRuntime`. The foundation agent never imports
PrivateVault or `agent_dna`.

## Define a profile

```python
from cbrain.agent import AgentProfile

profile = AgentProfile(
    agent_id="ops-agent",
    instructions="Answer operational questions with concise evidence.",
    model_route="local",
    permitted_tools=frozenset({"lookup_status"}),
    max_model_turns=6,
    max_tool_calls=3,
    timeout_seconds=120.0,
    metadata={"team": "platform"},
)
```

## Register permitted tools

```python
from cbrain.agent import GovernedTool, ToolRegistry

registry = ToolRegistry(
    [
        GovernedTool(
            name="lookup_status",
            capability="ops.status.read",
            description="Read the current service status document",
            input_schema={
                "type": "object",
                "properties": {"service": {"type": "string"}},
            },
        ),
    ]
)

handlers = {
    "lookup_status": lambda arguments: {
        "service": arguments["service"],
        "status": "healthy",
    },
}
```

Only tools listed in `permitted_tools` are exposed to the model and eligible for
execution. Everything else is rejected before `GovernedRuntime.execute`.

## Wire governed execution (optional PrivateVault)

```python
import time

from cbrain import GovernedRuntime
from cbrain.agent import FoundationAgent, RunInput
from cbrain.models import ModelRouter

class AllowGateway:
    independent_execution = False

    def decide_and_execute(self, action, handler):
        from cbrain import ExecutionStatus, GovernedExecution

        return GovernedExecution(
            status=ExecutionStatus.EXECUTED,
            request_id=action.request_id,
            tool_executed=True,
            reason="local_allow",
            output=handler(action.arguments),
        )

agent = FoundationAgent(
    profile=profile,
    runtime=GovernedRuntime(AllowGateway()),
    model_router=ModelRouter({"local": your_model_adapter}),
    tools=registry,
    handlers=handlers,
    clock=time.monotonic,
)

result = agent.run(RunInput(task="Is the payments service healthy?"))
```

In production, replace `AllowGateway` with the existing
`PrivateVaultExecutionGateway` composition. CBrain agent logic stays the same;
only the gateway adapter changes.

## Run lifecycle

```text
validated RunInput
→ initialized run (run_id, monotonic deadline, event trace)
→ model turn through ModelRouter
→ TextOutput → completed
→ ToolCall → allowlist check → ActionIntent → GovernedRuntime.execute
→ structured tool observation appended to bounded in-run history
→ repeat until limit, timeout, cancellation, or terminal failure
→ RunResult
```

Terminal statuses include `completed`, `rejected`, `invalid_model_response`,
`model_failure`, `tool_failure`, `limit_reached`, `timed_out`, and `cancelled`.

## Timeout and cancellation (cooperative)

`FoundationAgent.run()` is synchronous. Timeout and cancellation are checked
only **between** model turns. A model call or governed tool handler that is
already running is not interrupted.

Use a monotonic clock such as `time.monotonic` for the injected `clock`.
Wall-clock time can move backwards and break deadline enforcement.

Cancellation is cooperative via the optional `cancelled` callable, also checked
between turns.

## In-run bounds

`AgentProfile.limits` (`RunLimits`) caps task size, model text size, tool
observation size, context message count, and model output tokens for one run.
These bounds are explicit and configurable; there is no persistent memory.

## Concurrency

One `FoundationAgent` instance may be reused for sequential runs. Concurrent
`run()` calls on the same instance are not supported.

## Create a future company agent

Do not copy the loop. Supply a different profile, instructions, tool registry,
handlers, and model route:

```python
sales_profile = AgentProfile(
    agent_id="sales-agent",
    instructions="Draft customer-safe summaries.",
    model_route="anthropic",
    permitted_tools=frozenset({"crm_lookup"}),
    max_model_turns=8,
    max_tool_calls=4,
    timeout_seconds=180.0,
)

sales_agent = FoundationAgent(
    profile=sales_profile,
    runtime=shared_runtime,
    model_router=shared_router,
    tools=sales_registry,
    handlers=sales_handlers,
    clock=shared_clock,  # use time.monotonic
)
```

The same `FoundationAgent` class serves every configured agent.

## Insights and personalization

Human-approved personalization may change instructions and non-authoritative
metadata only. It never authorizes tools, changes routes, or widens limits.
See `docs/hyperpersonalized-harness.md`.

Learning storage stays outside the foundation loop. Observe a finished run
explicitly:

```python
result = agent.run(run_input)
LearningRecorder(store).record_run(
    agent_id=profile.agent_id,
    result=result,
)
```

A learning-store failure must not change `result`. Every consequential
`ActionIntent` still enters `GovernedRuntime`.

## Separation from PrivateVault

| Layer | Responsibility |
| --- | --- |
| `FoundationAgent` | Reasoning loop, model calls, tool selection, run state |
| `GovernedRuntime` | Framework-neutral execution entry point |
| PrivateVault adapters | Optional authorization, evidence, and dispatch enforcement |

The model never calls a provider tool implementation, shell, or network client
directly. Effectful work always passes through `GovernedRuntime.execute`.

Indeterminate tool outcomes are never automatically retried. This matches the
existing runtime contract.
