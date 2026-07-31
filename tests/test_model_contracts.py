from __future__ import annotations

import pytest

from cbrain.models import (
    CompletionRequest,
    Message,
    MessageRole,
    ModelContractError,
    ToolCall,
    ToolDefinition,
)


def tool(schema: dict[str, object] | None = None) -> ToolDefinition:
    return ToolDefinition.capture(
        name="payments_transfer_initiate",
        description="Initiate a payment transfer.",
        input_schema=schema
        or {
            "type": "object",
            "properties": {"amount": {"type": "string"}},
            "required": ["amount"],
            "additionalProperties": False,
        },
    )


@pytest.mark.parametrize(
    "name",
    ["api_key", "accessToken", "authorization", "private-key", "password"],
)
def test_tool_schema_rejects_credential_properties(name: str) -> None:
    with pytest.raises(ModelContractError, match="credential-shaped"):
        tool(
            {
                "type": "object",
                "properties": {name: {"type": "string"}},
                "required": [name],
            }
        )


def test_model_tool_arguments_reject_credentials_recursively() -> None:
    with pytest.raises(ModelContractError, match="credential-shaped"):
        ToolCall.capture(
            call_id="call-1",
            name="payments_transfer_initiate",
            arguments={"nested": {"bearer_token": "must-not-enter-context"}},
        )


def test_tool_snapshots_schema_and_arguments() -> None:
    schema: dict[str, object] = {
        "type": "object",
        "properties": {"amount": {"type": "string"}},
    }
    arguments = {"amount": "50000.00"}
    definition = tool(schema)
    call = ToolCall.capture(call_id="call-1", name=definition.name, arguments=arguments)

    schema["properties"] = {}
    arguments["amount"] = "1.00"

    assert definition.input_schema["properties"] == {"amount": {"type": "string"}}
    assert call.arguments == {"amount": "50000.00"}


def test_completion_request_rejects_duplicate_tools() -> None:
    definition = tool()
    with pytest.raises(ModelContractError, match="unique"):
        CompletionRequest(
            messages=(Message(MessageRole.USER, "Transfer funds."),),
            tools=(definition, definition),
        )


def test_tool_messages_require_call_identity_and_name() -> None:
    with pytest.raises(ModelContractError, match="tool_call_id"):
        Message(MessageRole.TOOL, "done")

    message = Message(
        MessageRole.TOOL,
        "done",
        tool_call_id="call-1",
        tool_name="payments_transfer_initiate",
    )
    assert message.tool_name == "payments_transfer_initiate"
