from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from cbrain.models import (
    AnthropicAdapter,
    CompletionRequest,
    GoogleAdapter,
    Message,
    MessageRole,
    ModelResponseError,
    OpenAICompatibleAdapter,
    TextOutput,
    ToolCall,
    ToolDefinition,
)


class CaptureTransport:
    def __init__(self, response: Mapping[str, Any]) -> None:
        self.response = response
        self.calls: list[tuple[str, Mapping[str, Any], Mapping[str, str]]] = []

    def post_json(
        self,
        path: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str],
    ) -> Mapping[str, Any]:
        self.calls.append((path, payload, headers))
        return self.response


def request() -> CompletionRequest:
    return CompletionRequest(
        messages=(
            Message(MessageRole.SYSTEM, "Use tools carefully."),
            Message(MessageRole.USER, "Read contact-1."),
        ),
        tools=(
            ToolDefinition.capture(
                name="crm_contact_read",
                description="Read one CRM contact.",
                input_schema={
                    "type": "object",
                    "properties": {"contact_id": {"type": "string"}},
                    "required": ["contact_id"],
                    "additionalProperties": False,
                },
            ),
        ),
    )


def test_openai_adapter_translates_and_parses_one_tool_call() -> None:
    transport = CaptureTransport(
        {
            "choices": [
                {
                    "message": {
                        "content": None,
                        "tool_calls": [
                            {
                                "id": "call-openai",
                                "function": {
                                    "name": "crm_contact_read",
                                    "arguments": '{"contact_id":"contact-1"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }
    )
    adapter = OpenAICompatibleAdapter(
        provider="openai",
        model="configured-openai-model",
        transport=transport,
        headers_provider=lambda: {"Authorization": "Bearer hidden"},
    )

    output = adapter.complete(request())

    assert isinstance(output, ToolCall)
    assert output.arguments == {"contact_id": "contact-1"}
    path, payload, headers = transport.calls[0]
    assert path == "/v1/chat/completions"
    assert payload["model"] == "configured-openai-model"
    assert payload["tools"][0]["function"]["name"] == "crm_contact_read"
    assert "hidden" not in str(payload)
    assert headers == {"Authorization": "Bearer hidden"}


def test_anthropic_adapter_keeps_system_separate_and_parses_tool_use() -> None:
    transport = CaptureTransport(
        {
            "content": [
                {
                    "type": "tool_use",
                    "id": "call-anthropic",
                    "name": "crm_contact_read",
                    "input": {"contact_id": "contact-1"},
                }
            ]
        }
    )
    adapter = AnthropicAdapter(
        model="configured-anthropic-model",
        transport=transport,
        headers_provider=lambda: {"x-api-key": "hidden"},
    )

    output = adapter.complete(request())

    assert isinstance(output, ToolCall)
    path, payload, headers = transport.calls[0]
    assert path == "/v1/messages"
    assert payload["system"] == "Use tools carefully."
    assert payload["messages"] == [{"role": "user", "content": "Read contact-1."}]
    assert headers["anthropic-version"] == "2023-06-01"
    assert "hidden" not in str(payload)


def test_google_adapter_translates_and_derives_stable_call_id() -> None:
    transport = CaptureTransport(
        {
            "candidates": [
                {
                    "content": {
                        "parts": [
                            {
                                "functionCall": {
                                    "name": "crm_contact_read",
                                    "args": {"contact_id": "contact-1"},
                                }
                            }
                        ]
                    }
                }
            ]
        }
    )
    adapter = GoogleAdapter(
        model="configured-google-model",
        transport=transport,
        headers_provider=lambda: {"x-goog-api-key": "hidden"},
    )

    first = adapter.complete(request())
    second = adapter.complete(request())

    assert isinstance(first, ToolCall)
    assert isinstance(second, ToolCall)
    assert first.call_id == second.call_id
    path, payload, _ = transport.calls[0]
    assert path.endswith("configured-google-model:generateContent")
    assert payload["systemInstruction"] == {"parts": [{"text": "Use tools carefully."}]}
    assert "hidden" not in str(payload)


def test_openai_text_response_is_neutral_text_output() -> None:
    transport = CaptureTransport(
        {"choices": [{"message": {"content": "No tool is required."}}]}
    )
    adapter = OpenAICompatibleAdapter(
        provider="xai",
        model="configured-xai-model",
        transport=transport,
        headers_provider=lambda: {},
    )

    assert adapter.complete(request()) == TextOutput("No tool is required.")


def test_provider_refuses_ambiguous_text_and_tool_call() -> None:
    transport = CaptureTransport(
        {
            "choices": [
                {
                    "message": {
                        "content": "I will call a tool.",
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "function": {
                                    "name": "crm_contact_read",
                                    "arguments": '{"contact_id":"contact-1"}',
                                },
                            }
                        ],
                    }
                }
            ]
        }
    )
    adapter = OpenAICompatibleAdapter(
        provider="runpod",
        model="configured-runpod-model",
        transport=transport,
        headers_provider=lambda: {},
    )

    with pytest.raises(ModelResponseError, match="both text and a tool call"):
        adapter.complete(request())


def test_provider_refuses_credential_shaped_returned_arguments() -> None:
    transport = CaptureTransport(
        {
            "choices": [
                {
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call-1",
                                "function": {
                                    "name": "crm_contact_read",
                                    "arguments": '{"api_key":"stolen"}',
                                },
                            }
                        ]
                    }
                }
            ]
        }
    )
    adapter = OpenAICompatibleAdapter(
        provider="openai",
        model="configured-model",
        transport=transport,
        headers_provider=lambda: {},
    )

    with pytest.raises(ModelResponseError, match="credential-shaped"):
        adapter.complete(request())
