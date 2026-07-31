"""OpenAI-compatible adapter for OpenAI, xAI, and RunPod/vLLM."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .contracts import (
    CompletionRequest,
    MessageRole,
    ModelOutput,
    ModelResponseError,
    TextOutput,
    ToolCall,
    parse_arguments,
    required_list,
    required_mapping,
    required_response_text,
)
from .transport import HeadersProvider, JsonModelTransport


class OpenAICompatibleAdapter:
    """Translate neutral requests to the Chat Completions wire contract."""

    def __init__(
        self,
        *,
        provider: str,
        model: str,
        transport: JsonModelTransport,
        headers_provider: HeadersProvider,
        path: str = "/v1/chat/completions",
    ) -> None:
        if not provider.strip() or not model.strip():
            raise ValueError("provider and model must be non-empty")
        if not path.startswith("/") or path.startswith("//"):
            raise ValueError("path must be an absolute origin path")
        self._provider = provider
        self._model = model
        self._transport = transport
        self._headers_provider = headers_provider
        self._path = path

    @property
    def provider(self) -> str:
        return self._provider

    @property
    def model(self) -> str:
        return self._model

    def complete(self, request: CompletionRequest) -> ModelOutput:
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [_message(item) for item in request.messages],
            "max_tokens": request.max_output_tokens,
            "temperature": request.temperature,
        }
        if request.tools:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.input_schema,
                    },
                }
                for tool in request.tools
            ]
            payload["tool_choice"] = "auto"
        response = self._transport.post_json(
            self._path,
            payload,
            self._headers_provider(),
        )
        return _parse_response(response)


def _message(message: object) -> dict[str, Any]:
    from .contracts import Message

    if not isinstance(message, Message):
        raise TypeError("message must be a Message")
    payload: dict[str, Any] = {
        "role": message.role.value,
        "content": message.content,
    }
    if message.role is MessageRole.TOOL:
        payload["tool_call_id"] = message.tool_call_id
        payload["name"] = message.tool_name
    return payload


def _parse_response(response: Mapping[str, Any]) -> ModelOutput:
    choices = required_list(response.get("choices"), "choices")
    if not choices:
        raise ModelResponseError("choices must not be empty")
    choice = required_mapping(choices[0], "choices[0]")
    message = required_mapping(choice.get("message"), "choices[0].message")
    raw_calls = message.get("tool_calls")
    content = message.get("content")
    nonempty_content = isinstance(content, str) and bool(content.strip())
    if raw_calls is not None:
        calls = required_list(raw_calls, "message.tool_calls")
        if len(calls) != 1:
            raise ModelResponseError("provider must return exactly one tool call")
        if nonempty_content:
            raise ModelResponseError("provider returned both text and a tool call")
        call = required_mapping(calls[0], "message.tool_calls[0]")
        function = required_mapping(call.get("function"), "tool call function")
        return ToolCall.capture(
            call_id=required_response_text(call.get("id"), "tool call ID"),
            name=required_response_text(function.get("name"), "tool call name"),
            arguments=parse_arguments(function.get("arguments"), "tool arguments"),
        )
    if nonempty_content:
        return TextOutput(required_response_text(content, "message content"))
    raise ModelResponseError("provider returned neither text nor a tool call")


__all__ = ["OpenAICompatibleAdapter"]
