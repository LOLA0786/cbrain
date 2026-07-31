"""Anthropic Messages API adapter."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .contracts import (
    CompletionRequest,
    Message,
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


class AnthropicAdapter:
    def __init__(
        self,
        *,
        model: str,
        transport: JsonModelTransport,
        headers_provider: HeadersProvider,
        anthropic_version: str = "2023-06-01",
    ) -> None:
        if not model.strip() or not anthropic_version.strip():
            raise ValueError("model and anthropic_version must be non-empty")
        self._model = model
        self._transport = transport
        self._headers_provider = headers_provider
        self._anthropic_version = anthropic_version

    @property
    def provider(self) -> str:
        return "anthropic"

    @property
    def model(self) -> str:
        return self._model

    def complete(self, request: CompletionRequest) -> ModelOutput:
        system = "\n\n".join(
            item.content for item in request.messages if item.role is MessageRole.SYSTEM
        )
        payload: dict[str, Any] = {
            "model": self._model,
            "messages": [
                _message(item)
                for item in request.messages
                if item.role is not MessageRole.SYSTEM
            ],
            "max_tokens": request.max_output_tokens,
            "temperature": request.temperature,
        }
        if system:
            payload["system"] = system
        if request.tools:
            payload["tools"] = [
                {
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": tool.input_schema,
                }
                for tool in request.tools
            ]
        headers = dict(self._headers_provider())
        headers["anthropic-version"] = self._anthropic_version
        response = self._transport.post_json("/v1/messages", payload, headers)
        return _parse_response(response)


def _message(message: Message) -> dict[str, Any]:
    if message.role is MessageRole.TOOL:
        return {
            "role": "user",
            "content": [
                {
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id,
                    "content": message.content,
                }
            ],
        }
    return {"role": message.role.value, "content": message.content}


def _parse_response(response: Mapping[str, Any]) -> ModelOutput:
    content = required_list(response.get("content"), "content")
    tool_blocks: list[Mapping[str, Any]] = []
    text_parts: list[str] = []
    for index, raw_block in enumerate(content):
        block = required_mapping(raw_block, f"content[{index}]")
        block_type = block.get("type")
        if block_type == "tool_use":
            tool_blocks.append(block)
        elif block_type == "text":
            text = block.get("text")
            if isinstance(text, str) and text.strip():
                text_parts.append(text)
    if tool_blocks:
        if len(tool_blocks) != 1:
            raise ModelResponseError("provider must return exactly one tool call")
        if text_parts:
            raise ModelResponseError("provider returned both text and a tool call")
        block = tool_blocks[0]
        return ToolCall.capture(
            call_id=required_response_text(block.get("id"), "tool call ID"),
            name=required_response_text(block.get("name"), "tool call name"),
            arguments=parse_arguments(block.get("input"), "tool arguments"),
        )
    if text_parts:
        return TextOutput("".join(text_parts))
    raise ModelResponseError("provider returned neither text nor a tool call")


__all__ = ["AnthropicAdapter"]
