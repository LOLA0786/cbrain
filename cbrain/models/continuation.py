"""Validate assistant history without granting it any execution authority."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .contracts import (
    ModelContractError,
    ProviderContinuation,
    ToolCall,
    _canonical_object,
    parse_arguments,
    required_list,
    required_mapping,
    required_response_text,
)


def validate_continuation_call(turn: ProviderContinuation, call: ToolCall) -> None:
    message = turn.message
    expected_role = "model" if turn.wire_format == "google" else "assistant"
    if message.get("role") != expected_role:
        raise ModelContractError("continuation must contain an assistant turn")
    wire_id: str | None
    if turn.wire_format == "chat_completions":
        _keys(message, {"role", "content", "tool_calls", "reasoning_content"})
        for key in ("content", "reasoning_content"):
            if message.get(key) is not None and not isinstance(message[key], str):
                raise ModelContractError("unsupported assistant text content")
        calls = required_list(message.get("tool_calls"), "assistant tool calls")
        block = _one(calls)
        _keys(block, {"id", "type", "function"})
        if block.get("type") != "function":
            raise ModelContractError("unsupported assistant tool type")
        function = required_mapping(block.get("function"), "assistant function")
        _keys(function, {"name", "arguments"})
        wire_id = required_response_text(block.get("id"), "assistant call ID")
        name = function.get("name")
        arguments = parse_arguments(function.get("arguments"), "assistant arguments")
    elif turn.wire_format == "anthropic":
        _keys(message, {"role", "content"})
        blocks = required_list(message.get("content"), "assistant content")
        calls = []
        for raw in blocks:
            block = required_mapping(raw, "assistant content block")
            kind = block.get("type")
            if kind == "tool_use":
                _keys(block, {"type", "id", "name", "input"})
                calls.append(block)
            elif kind == "text":
                _keys(block, {"type", "text", "citations"})
                _text(block.get("text"))
            elif kind == "thinking":
                _keys(block, {"type", "thinking", "signature"})
                _text(block.get("thinking"))
                required_response_text(block.get("signature"), "thinking signature")
            elif kind == "redacted_thinking":
                _keys(block, {"type", "data"})
                required_response_text(block.get("data"), "redacted thinking")
            else:
                raise ModelContractError("unsupported assistant content block")
        block = _one(calls)
        wire_id = required_response_text(block.get("id"), "assistant call ID")
        name = block.get("name")
        arguments = parse_arguments(block.get("input"), "assistant arguments")
    else:
        _keys(message, {"role", "parts"})
        parts = required_list(message.get("parts"), "assistant parts")
        calls = []
        for raw in parts:
            part = required_mapping(raw, "assistant part")
            _keys(part, {"functionCall", "text", "thought", "thoughtSignature"})
            if "thoughtSignature" in part:
                required_response_text(part["thoughtSignature"], "thought signature")
            if "thought" in part and not isinstance(part["thought"], bool):
                raise ModelContractError("invalid thought marker")
            if "functionCall" in part and "text" not in part:
                calls.append(
                    required_mapping(part["functionCall"], "assistant function")
                )
            elif "text" in part and "functionCall" not in part:
                _text(part["text"])
            else:
                raise ModelContractError("unsupported assistant part")
        function = _one(calls)
        _keys(function, {"id", "name", "args"})
        wire_id = (
            required_response_text(function["id"], "assistant call ID")
            if "id" in function
            else None
        )
        name = function.get("name")
        arguments = parse_arguments(function.get("args", {}), "assistant arguments")
    if (wire_id is not None and wire_id != call.call_id) or name != call.name:
        raise ModelContractError("continuation tool identity mismatch")
    if _canonical_object(arguments, "continuation arguments") != _canonical_object(
        call.arguments, "tool arguments"
    ):
        raise ModelContractError("continuation tool arguments mismatch")


def _keys(value: Mapping[str, Any], allowed: set[str]) -> None:
    if set(value) - allowed:
        raise ModelContractError("unsupported continuation fields")


def _one(values: list[Any]) -> Mapping[str, Any]:
    if len(values) != 1:
        raise ModelContractError("continuation requires exactly one tool call")
    return required_mapping(values[0], "assistant tool call")


def _text(value: object) -> None:
    if not isinstance(value, str):
        raise ModelContractError("assistant text must be a string")
