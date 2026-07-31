"""Google Gemini generateContent adapter."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

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


class GoogleAdapter:
    def __init__(
        self,
        *,
        model: str,
        transport: JsonModelTransport,
        headers_provider: HeadersProvider,
    ) -> None:
        normalized = model.removeprefix("models/")
        if not normalized.strip() or "/" in normalized:
            raise ValueError("Google model must be one non-empty model identifier")
        self._model = normalized
        self._transport = transport
        self._headers_provider = headers_provider

    @property
    def provider(self) -> str:
        return "google"

    @property
    def model(self) -> str:
        return self._model

    def complete(self, request: CompletionRequest) -> ModelOutput:
        system = "\n\n".join(
            item.content for item in request.messages if item.role is MessageRole.SYSTEM
        )
        payload: dict[str, Any] = {
            "contents": [
                _message(item)
                for item in request.messages
                if item.role is not MessageRole.SYSTEM
            ],
            "generationConfig": {
                "maxOutputTokens": request.max_output_tokens,
                "temperature": request.temperature,
            },
        }
        if system:
            payload["systemInstruction"] = {"parts": [{"text": system}]}
        if request.tools:
            payload["tools"] = [
                {
                    "functionDeclarations": [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "parameters": tool.input_schema,
                        }
                        for tool in request.tools
                    ]
                }
            ]
        path = f"/v1beta/models/{quote(self._model, safe='-._')}:generateContent"
        response = self._transport.post_json(
            path,
            payload,
            self._headers_provider(),
        )
        return _parse_response(response)


def _message(message: Message) -> dict[str, Any]:
    if message.role is MessageRole.TOOL:
        return {
            "role": "user",
            "parts": [
                {
                    "functionResponse": {
                        "name": message.tool_name,
                        "response": {"content": message.content},
                    }
                }
            ],
        }
    role = "model" if message.role is MessageRole.ASSISTANT else "user"
    return {"role": role, "parts": [{"text": message.content}]}


def _parse_response(response: Mapping[str, Any]) -> ModelOutput:
    candidates = required_list(response.get("candidates"), "candidates")
    if not candidates:
        raise ModelResponseError("candidates must not be empty")
    candidate = required_mapping(candidates[0], "candidates[0]")
    content = required_mapping(candidate.get("content"), "candidate content")
    parts = required_list(content.get("parts"), "candidate parts")
    calls: list[Mapping[str, Any]] = []
    texts: list[str] = []
    for index, raw_part in enumerate(parts):
        part = required_mapping(raw_part, f"candidate parts[{index}]")
        if "functionCall" in part:
            calls.append(required_mapping(part["functionCall"], "functionCall"))
        text = part.get("text")
        if isinstance(text, str) and text.strip():
            texts.append(text)
    if calls:
        if len(calls) != 1:
            raise ModelResponseError("provider must return exactly one tool call")
        if texts:
            raise ModelResponseError("provider returned both text and a tool call")
        call = calls[0]
        name = required_response_text(call.get("name"), "tool call name")
        arguments = parse_arguments(call.get("args"), "tool arguments")
        canonical = json.dumps(
            {"name": name, "arguments": arguments},
            separators=(",", ":"),
            sort_keys=True,
        ).encode()
        return ToolCall.capture(
            call_id=f"google-{hashlib.sha256(canonical).hexdigest()[:24]}",
            name=name,
            arguments=arguments,
        )
    if texts:
        return TextOutput("".join(texts))
    raise ModelResponseError("provider returned neither text nor a tool call")


__all__ = ["GoogleAdapter"]
