"""Google Gemini generateContent adapter."""

from __future__ import annotations

import uuid
from collections.abc import Mapping
from typing import Any
from urllib.parse import quote

from .contracts import (
    CompletionRequest,
    Message,
    MessageRole,
    ModelOutput,
    ModelResponseError,
    ProviderContinuation,
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
                _message(
                    item,
                    model=self.model,
                    preceding_call=request.messages[index - 1].tool_call
                    if index
                    else None,
                )
                for index, item in enumerate(request.messages)
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
        return _parse_response(response, model=self.model)


def _message(
    message: Message, *, model: str, preceding_call: ToolCall | None = None
) -> dict[str, Any]:
    if message.tool_call is not None:
        call = message.tool_call
        if call.continuation is not None:
            return call.continuation.for_route(
                provider="google", model=model, wire_format="google"
            )
        return {
            "role": "model",
            "parts": [
                {
                    "functionCall": {
                        "name": call.name,
                        "args": call.arguments,
                    }
                }
            ],
        }
    if message.role is MessageRole.TOOL:
        function_response: dict[str, Any] = {
            "name": message.tool_name,
            "response": {"content": message.content},
        }
        if preceding_call is not None and preceding_call.continuation is not None:
            assistant = preceding_call.continuation.for_route(
                provider="google", model=model, wire_format="google"
            )
            for part in assistant["parts"]:
                if "functionCall" in part and "id" in part["functionCall"]:
                    function_response["id"] = part["functionCall"]["id"]
        return {
            "role": "user",
            "parts": [{"functionResponse": function_response}],
        }
    role = "model" if message.role is MessageRole.ASSISTANT else "user"
    return {"role": role, "parts": [{"text": message.content}]}


def _parse_response(response: Mapping[str, Any], *, model: str) -> ModelOutput:
    candidates = required_list(response.get("candidates"), "candidates")
    if len(candidates) != 1:
        raise ModelResponseError("provider must return exactly one candidate")
    candidate = required_mapping(candidates[0], "candidates[0]")
    if candidate.get("finishReason", "STOP") != "STOP":
        raise ModelResponseError("provider response did not complete")
    content = required_mapping(candidate.get("content"), "candidate content")
    parts = required_list(content.get("parts"), "candidate parts")
    calls: list[Mapping[str, Any]] = []
    texts: list[str] = []
    for index, raw_part in enumerate(parts):
        part = required_mapping(raw_part, f"candidate parts[{index}]")
        if "functionCall" in part:
            calls.append(required_mapping(part["functionCall"], "functionCall"))
        text = part.get("text")
        if isinstance(text, str) and text.strip() and not part.get("thought", False):
            texts.append(text)
    if calls:
        if len(calls) != 1:
            raise ModelResponseError("provider must return exactly one tool call")
        call = calls[0]
        name = required_response_text(call.get("name"), "tool call name")
        arguments = parse_arguments(call.get("args", {}), "tool arguments")
        return ToolCall.capture(
            call_id=(
                required_response_text(call["id"], "tool call ID")
                if "id" in call
                else f"google-{uuid.uuid4().hex}"
            ),
            name=name,
            arguments=arguments,
            continuation=ProviderContinuation.capture(
                provider="google",
                model=model,
                wire_format="google",
                message={"role": "model", "parts": parts},
            ),
        )
    if texts:
        return TextOutput("".join(texts))
    raise ModelResponseError("provider returned neither text nor a tool call")


__all__ = ["GoogleAdapter"]
