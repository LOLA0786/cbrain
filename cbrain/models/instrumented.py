"""Latency and token-usage instrumentation for model adapters."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from .contracts import (
    CompletionRequest,
    ModelAdapter,
    ModelOutput,
    TextOutput,
    ToolCall,
)
from .usage import TokenUsage, UsageSource


@dataclass(frozen=True, slots=True)
class CompletionObservation:
    output: ModelOutput
    usage: TokenUsage
    latency_seconds: float
    route_id: str


class InstrumentedModelAdapter:
    """Wrap a model adapter and record latency plus usage for each completion."""

    def __init__(
        self,
        *,
        route_id: str,
        adapter: ModelAdapter,
        run_id: str | None = None,
        chars_per_token: int = 4,
        cached_input_tokens: int = 0,
    ) -> None:
        if not route_id.strip():
            raise ValueError("route_id must be non-empty")
        if chars_per_token < 1:
            raise ValueError("chars_per_token must be positive")
        if cached_input_tokens < 0:
            raise ValueError("cached_input_tokens must be non-negative")
        self._route_id = route_id
        self._adapter = adapter
        self._run_id = run_id
        self._chars_per_token = chars_per_token
        self._cached_input_tokens = cached_input_tokens
        self.observations: list[CompletionObservation] = []

    @property
    def route_id(self) -> str:
        return self._route_id

    @property
    def provider(self) -> str:
        return self._adapter.provider

    @property
    def model(self) -> str:
        return self._adapter.model

    def complete(self, request: CompletionRequest) -> ModelOutput:
        started = time.monotonic()
        output = self._adapter.complete(request)
        latency = time.monotonic() - started
        if not isinstance(output, (TextOutput, ToolCall)):
            usage = TokenUsage.unknown(
                provider=self._adapter.provider,
                model=self._adapter.model,
                run_id=self._run_id,
            )
        else:
            usage = self._resolve_usage(request=request, output=output)
        self.observations.append(
            CompletionObservation(
                output=output,
                usage=usage,
                latency_seconds=latency,
                route_id=self._route_id,
            )
        )
        return output

    def _resolve_usage(
        self,
        *,
        request: CompletionRequest,
        output: ModelOutput,
    ) -> TokenUsage:
        reported = getattr(self._adapter, "last_reported_usage", None)
        if isinstance(reported, Mapping):
            return _provider_reported_usage(
                provider=self._adapter.provider,
                model=self._adapter.model,
                payload=reported,
                run_id=self._run_id,
            )
        return _estimate_usage(
            provider=self._adapter.provider,
            model=self._adapter.model,
            request=request,
            output=output,
            chars_per_token=self._chars_per_token,
            run_id=self._run_id,
            cached_input_tokens=self._cached_input_tokens,
        )


def _estimate_usage(
    *,
    provider: str,
    model: str,
    request: CompletionRequest,
    output: ModelOutput,
    chars_per_token: int,
    run_id: str | None,
    cached_input_tokens: int = 0,
) -> TokenUsage:
    input_chars = sum(len(message.content) for message in request.messages)
    input_chars += sum(
        len(json.dumps(tool.input_schema, sort_keys=True)) for tool in request.tools
    )
    input_tokens = _tokens_from_chars(input_chars, chars_per_token)
    if cached_input_tokens > input_tokens:
        cached_input_tokens = input_tokens
    billable_input = input_tokens - cached_input_tokens
    if isinstance(output, TextOutput):
        output_tokens = _tokens_from_chars(len(output.text), chars_per_token)
    else:
        output_tokens = _tokens_from_chars(
            len(json.dumps(output.arguments, sort_keys=True)),
            chars_per_token,
        )
    return TokenUsage(
        provider=provider,
        model=model,
        source=UsageSource.ESTIMATED,
        input_tokens=billable_input,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=0,
        total_tokens=billable_input + cached_input_tokens + output_tokens,
        run_id=run_id,
        cached_input_accounting=(
            "simulated_assumption" if cached_input_tokens > 0 else None
        ),
    )


def _provider_reported_usage(
    *,
    provider: str,
    model: str,
    payload: Mapping[str, Any],
    run_id: str | None,
) -> TokenUsage:
    cached = _optional_int(payload.get("cached_input_tokens"))
    return TokenUsage(
        provider=provider,
        model=model,
        source=UsageSource.PROVIDER_REPORTED,
        input_tokens=_optional_int(payload.get("input_tokens")),
        cached_input_tokens=cached,
        output_tokens=_optional_int(payload.get("output_tokens")),
        reasoning_tokens=_optional_int(payload.get("reasoning_tokens")),
        total_tokens=_optional_int(payload.get("total_tokens")),
        request_id=_optional_text(payload.get("request_id")),
        run_id=run_id,
        cached_input_accounting=(
            "provider_reported" if cached is not None and cached > 0 else None
        ),
    )


def _tokens_from_chars(char_count: int, chars_per_token: int) -> int:
    if char_count <= 0:
        return 0
    return max(1, (char_count + chars_per_token - 1) // chars_per_token)


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("usage integer field is invalid")
    return value


def _optional_text(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError("usage text field is invalid")
    return value


__all__ = [
    "CompletionObservation",
    "InstrumentedModelAdapter",
]
