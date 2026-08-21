"""Strict token usage accounting for model completions."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum
from typing import Any


class UsageSource(StrEnum):
    PROVIDER_REPORTED = "provider_reported"
    ESTIMATED = "estimated"
    UNKNOWN = "unknown"


class UsageContractError(ValueError):
    """Token usage payload is missing, inconsistent, or invalid."""


@dataclass(frozen=True, slots=True)
class TokenUsage:
    """Provider-neutral token usage for one model completion."""

    provider: str
    model: str
    source: UsageSource
    input_tokens: int | None = None
    cached_input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None
    request_id: str | None = None
    run_id: str | None = None
    cached_input_accounting: str | None = None

    def __post_init__(self) -> None:
        _required_text(self.provider, "provider")
        _required_text(self.model, "model")
        if not isinstance(self.source, UsageSource):
            raise UsageContractError("source must be a UsageSource")
        for field_name in (
            "input_tokens",
            "cached_input_tokens",
            "output_tokens",
            "reasoning_tokens",
            "total_tokens",
        ):
            value = getattr(self, field_name)
            if value is not None:
                _non_negative_int(value, field_name)
        if self.source is UsageSource.UNKNOWN and any(
            getattr(self, field_name) is not None
            for field_name in (
                "input_tokens",
                "cached_input_tokens",
                "output_tokens",
                "reasoning_tokens",
                "total_tokens",
            )
        ):
            raise UsageContractError(
                "unknown usage must not include numeric token fields"
            )
        if (
            self.source is UsageSource.PROVIDER_REPORTED
            and self.input_tokens is None
            and self.output_tokens is None
        ):
            raise UsageContractError(
                "provider-reported usage requires input or output tokens"
            )
        if self.cached_input_accounting is not None:
            allowed = {"simulated_assumption", "provider_reported"}
            if self.cached_input_accounting not in allowed:
                raise UsageContractError("cached_input_accounting is invalid")
        if self.request_id is not None:
            _required_text(self.request_id, "request_id")
        if self.run_id is not None:
            _required_text(self.run_id, "run_id")

    @classmethod
    def unknown(
        cls,
        *,
        provider: str,
        model: str,
        request_id: str | None = None,
        run_id: str | None = None,
    ) -> TokenUsage:
        return cls(
            provider=provider,
            model=model,
            source=UsageSource.UNKNOWN,
            request_id=request_id,
            run_id=run_id,
        )

    def to_payload(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "source": self.source.value,
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
            "request_id": self.request_id,
            "run_id": self.run_id,
            "cached_input_accounting": self.cached_input_accounting,
        }


@dataclass(frozen=True, slots=True)
class TokenUsageTotals:
    input_tokens: int
    cached_input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    total_tokens: int
    measured_completions: int
    estimated_completions: int
    unknown_completions: int

    simulated_cache_completions: int

    def to_payload(self) -> dict[str, Any]:
        return {
            "input_tokens": self.input_tokens,
            "cached_input_tokens": self.cached_input_tokens,
            "output_tokens": self.output_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "total_tokens": self.total_tokens,
            "measured_completions": self.measured_completions,
            "estimated_completions": self.estimated_completions,
            "unknown_completions": self.unknown_completions,
            "simulated_cache_completions": self.simulated_cache_completions,
        }


def aggregate_usage(records: tuple[TokenUsage, ...]) -> TokenUsageTotals:
    input_tokens = 0
    cached_input_tokens = 0
    output_tokens = 0
    reasoning_tokens = 0
    total_tokens = 0
    measured = 0
    estimated = 0
    unknown = 0
    simulated_cache = 0
    for record in records:
        if record.source is UsageSource.UNKNOWN:
            unknown += 1
            continue
        if record.source is UsageSource.PROVIDER_REPORTED:
            measured += 1
        else:
            estimated += 1
        if record.cached_input_accounting == "simulated_assumption":
            simulated_cache += 1
        input_tokens += record.input_tokens or 0
        cached_input_tokens += record.cached_input_tokens or 0
        output_tokens += record.output_tokens or 0
        reasoning_tokens += record.reasoning_tokens or 0
        if record.total_tokens is not None:
            total_tokens += record.total_tokens
        else:
            total_tokens += sum(
                value or 0
                for value in (
                    record.input_tokens,
                    record.cached_input_tokens,
                    record.output_tokens,
                    record.reasoning_tokens,
                )
            )
    return TokenUsageTotals(
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        total_tokens=total_tokens,
        measured_completions=measured,
        estimated_completions=estimated,
        unknown_completions=unknown,
        simulated_cache_completions=simulated_cache,
    )


def _non_negative_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise UsageContractError(f"{field_name} must be an integer")
    if value < 0:
        raise UsageContractError(f"{field_name} must be non-negative")
    return value


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise UsageContractError(f"{field_name} must be non-empty text")
    return value


def reject_non_finite_number(value: object, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise UsageContractError(f"{field_name} must be a finite number")
    if not math.isfinite(float(value)):
        raise UsageContractError(f"{field_name} must be finite")


__all__ = [
    "TokenUsage",
    "TokenUsageTotals",
    "UsageContractError",
    "UsageSource",
    "aggregate_usage",
    "reject_non_finite_number",
]
