"""Tests for strict token usage accounting."""

from __future__ import annotations

import pytest

from cbrain.models.usage import (
    TokenUsage,
    UsageContractError,
    UsageSource,
    aggregate_usage,
    reject_non_finite_number,
)


def test_provider_reported_usage_requires_tokens() -> None:
    with pytest.raises(UsageContractError):
        TokenUsage(
            provider="openai",
            model="gpt-test",
            source=UsageSource.PROVIDER_REPORTED,
        )


def test_unknown_usage_has_no_numeric_fields() -> None:
    with pytest.raises(UsageContractError):
        TokenUsage(
            provider="openai",
            model="gpt-test",
            source=UsageSource.UNKNOWN,
            input_tokens=1,
        )


def test_rejects_bool_as_token_count() -> None:
    with pytest.raises(UsageContractError):
        TokenUsage(
            provider="openai",
            model="gpt-test",
            source=UsageSource.ESTIMATED,
            input_tokens=True,  # type: ignore[arg-type]
            output_tokens=1,
        )


def test_rejects_non_finite_numbers() -> None:
    with pytest.raises(UsageContractError):
        reject_non_finite_number(float("nan"), "value")


def test_aggregate_usage_tracks_sources() -> None:
    totals = aggregate_usage(
        (
            TokenUsage(
                provider="sequence",
                model="sequence-v1",
                source=UsageSource.ESTIMATED,
                input_tokens=10,
                output_tokens=5,
            ),
            TokenUsage.unknown(provider="sequence", model="sequence-v1"),
        )
    )
    assert totals.input_tokens == 10
    assert totals.estimated_completions == 1
    assert totals.unknown_completions == 1
