"""Tests for evaluation pricing and cost accounting."""

from __future__ import annotations

from decimal import Decimal

import pytest

from cbrain.evaluation.cost import aggregate_costs, cost_for_usage, cost_formula_text
from cbrain.evaluation.pricing import PricingError, load_pricing_catalog
from cbrain.models.usage import TokenUsage, UsageSource


def test_fixture_pricing_catalog_loads() -> None:
    catalog = load_pricing_catalog("cbrain/evaluation/data/pricing_catalog_v1.json")
    entry = catalog.lookup(provider="sequence", model="sequence-v1")
    assert entry is not None
    assert entry.currency == "USD"
    assert entry.source_url.startswith("https://")


def test_unknown_pricing_yields_cost_unknown() -> None:
    catalog = load_pricing_catalog("cbrain/evaluation/data/pricing_catalog_v1.json")
    usage = TokenUsage(
        provider="missing",
        model="missing-model",
        source=UsageSource.ESTIMATED,
        input_tokens=1000,
        output_tokens=200,
    )
    cost = cost_for_usage(usage, catalog=catalog)
    assert cost.status == "cost_unknown"
    assert cost.total_cost is None


def test_cost_formula_uses_decimal_math() -> None:
    catalog = load_pricing_catalog("cbrain/evaluation/data/pricing_catalog_v1.json")
    usage = TokenUsage(
        provider="sequence",
        model="sequence-v1",
        source=UsageSource.ESTIMATED,
        input_tokens=1_000_000,
        output_tokens=0,
    )
    cost = cost_for_usage(usage, catalog=catalog)
    assert cost.total_cost == Decimal("0.50")
    assert "cost_unknown" in cost_formula_text()


def test_unknown_usage_never_costs_zero() -> None:
    catalog = load_pricing_catalog("cbrain/evaluation/data/pricing_catalog_v1.json")
    cost = aggregate_costs(
        (TokenUsage.unknown(provider="sequence", model="sequence-v1"),),
        catalog=catalog,
    )
    assert cost.status == "cost_unknown"
    assert cost.total_cost is None


def test_rejects_invalid_pricing_payload() -> None:
    with pytest.raises(PricingError):
        load_pricing_catalog("/dev/null/does-not-exist.json")
