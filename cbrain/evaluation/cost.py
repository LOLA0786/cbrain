"""Cost calculation from token usage and pricing catalogs."""

from __future__ import annotations

from dataclasses import dataclass
from decimal import Decimal
from typing import Any

from cbrain.models.usage import (
    TokenUsage,
    TokenUsageTotals,
    UsageSource,
)

from .pricing import ModelPricing, PricingCatalog, ReasoningTokenTreatment


class CostError(ValueError):
    """Cost could not be calculated safely."""


@dataclass(frozen=True, slots=True)
class CostBreakdown:
    currency: str | None
    input_cost: Decimal | None
    cached_input_cost: Decimal | None
    output_cost: Decimal | None
    reasoning_cost: Decimal | None
    total_cost: Decimal | None
    known_cost_subtotal: Decimal | None
    status: str
    pricing_source_url: str | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "currency": self.currency,
            "input_cost": _decimal_text(self.input_cost),
            "cached_input_cost": _decimal_text(self.cached_input_cost),
            "output_cost": _decimal_text(self.output_cost),
            "reasoning_cost": _decimal_text(self.reasoning_cost),
            "total_cost": _decimal_text(self.total_cost),
            "known_cost_subtotal": _decimal_text(self.known_cost_subtotal),
            "status": self.status,
            "pricing_source_url": self.pricing_source_url,
        }


_MILLION = Decimal("1000000")
_INCOMPLETE_COST_STATUSES = frozenset({"partial_unknown", "cost_unknown"})


def cost_is_complete(status: str) -> bool:
    return status not in _INCOMPLETE_COST_STATUSES


def cost_for_usage(
    usage: TokenUsage,
    *,
    catalog: PricingCatalog,
) -> CostBreakdown:
    if usage.source is UsageSource.UNKNOWN:
        return CostBreakdown(
            currency=None,
            input_cost=None,
            cached_input_cost=None,
            output_cost=None,
            reasoning_cost=None,
            total_cost=None,
            known_cost_subtotal=None,
            status="cost_unknown",
        )
    pricing = catalog.lookup(provider=usage.provider, model=usage.model)
    if pricing is None:
        return CostBreakdown(
            currency=None,
            input_cost=None,
            cached_input_cost=None,
            output_cost=None,
            reasoning_cost=None,
            total_cost=None,
            known_cost_subtotal=None,
            status="cost_unknown",
        )
    if usage.input_tokens is None and usage.output_tokens is None:
        return CostBreakdown(
            currency=pricing.currency,
            input_cost=None,
            cached_input_cost=None,
            output_cost=None,
            reasoning_cost=None,
            total_cost=None,
            known_cost_subtotal=None,
            status="cost_unknown",
            pricing_source_url=pricing.source_url,
        )
    input_cost = _line_item(usage.input_tokens, pricing.input_price_per_million)
    cached_cost = _line_item(
        usage.cached_input_tokens, pricing.cached_input_price_per_million
    )
    output_cost = _line_item(usage.output_tokens, pricing.output_price_per_million)
    reasoning_cost = _reasoning_cost(usage, pricing)
    total = sum(
        (
            value
            for value in (input_cost, cached_cost, output_cost, reasoning_cost)
            if value is not None
        ),
        Decimal("0"),
    )
    return CostBreakdown(
        currency=pricing.currency,
        input_cost=input_cost,
        cached_input_cost=cached_cost,
        output_cost=output_cost,
        reasoning_cost=reasoning_cost,
        total_cost=total,
        known_cost_subtotal=total,
        status="measured"
        if usage.source is UsageSource.PROVIDER_REPORTED
        else "estimated",
        pricing_source_url=pricing.source_url,
    )


def aggregate_costs(
    usages: tuple[TokenUsage, ...],
    *,
    catalog: PricingCatalog,
) -> CostBreakdown:
    if not usages:
        return CostBreakdown(
            currency=None,
            input_cost=Decimal("0"),
            cached_input_cost=Decimal("0"),
            output_cost=Decimal("0"),
            reasoning_cost=Decimal("0"),
            total_cost=Decimal("0"),
            known_cost_subtotal=Decimal("0"),
            status="zero_inference",
        )
    currency: str | None = None
    input_cost = Decimal("0")
    cached_cost = Decimal("0")
    output_cost = Decimal("0")
    reasoning_cost = Decimal("0")
    total_cost = Decimal("0")
    has_unknown = False
    has_estimated = False
    has_measured = False
    source_url: str | None = None
    for usage in usages:
        item = cost_for_usage(usage, catalog=catalog)
        if item.status == "cost_unknown":
            has_unknown = True
            continue
        if item.currency is not None:
            currency = item.currency
        source_url = item.pricing_source_url or source_url
        if item.status == "estimated":
            has_estimated = True
        if item.status == "measured":
            has_measured = True
        input_cost += item.input_cost or Decimal("0")
        cached_cost += item.cached_input_cost or Decimal("0")
        output_cost += item.output_cost or Decimal("0")
        reasoning_cost += item.reasoning_cost or Decimal("0")
        total_cost += item.total_cost or Decimal("0")
    if has_unknown and total_cost == 0:
        return CostBreakdown(
            currency=currency,
            input_cost=None,
            cached_input_cost=None,
            output_cost=None,
            reasoning_cost=None,
            total_cost=None,
            known_cost_subtotal=None,
            status="cost_unknown",
            pricing_source_url=source_url,
        )
    if has_unknown:
        status = "partial_unknown"
        return CostBreakdown(
            currency=currency,
            input_cost=input_cost,
            cached_input_cost=cached_cost,
            output_cost=output_cost,
            reasoning_cost=reasoning_cost,
            total_cost=None,
            known_cost_subtotal=total_cost,
            status=status,
            pricing_source_url=source_url,
        )
    if has_estimated and has_measured:
        status = "mixed"
    elif has_estimated:
        status = "estimated"
    else:
        status = "measured"
    return CostBreakdown(
        currency=currency,
        input_cost=input_cost,
        cached_input_cost=cached_cost,
        output_cost=output_cost,
        reasoning_cost=reasoning_cost,
        total_cost=total_cost,
        known_cost_subtotal=total_cost,
        status=status,
        pricing_source_url=source_url,
    )


def cost_per_successful_task(
    *,
    total_cost: Decimal | None,
    successful_tasks: int,
    cost_status: str,
) -> Decimal | None:
    if not cost_is_complete(cost_status):
        return None
    if total_cost is None or successful_tasks <= 0:
        return None
    return total_cost / Decimal(successful_tasks)


def blended_cost_per_million_tokens(
    totals: TokenUsageTotals,
    *,
    total_cost: Decimal | None,
    cost_status: str,
) -> Decimal | None:
    if not cost_is_complete(cost_status):
        return None
    if total_cost is None or totals.total_tokens <= 0:
        return None
    return (total_cost / Decimal(totals.total_tokens)) * _MILLION


def cost_formula_text() -> str:
    return (
        "input_cost = (input_tokens / 1_000_000) * input_price_per_million; "
        "cached_input_cost = (cached_input_tokens / 1_000_000) "
        "* cached_input_price_per_million; "
        "output_cost = (output_tokens / 1_000_000) * output_price_per_million; "
        "reasoning_cost follows reasoning_token_treatment "
        "(bill_as_output | bill_separately | ignore); "
        "total_cost = input_cost + cached_input_cost + output_cost + reasoning_cost; "
        "unknown usage or missing pricing yields cost_unknown, never zero."
    )


def _line_item(token_count: int | None, price_per_million: Decimal) -> Decimal | None:
    if token_count is None:
        return None
    return (Decimal(token_count) / _MILLION) * price_per_million


def _reasoning_cost(usage: TokenUsage, pricing: ModelPricing) -> Decimal | None:
    if usage.reasoning_tokens is None or usage.reasoning_tokens == 0:
        return Decimal("0")
    if pricing.reasoning_token_treatment is ReasoningTokenTreatment.IGNORE:
        return Decimal("0")
    if pricing.reasoning_token_treatment is ReasoningTokenTreatment.BILL_AS_OUTPUT:
        return _line_item(usage.reasoning_tokens, pricing.output_price_per_million)
    return _line_item(usage.reasoning_tokens, pricing.reasoning_price_per_million)


def _decimal_text(value: Decimal | None) -> str | None:
    if value is None:
        return None
    return format(value, "f")


__all__ = [
    "CostBreakdown",
    "CostError",
    "aggregate_costs",
    "blended_cost_per_million_tokens",
    "cost_for_usage",
    "cost_formula_text",
    "cost_is_complete",
    "cost_per_successful_task",
]
