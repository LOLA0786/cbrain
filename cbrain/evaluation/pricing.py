"""Versioned pricing catalog for agent evaluation cost accounting."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from enum import StrEnum
from pathlib import Path
from typing import Any

from cbrain.models.usage import reject_non_finite_number


class PricingError(ValueError):
    """Pricing catalog payload is missing, inconsistent, or invalid."""


class ReasoningTokenTreatment(StrEnum):
    BILL_AS_OUTPUT = "bill_as_output"
    BILL_SEPARATELY = "bill_separately"
    IGNORE = "ignore"


@dataclass(frozen=True, slots=True)
class ModelPricing:
    provider: str
    model: str
    effective_date: str
    currency: str
    input_price_per_million: Decimal
    cached_input_price_per_million: Decimal
    output_price_per_million: Decimal
    reasoning_price_per_million: Decimal
    reasoning_token_treatment: ReasoningTokenTreatment
    source_url: str

    def pricing_key(self) -> tuple[str, str]:
        return (self.provider, self.model)

    def to_payload(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "effective_date": self.effective_date,
            "currency": self.currency,
            "input_price_per_million": str(self.input_price_per_million),
            "cached_input_price_per_million": str(self.cached_input_price_per_million),
            "output_price_per_million": str(self.output_price_per_million),
            "reasoning_price_per_million": str(self.reasoning_price_per_million),
            "reasoning_token_treatment": self.reasoning_token_treatment.value,
            "source_url": self.source_url,
        }


@dataclass(frozen=True, slots=True)
class PricingCatalog:
    schema_version: int
    catalog_id: str
    entries: tuple[ModelPricing, ...]

    def lookup(self, *, provider: str, model: str) -> ModelPricing | None:
        key = (provider, model)
        for entry in self.entries:
            if entry.pricing_key() == key:
                return entry
        return None

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "catalog_id": self.catalog_id,
            "entries": [entry.to_payload() for entry in self.entries],
        }


def default_pricing_catalog_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "pricing_catalog_v1.json"


def load_pricing_catalog(path: str | Path) -> PricingCatalog:
    raw_path = Path(path)
    try:
        payload = json.loads(raw_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PricingError(f"pricing catalog {raw_path} is unreadable") from exc
    if not isinstance(payload, dict):
        raise PricingError("pricing catalog must be a JSON object")
    return pricing_catalog_from_mapping(payload)


def pricing_catalog_from_mapping(payload: Mapping[str, Any]) -> PricingCatalog:
    schema_version = payload.get("schema_version")
    if schema_version != 1:
        raise PricingError(f"unsupported pricing schema version: {schema_version!r}")
    catalog_id = payload.get("catalog_id")
    if not isinstance(catalog_id, str) or not catalog_id.strip():
        raise PricingError("catalog_id must be non-empty")
    entries_raw = payload.get("entries")
    if not isinstance(entries_raw, list) or not entries_raw:
        raise PricingError("entries must be a non-empty list")
    entries = tuple(
        _parse_entry(item, index=index) for index, item in enumerate(entries_raw)
    )
    keys = [entry.pricing_key() for entry in entries]
    if len(keys) != len(set(keys)):
        raise PricingError("duplicate provider/model pricing entries")
    return PricingCatalog(
        schema_version=schema_version, catalog_id=catalog_id, entries=entries
    )


def _parse_entry(payload: object, *, index: int) -> ModelPricing:
    if not isinstance(payload, dict):
        raise PricingError(f"entry at index {index} must be an object")
    provider = _required_text(payload.get("provider"), f"entries[{index}].provider")
    model = _required_text(payload.get("model"), f"entries[{index}].model")
    effective_date = _required_text(
        payload.get("effective_date"), f"entries[{index}].effective_date"
    )
    currency = _required_text(payload.get("currency"), f"entries[{index}].currency")
    treatment_raw = payload.get("reasoning_token_treatment")
    if not isinstance(treatment_raw, str):
        raise PricingError(
            f"entries[{index}].reasoning_token_treatment must be a string"
        )
    try:
        treatment = ReasoningTokenTreatment(treatment_raw)
    except ValueError as exc:
        raise PricingError(
            f"entries[{index}].reasoning_token_treatment is invalid"
        ) from exc
    source_url = _required_text(
        payload.get("source_url"), f"entries[{index}].source_url"
    )
    return ModelPricing(
        provider=provider,
        model=model,
        effective_date=effective_date,
        currency=currency,
        input_price_per_million=_decimal_price(
            payload.get("input_price_per_million"),
            f"entries[{index}].input_price_per_million",
        ),
        cached_input_price_per_million=_decimal_price(
            payload.get("cached_input_price_per_million"),
            f"entries[{index}].cached_input_price_per_million",
        ),
        output_price_per_million=_decimal_price(
            payload.get("output_price_per_million"),
            f"entries[{index}].output_price_per_million",
        ),
        reasoning_price_per_million=_decimal_price(
            payload.get("reasoning_price_per_million"),
            f"entries[{index}].reasoning_price_per_million",
        ),
        reasoning_token_treatment=treatment,
        source_url=source_url,
    )


def _decimal_price(value: object, field_name: str) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        raise PricingError(f"{field_name} must be a decimal string")
    reject_non_finite_number(value, field_name) if isinstance(value, int) else None
    try:
        parsed = Decimal(str(value))
    except InvalidOperation as exc:
        raise PricingError(f"{field_name} must be a decimal string") from exc
    if parsed < 0:
        raise PricingError(f"{field_name} must be non-negative")
    return parsed


def _required_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PricingError(f"{field_name} must be non-empty text")
    return value


__all__ = [
    "ModelPricing",
    "PricingCatalog",
    "PricingError",
    "ReasoningTokenTreatment",
    "default_pricing_catalog_path",
    "load_pricing_catalog",
    "pricing_catalog_from_mapping",
]
