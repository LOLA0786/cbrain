"""Deployment-owned ERP replica records. Connection strings never appear here."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from cbrain.knowledge.contracts import FORBIDDEN_METADATA_KEYS, require_text


class ProcurementError(ValueError):
    """Replica extract or procurement adapter input is invalid."""


class ErpSystem(StrEnum):
    ORACLE = "oracle"
    SAP = "sap"
    SQL_SERVER = "sql_server"


_SECRET_MARKERS = frozenset(
    {
        "dsn=",
        "jdbc:",
        "password=",
        "pwd=",
        "private_key",
        "api_key",
        "rfc://",
        "odbc:",
    }
)


def _reject_secret_shaped(value: str, *, field_name: str) -> str:
    text = require_text(value, field_name)
    lowered = text.casefold()
    if lowered in FORBIDDEN_METADATA_KEYS:
        raise ProcurementError(f"{field_name} must not be a secret field name")
    if any(marker in lowered for marker in _SECRET_MARKERS):
        raise ProcurementError(f"{field_name} must not contain connection material")
    return text


@dataclass(frozen=True, slots=True)
class VendorRecord:
    vendor_id: str
    name: str
    email: str
    registered: bool
    source_system: ErpSystem

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "vendor_id",
            _reject_secret_shaped(self.vendor_id, field_name="vendor_id"),
        )
        object.__setattr__(
            self,
            "name",
            _reject_secret_shaped(self.name, field_name="name"),
        )
        email = _reject_secret_shaped(self.email, field_name="email")
        if "@" not in email or email.startswith("@"):
            raise ProcurementError("email must be a vendor mailbox, not a connection")
        object.__setattr__(self, "email", email)
        if not isinstance(self.registered, bool):
            raise ProcurementError("registered must be a bool")
        if not isinstance(self.source_system, ErpSystem):
            raise ProcurementError("source_system must be an ErpSystem")


@dataclass(frozen=True, slots=True)
class CatalogRecord:
    material_id: str
    description: str
    source_system: ErpSystem

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "material_id",
            _reject_secret_shaped(self.material_id, field_name="material_id"),
        )
        object.__setattr__(
            self,
            "description",
            _reject_secret_shaped(self.description, field_name="description"),
        )
        if not isinstance(self.source_system, ErpSystem):
            raise ProcurementError("source_system must be an ErpSystem")


@dataclass(frozen=True, slots=True)
class OpenPurchaseOrder:
    po_id: str
    vendor_id: str
    material_id: str
    amount_minor: str
    currency: str
    source_system: ErpSystem

    def __post_init__(self) -> None:
        object.__setattr__(
            self, "po_id", _reject_secret_shaped(self.po_id, field_name="po_id")
        )
        object.__setattr__(
            self,
            "vendor_id",
            _reject_secret_shaped(self.vendor_id, field_name="vendor_id"),
        )
        object.__setattr__(
            self,
            "material_id",
            _reject_secret_shaped(self.material_id, field_name="material_id"),
        )
        object.__setattr__(
            self,
            "amount_minor",
            _reject_secret_shaped(self.amount_minor, field_name="amount_minor"),
        )
        object.__setattr__(
            self,
            "currency",
            _reject_secret_shaped(self.currency, field_name="currency"),
        )
        if not isinstance(self.source_system, ErpSystem):
            raise ProcurementError("source_system must be an ErpSystem")


class ReplicaSource(Protocol):
    """Read-only ERP extract. Implementations must not accept model-chosen DSNs."""

    def system(self) -> ErpSystem: ...

    def vendors(self) -> tuple[VendorRecord, ...]: ...

    def catalog(self) -> tuple[CatalogRecord, ...]: ...

    def open_purchase_orders(self) -> tuple[OpenPurchaseOrder, ...]: ...


@dataclass(frozen=True, slots=True)
class StaticReplicaSource:
    """Snapshot already extracted from Oracle, SAP, or SQL Server by deployment ETL."""

    erp_system: ErpSystem
    vendor_rows: tuple[VendorRecord, ...]
    catalog_rows: tuple[CatalogRecord, ...]
    open_po_rows: tuple[OpenPurchaseOrder, ...]

    def system(self) -> ErpSystem:
        return self.erp_system

    def vendors(self) -> tuple[VendorRecord, ...]:
        return self.vendor_rows

    def catalog(self) -> tuple[CatalogRecord, ...]:
        return self.catalog_rows

    def open_purchase_orders(self) -> tuple[OpenPurchaseOrder, ...]:
        return self.open_po_rows


def merge_replica_sources(
    sources: Sequence[ReplicaSource],
) -> StaticReplicaSource:
    if not sources:
        raise ProcurementError("at least one replica source is required")
    vendors: list[VendorRecord] = []
    catalog: list[CatalogRecord] = []
    open_pos: list[OpenPurchaseOrder] = []
    seen_vendors: set[str] = set()
    for source in sources:
        for vendor in source.vendors():
            if vendor.vendor_id in seen_vendors:
                raise ProcurementError(f"duplicate vendor_id {vendor.vendor_id!r}")
            seen_vendors.add(vendor.vendor_id)
            vendors.append(vendor)
        catalog.extend(source.catalog())
        open_pos.extend(source.open_purchase_orders())
    return StaticReplicaSource(
        erp_system=sources[0].system(),
        vendor_rows=tuple(vendors),
        catalog_rows=tuple(catalog),
        open_po_rows=tuple(open_pos),
    )


def parse_json_boolean(value: object, *, field_name: str) -> bool:
    """Accept only JSON booleans. Strings, integers, and null fail closed."""

    if type(value) is not bool:
        raise ProcurementError(f"{field_name} must be a JSON boolean")
    return value


def mapping_without_secrets(payload: Mapping[str, object]) -> Mapping[str, object]:
    for key in payload:
        if str(key).casefold() in FORBIDDEN_METADATA_KEYS:
            raise ProcurementError("replica extract must not contain secret fields")
    return payload


def records_from_mapping(
    payload: Mapping[str, object], *, system: ErpSystem
) -> StaticReplicaSource:
    mapping_without_secrets(payload)
    vendors = tuple(
        VendorRecord(
            vendor_id=str(row["vendor_id"]),
            name=str(row["name"]),
            email=str(row["email"]),
            registered=_registered_from_row(row),
            source_system=system,
        )
        for row in _object_rows(payload.get("vendors"), field_name="vendors")
    )
    catalog = tuple(
        CatalogRecord(
            material_id=str(row["material_id"]),
            description=str(row["description"]),
            source_system=system,
        )
        for row in _object_rows(payload.get("catalog"), field_name="catalog")
    )
    open_pos = tuple(
        OpenPurchaseOrder(
            po_id=str(row["po_id"]),
            vendor_id=str(row["vendor_id"]),
            material_id=str(row["material_id"]),
            amount_minor=str(row["amount_minor"]),
            currency=str(row["currency"]),
            source_system=system,
        )
        for row in _object_rows(
            payload.get("open_purchase_orders"), field_name="open_purchase_orders"
        )
    )
    return StaticReplicaSource(
        erp_system=system,
        vendor_rows=vendors,
        catalog_rows=catalog,
        open_po_rows=open_pos,
    )


def _object_rows(value: object, *, field_name: str) -> Iterable[Mapping[str, object]]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise ProcurementError(f"{field_name} must be a list")
    rows: list[Mapping[str, object]] = []
    for item in value:
        if not isinstance(item, Mapping):
            raise ProcurementError(f"{field_name} entries must be objects")
        mapping_without_secrets(item)
        rows.append(item)
    return rows


def _registered_from_row(row: Mapping[str, object]) -> bool:
    if "registered" not in row:
        raise ProcurementError("registered must be a JSON boolean")
    return parse_json_boolean(row["registered"], field_name="registered")


__all__ = [
    "CatalogRecord",
    "ErpSystem",
    "OpenPurchaseOrder",
    "ProcurementError",
    "ReplicaSource",
    "StaticReplicaSource",
    "VendorRecord",
    "merge_replica_sources",
    "parse_json_boolean",
    "records_from_mapping",
]
