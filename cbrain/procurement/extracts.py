"""Turn ERP replica extracts into untrusted knowledge documents."""

from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType

from cbrain.knowledge.contracts import SourceDocument

from .erp import (
    ErpSystem,
    ProcurementError,
    ReplicaSource,
    StaticReplicaSource,
    records_from_mapping,
)

COLLECTION_VENDORS = "procurement-vendors"
COLLECTION_POS = "procurement-open-pos"
COLLECTION_CATALOG = "procurement-catalog"


def load_json_extract(path: Path, *, system: ErpSystem) -> StaticReplicaSource:
    """Load a deployment-owned replica file. The path is never a tool argument."""

    if not isinstance(path, Path):
        raise ProcurementError("extract path must be a filesystem Path")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ProcurementError("replica extract could not be read") from exc
    if not isinstance(payload, dict):
        raise ProcurementError("replica extract must be a JSON object")
    return records_from_mapping(payload, system=system)


def documents_from_replica(
    source: ReplicaSource,
    *,
    tenant_id: str,
    principal_id: str,
    created_at: float,
) -> tuple[SourceDocument, ...]:
    documents: list[SourceDocument] = []
    system = source.system().value
    for vendor in source.vendors():
        documents.append(
            SourceDocument(
                tenant_id=tenant_id,
                collection_id=COLLECTION_VENDORS,
                source_id=f"{system}-vendor-{vendor.vendor_id}",
                media_type="text/plain",
                text=(
                    f"Vendor {vendor.vendor_id} named {vendor.name} is "
                    f"{'registered' if vendor.registered else 'unregistered'} "
                    f"in {system}."
                ),
                acl_principals=frozenset({principal_id}),
                created_at=created_at,
                provenance=f"{system}-vendor-master",
                metadata=MappingProxyType(
                    {
                        "source_system": system,
                        "vendor_id": vendor.vendor_id,
                        "registered": "true" if vendor.registered else "false",
                    }
                ),
            )
        )
    for item in source.catalog():
        documents.append(
            SourceDocument(
                tenant_id=tenant_id,
                collection_id=COLLECTION_CATALOG,
                source_id=f"{system}-material-{item.material_id}",
                media_type="text/plain",
                text=f"Material {item.material_id}: {item.description} from {system}.",
                acl_principals=frozenset({principal_id}),
                created_at=created_at,
                provenance=f"{system}-material-master",
                metadata=MappingProxyType(
                    {
                        "source_system": system,
                        "material_id": item.material_id,
                    }
                ),
            )
        )
    for order in source.open_purchase_orders():
        documents.append(
            SourceDocument(
                tenant_id=tenant_id,
                collection_id=COLLECTION_POS,
                source_id=f"{system}-po-{order.po_id}",
                media_type="text/plain",
                text=(
                    f"Open purchase order {order.po_id} for vendor {order.vendor_id} "
                    f"material {order.material_id} amount {order.amount_minor} "
                    f"{order.currency} from {system}."
                ),
                acl_principals=frozenset({principal_id}),
                created_at=created_at,
                provenance=f"{system}-open-po",
                metadata=MappingProxyType(
                    {
                        "source_system": system,
                        "po_id": order.po_id,
                        "vendor_id": order.vendor_id,
                    }
                ),
            )
        )
    if not documents:
        raise ProcurementError("replica extract produced no documents")
    return tuple(documents)


__all__ = [
    "COLLECTION_CATALOG",
    "COLLECTION_POS",
    "COLLECTION_VENDORS",
    "documents_from_replica",
    "load_json_extract",
]
