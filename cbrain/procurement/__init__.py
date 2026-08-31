"""Procurement bots over shared FoundationAgent and governed ERP adapters.

Oracle, SAP, and SQL Server never enter the chat model. Replica extracts feed
untrusted knowledge; named GovernedTools perform live lookups and writes.
"""

from .bots import (
    BUYER_PROFILE,
    CATEGORY_MANAGER_PROFILE,
    VENDOR_ONBOARDING_PROFILE,
    independent_procurement_profiles,
)
from .erp import (
    CatalogRecord,
    ErpSystem,
    OpenPurchaseOrder,
    ProcurementError,
    ReplicaSource,
    StaticReplicaSource,
    VendorRecord,
    merge_replica_sources,
)
from .extracts import (
    documents_from_replica,
    load_json_extract,
)
from .proof import (
    PROCUREMENT_PROOF_SCHEMA,
    ProcurementProof,
    build_procurement_proof,
)

__all__ = [
    "BUYER_PROFILE",
    "CATEGORY_MANAGER_PROFILE",
    "CatalogRecord",
    "ErpSystem",
    "OpenPurchaseOrder",
    "PROCUREMENT_PROOF_SCHEMA",
    "ProcurementError",
    "ProcurementProof",
    "ReplicaSource",
    "StaticReplicaSource",
    "VENDOR_ONBOARDING_PROFILE",
    "VendorRecord",
    "build_procurement_proof",
    "documents_from_replica",
    "independent_procurement_profiles",
    "load_json_extract",
    "merge_replica_sources",
]
