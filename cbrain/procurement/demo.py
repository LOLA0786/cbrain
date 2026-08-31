"""Offline replica snapshots tagged by source ERP. Not live JDBC/RFC clients."""

from __future__ import annotations

from .erp import (
    CatalogRecord,
    ErpSystem,
    OpenPurchaseOrder,
    StaticReplicaSource,
    VendorRecord,
)


def demo_replica_sources() -> tuple[StaticReplicaSource, ...]:
    oracle = StaticReplicaSource(
        erp_system=ErpSystem.ORACLE,
        vendor_rows=(
            VendorRecord(
                vendor_id="vendor-oracle-1",
                name="Oracle Steel Co",
                email="oracle-steel@example.com",
                registered=True,
                source_system=ErpSystem.ORACLE,
            ),
            VendorRecord(
                vendor_id="vendor-ghost",
                name="Unlisted Broker",
                email="ghost@example.com",
                registered=False,
                source_system=ErpSystem.ORACLE,
            ),
        ),
        catalog_rows=(
            CatalogRecord(
                material_id="mat-steel-rod",
                description="Steel rod 20mm",
                source_system=ErpSystem.ORACLE,
            ),
        ),
        open_po_rows=(
            OpenPurchaseOrder(
                po_id="po-oracle-1",
                vendor_id="vendor-oracle-1",
                material_id="mat-steel-rod",
                amount_minor="8000",
                currency="USD",
                source_system=ErpSystem.ORACLE,
            ),
        ),
    )
    sap = StaticReplicaSource(
        erp_system=ErpSystem.SAP,
        vendor_rows=(
            VendorRecord(
                vendor_id="vendor-sap-1",
                name="SAP Metals GmbH",
                email="sap-metals@example.com",
                registered=True,
                source_system=ErpSystem.SAP,
            ),
        ),
        catalog_rows=(),
        open_po_rows=(),
    )
    sql_server = StaticReplicaSource(
        erp_system=ErpSystem.SQL_SERVER,
        vendor_rows=(
            VendorRecord(
                vendor_id="vendor-sql-1",
                name="SQL Mill LLC",
                email="sql-mill@example.com",
                registered=True,
                source_system=ErpSystem.SQL_SERVER,
            ),
        ),
        catalog_rows=(),
        open_po_rows=(),
    )
    return (oracle, sap, sql_server)


__all__ = ["demo_replica_sources"]
