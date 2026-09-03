"""Package-level steel vendor catalog mirrored from tests/fixtures/vendors.json."""

from __future__ import annotations

from importlib.resources import files


def vendor_catalog_bytes() -> bytes:
    return files("cbrain.simulators.catalog").joinpath("vendors.json").read_bytes()


__all__ = ["vendor_catalog_bytes"]
