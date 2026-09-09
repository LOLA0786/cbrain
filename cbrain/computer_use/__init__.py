"""Governed computer / browser surface.

Agents may operate web pages the way a human does — observe, click, type,
submit — but destinations and credentials never enter model context, and every
consequential act enters ``GovernedRuntime`` (via FoundationAgent or
``GovernedComputerSession``). Observation snapshots never include raw URLs or
screenshot bytes — only digests.

Production assemblies must use ``RemoteComputerBackend`` against an
out-of-process worker. ``DevOnlyBrowserBackend`` is DEV_ONLY.
"""

from __future__ import annotations

from .catalog import PageAlias, PageCatalog, PageCatalogError
from .client import RemoteComputerBackend
from .session import (
    ComputerAct,
    ComputerObservation,
    ComputerUseError,
    GovernedComputerSession,
)
from .surface import (
    ComputerBackend,
    DevOnlyBrowserBackend,
    require_production_computer_backend,
)
from .tools import ComputerToolBridge, computer_tools

__all__ = [
    "ComputerAct",
    "ComputerBackend",
    "ComputerObservation",
    "ComputerToolBridge",
    "ComputerUseError",
    "DevOnlyBrowserBackend",
    "GovernedComputerSession",
    "PageAlias",
    "PageCatalog",
    "PageCatalogError",
    "RemoteComputerBackend",
    "computer_tools",
    "require_production_computer_backend",
]
