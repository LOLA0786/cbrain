"""Governed computer / browser surface.

Agents may operate web pages the way a human does — observe, click, type,
submit — but destinations and credentials never enter model context, and every
consequential act enters ``GovernedRuntime``. Observation is read-only against
an already-resolved page; navigation and mutation are tool intents.

Production assemblies must not put a browser dialer inside the agent process
toward business destinations. The backend protocol is the seam; in-process
backends are DEV_ONLY (same rule as ``InProcessDispatchTransport``).
"""

from __future__ import annotations

from .catalog import PageAlias, PageCatalog, PageCatalogError
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

__all__ = [
    "ComputerAct",
    "ComputerBackend",
    "ComputerObservation",
    "ComputerUseError",
    "DevOnlyBrowserBackend",
    "GovernedComputerSession",
    "PageAlias",
    "PageCatalog",
    "PageCatalogError",
    "require_production_computer_backend",
]
