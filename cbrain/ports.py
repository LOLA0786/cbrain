from __future__ import annotations

from collections.abc import Callable, Mapping
from typing import Any, Protocol

from .contracts import ActionIntent, GovernedExecution

ToolHandler = Callable[[Mapping[str, Any]], Any]


class PrivateVaultGateway(Protocol):
    """Trusted adapter backed by the pinned PrivateVault implementation."""

    def decide_and_execute(
        self,
        action: ActionIntent,
        handler: ToolHandler,
    ) -> GovernedExecution:
        """Authorize, dispatch once, witness, close, and return evidence state."""
