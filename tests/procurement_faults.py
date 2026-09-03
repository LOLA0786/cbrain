"""Test-only fault injection for procurement order simulators.

Never imported by cbrain/. The hook is installed onto a live simulator instance
after construction so production OrderSimulator ships with the drop path off.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from cbrain.simulators.contracts import EffectReceipt
from cbrain.simulators.orders import PROCUREMENT_PO_CREATE, OrderSimulator


class DroppedSimulatorResponse(RuntimeError):
    """The target committed the effect, then the response was dropped."""


def inject_drop_po_create_response(simulator: OrderSimulator) -> None:
    def execute(
        capability: str,
        *,
        request_id: str,
        idempotency_key: str,
        arguments: Mapping[str, Any],
    ) -> EffectReceipt:
        receipt = OrderSimulator.execute(
            simulator,
            capability,
            request_id=request_id,
            idempotency_key=idempotency_key,
            arguments=arguments,
        )
        if capability == PROCUREMENT_PO_CREATE:
            raise DroppedSimulatorResponse("po create response dropped after commit")
        return receipt

    simulator.execute = execute  # type: ignore[method-assign]
