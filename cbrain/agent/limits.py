"""Explicit in-run bounds for the foundation agent."""

from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RunLimits:
    """Smallest explicit bounds for one synchronous agent run."""

    max_task_chars: int = 16_384
    max_model_text_chars: int = 16_384
    max_observation_chars: int = 16_384
    max_context_messages: int = 32
    max_output_tokens: int = 1024

    def __post_init__(self) -> None:
        for field_name in (
            "max_task_chars",
            "max_model_text_chars",
            "max_observation_chars",
            "max_context_messages",
            "max_output_tokens",
        ):
            value = getattr(self, field_name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{field_name} must be a positive integer")


def finite_positive_seconds(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a positive number")
    seconds = float(value)
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"{field_name} must be finite and positive")
    return seconds


__all__ = ["RunLimits", "finite_positive_seconds"]
