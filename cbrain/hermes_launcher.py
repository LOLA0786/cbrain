from __future__ import annotations

import importlib
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from .adapters.hermes import HermesPreToolDecisionHook

_CONFIGURATION_EXIT = 78
_REQUIRED_PLUGIN = "cbrain_guard"
_REQUIRED_HOOK = "pre_tool_call"
_FORBIDDEN_ARGUMENTS = frozenset(
    {
        "--ignore-rules",
        "--ignore-user-config",
        "--safe-mode",
        "--yolo",
    }
)


class HermesStartupError(RuntimeError):
    """Raised when Hermes cannot prove CBrain is its mandatory first guard."""


def validate_runtime_arguments(arguments: Sequence[str]) -> None:
    for argument in arguments:
        option = argument.split("=", 1)[0]
        if option in _FORBIDDEN_ARGUMENTS:
            raise HermesStartupError(f"Hermes runtime option {option!r} is forbidden")

    if "plugins" in arguments:
        raise HermesStartupError(
            "Hermes plugin administration is forbidden through cbrain-hermes"
        )


def verify_required_hook() -> None:
    """Load plugins and prove CBrain owns the first pre-tool callback."""

    try:
        plugins_module = importlib.import_module("hermes_cli.plugins")
    except ImportError as exc:
        raise HermesStartupError("Hermes is not installed") from exc

    discover = getattr(plugins_module, "discover_plugins", None)
    get_manager = getattr(plugins_module, "get_plugin_manager", None)
    if not callable(discover) or not callable(get_manager):
        raise HermesStartupError("Hermes plugin API is incompatible")

    discover(force=True)
    manager: Any = get_manager()

    loaded_plugins = getattr(manager, "_plugins", None)
    if not isinstance(loaded_plugins, Mapping):
        raise HermesStartupError("Hermes plugin registry is unavailable")
    if _REQUIRED_PLUGIN not in loaded_plugins:
        raise HermesStartupError("mandatory cbrain_guard plugin is not loaded")

    hooks = getattr(manager, "_hooks", None)
    if not isinstance(hooks, Mapping):
        raise HermesStartupError("Hermes hook registry is unavailable")

    callbacks = hooks.get(_REQUIRED_HOOK)
    if not isinstance(callbacks, list) or not callbacks:
        raise HermesStartupError("mandatory CBrain pre-tool hook is absent")

    callback = callbacks[0]
    owner = getattr(callback, "__self__", None)
    function = getattr(callback, "__func__", None)
    if not isinstance(owner, HermesPreToolDecisionHook):
        raise HermesStartupError("CBrain is not the first pre-tool guard")
    if function is not HermesPreToolDecisionHook.pre_tool_call:
        raise HermesStartupError("unexpected CBrain pre-tool callback")


def _run_hermes() -> int:
    try:
        main_module = importlib.import_module("hermes_cli.main")
    except ImportError as exc:
        raise HermesStartupError("Hermes CLI entrypoint is unavailable") from exc

    entrypoint = getattr(main_module, "main", None)
    if not callable(entrypoint):
        raise HermesStartupError("Hermes CLI entrypoint is incompatible")

    result = entrypoint()
    return result if type(result) is int else 0


def main() -> int:
    try:
        validate_runtime_arguments(sys.argv[1:])
        verify_required_hook()
        return _run_hermes()
    except HermesStartupError as exc:
        print(f"CBRAIN STARTUP REFUSED: {exc}", file=sys.stderr)
        return _CONFIGURATION_EXIT


__all__ = [
    "HermesStartupError",
    "main",
    "validate_runtime_arguments",
    "verify_required_hook",
]
