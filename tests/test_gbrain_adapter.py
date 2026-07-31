from __future__ import annotations

import pytest

from cbrain.adapters.gbrain import (
    GBRAIN_READ_TOOL_CAPABILITIES,
    GBRAIN_UPSTREAM_COMMIT,
    GBRAIN_VERSION,
    GBrainConfigurationError,
    GBrainStdioConfig,
)


def config() -> GBrainStdioConfig:
    return GBrainStdioConfig(
        command="/opt/gbrain/bin/gbrain",
        home="/var/lib/cbrain/gbrain",
    )


def test_pins_real_gbrain_upstream() -> None:
    assert GBRAIN_VERSION == "0.42.67.0"
    assert GBRAIN_UPSTREAM_COMMIT == "c6dc0adf26a2d20df1147d2ec87c8922ca86d410"


def test_exposes_only_explicit_read_and_skill_tools() -> None:
    selected = set(config().selected_tools)

    assert len(selected) == 15
    assert selected == set(GBRAIN_READ_TOOL_CAPABILITIES)
    assert "get_page" in selected
    assert "query" in selected
    assert "get_skill" in selected

    assert "put_page" not in selected
    assert "delete_page" not in selected
    assert "purge_deleted_pages" not in selected
    assert "file_upload" not in selected
    assert "submit_job" not in selected
    assert "sources_remove" not in selected


def test_builds_real_hermes_stdio_mcp_configuration() -> None:
    server = config().hermes_server_config()

    assert server == {
        "command": "/opt/gbrain/bin/gbrain",
        "args": ["serve"],
        "env": {
            "GBRAIN_HOME": "/var/lib/cbrain/gbrain",
            "DATABASE_URL": "",
            "GBRAIN_DATABASE_URL": "",
        },
        "connect_timeout": 60.0,
        "enabled": True,
        "tools": {
            "include": list(GBRAIN_READ_TOOL_CAPABILITIES),
        },
    }


def test_returned_configuration_is_not_shared_mutable_state() -> None:
    first = config().hermes_server_config()
    first["tools"]["include"].append("delete_page")

    second = config().hermes_server_config()
    assert "delete_page" not in second["tools"]["include"]


def test_capability_map_is_default_deny() -> None:
    capabilities = config().capability_map()

    assert capabilities.resolve("get_page") == "memory.page.read"
    with pytest.raises(ValueError):
        capabilities.resolve("put_page")


@pytest.mark.parametrize(
    ("command", "home", "timeout"),
    [
        ("gbrain", "/var/lib/gbrain", 60.0),
        ("/opt/gbrain", "relative/gbrain", 60.0),
        ("/opt/gbrain", "/var/lib/gbrain", 0.0),
    ],
)
def test_rejects_unsafe_process_configuration(
    command: str,
    home: str,
    timeout: float,
) -> None:
    with pytest.raises(GBrainConfigurationError):
        GBrainStdioConfig(
            command=command,
            home=home,
            connect_timeout_seconds=timeout,
        )
