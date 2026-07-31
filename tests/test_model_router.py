from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest

from cbrain.models import (
    FIVE_PROVIDER_ROUTES,
    CompletionRequest,
    EnvironmentCredentialHeaders,
    FiveProviderSettings,
    Message,
    MessageRole,
    ModelCredentialError,
    ModelRouter,
    ModelRoutingError,
    TextOutput,
    build_five_provider_router,
)


class Adapter:
    provider = "fake"
    model = "fake-model"

    def complete(self, request: CompletionRequest) -> TextOutput:
        return TextOutput(request.messages[-1].content)


def test_router_uses_only_deployment_owned_route() -> None:
    router = ModelRouter({"primary": Adapter()})
    request = CompletionRequest((Message(MessageRole.USER, "hello"),))

    assert router.complete("primary", request) == TextOutput("hello")
    with pytest.raises(ModelRoutingError, match="unknown model route"):
        router.complete("model-selected-url", request)


def test_environment_headers_resolve_at_call_time_without_storing_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    headers = EnvironmentCredentialHeaders(
        "TEST_PROVIDER_KEY", "Authorization", "Bearer "
    )
    assert "secret-value" not in repr(headers)
    with pytest.raises(ModelCredentialError, match="unavailable"):
        headers()

    monkeypatch.setenv("TEST_PROVIDER_KEY", "secret-value")
    assert headers() == {"Authorization": "Bearer secret-value"}
    assert "secret-value" not in repr(headers)


def test_five_provider_builder_has_exact_routes_and_no_resolved_keys() -> None:
    router = build_five_provider_router(
        FiveProviderSettings(
            anthropic_model="anthropic-model",
            openai_model="openai-model",
            xai_model="xai-model",
            google_model="google-model",
            runpod_model="runpod-model",
            runpod_base_url="https://example.runpod.net",
        )
    )

    assert tuple(router.routes) == FIVE_PROVIDER_ROUTES
    assert {key: value.provider for key, value in router.routes.items()} == {
        "anthropic": "anthropic",
        "openai": "openai",
        "xai": "xai",
        "google": "google",
        "runpod": "runpod",
    }
    assert "API_KEY" not in repr(router.routes)


def test_five_provider_settings_read_only_model_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    values: Mapping[str, str] = {
        "CBRAIN_ANTHROPIC_MODEL": "a",
        "CBRAIN_OPENAI_MODEL": "o",
        "CBRAIN_XAI_MODEL": "x",
        "CBRAIN_GOOGLE_MODEL": "g",
        "CBRAIN_RUNPOD_MODEL": "r",
        "CBRAIN_RUNPOD_BASE_URL": "https://example.runpod.net",
    }
    for key, value in values.items():
        monkeypatch.setenv(key, value)

    settings = FiveProviderSettings.from_environment()

    assert settings.runpod_base_url == "https://example.runpod.net"
    assert not any("key" in item.casefold() for item in settings.__dataclass_fields__)


def test_runpod_endpoint_must_be_https() -> None:
    with pytest.raises(ValueError, match="HTTPS"):
        build_five_provider_router(
            FiveProviderSettings(
                "a",
                "o",
                "x",
                "g",
                "r",
                "http://runpod.invalid",
            )
        )


def test_adapter_protocol_shape_is_mapping_free() -> None:
    """Guard against accidentally placing HTTP headers in neutral requests."""

    request = CompletionRequest((Message(MessageRole.USER, "hello"),))
    payload: dict[str, Any] = {
        name: getattr(request, name) for name in request.__slots__
    }
    assert "headers" not in payload
    assert "credentials" not in payload
