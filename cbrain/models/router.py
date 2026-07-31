"""Deployment-owned model routing for five provider families."""

from __future__ import annotations

import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from .anthropic import AnthropicAdapter
from .contracts import CompletionRequest, ModelAdapter, ModelOutput
from .google import GoogleAdapter
from .openai_compatible import OpenAICompatibleAdapter
from .transport import EnvironmentCredentialHeaders, HttpsJsonModelTransport

ANTHROPIC_ROUTE = "anthropic"
OPENAI_ROUTE = "openai"
XAI_ROUTE = "xai"
GOOGLE_ROUTE = "google"
RUNPOD_ROUTE = "runpod"
FIVE_PROVIDER_ROUTES = (
    ANTHROPIC_ROUTE,
    OPENAI_ROUTE,
    XAI_ROUTE,
    GOOGLE_ROUTE,
    RUNPOD_ROUTE,
)


class ModelRoutingError(RuntimeError):
    """A model route is missing, duplicated, or invalid."""


class ModelRouter:
    """Route by deployment-owned ID; model output cannot select a provider."""

    def __init__(self, routes: Mapping[str, ModelAdapter]) -> None:
        copied = dict(routes)
        if not copied:
            raise ValueError("at least one model route is required")
        for route_id, adapter in copied.items():
            if not route_id.strip():
                raise ValueError("model route IDs must be non-empty")
            if not hasattr(adapter, "complete"):
                raise ValueError(f"invalid adapter for route {route_id!r}")
        self._routes = MappingProxyType(copied)

    @property
    def routes(self) -> Mapping[str, ModelAdapter]:
        return self._routes

    def complete(self, route_id: str, request: CompletionRequest) -> ModelOutput:
        try:
            adapter = self._routes[route_id]
        except KeyError as exc:
            raise ModelRoutingError(f"unknown model route: {route_id}") from exc
        return adapter.complete(request)

    def require_routes(self, route_ids: Iterable[str]) -> None:
        required = tuple(route_ids)
        missing = sorted(set(required) - set(self._routes))
        if missing:
            raise ModelRoutingError(f"missing model routes: {missing}")


@dataclass(frozen=True, slots=True)
class FiveProviderSettings:
    anthropic_model: str
    openai_model: str
    xai_model: str
    google_model: str
    runpod_model: str
    runpod_base_url: str

    def __post_init__(self) -> None:
        for field_name in (
            "anthropic_model",
            "openai_model",
            "xai_model",
            "google_model",
            "runpod_model",
            "runpod_base_url",
        ):
            value = getattr(self, field_name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{field_name} must be non-empty")

    @classmethod
    def from_environment(cls) -> FiveProviderSettings:
        return cls(
            anthropic_model=_required_env("CBRAIN_ANTHROPIC_MODEL"),
            openai_model=_required_env("CBRAIN_OPENAI_MODEL"),
            xai_model=_required_env("CBRAIN_XAI_MODEL"),
            google_model=_required_env("CBRAIN_GOOGLE_MODEL"),
            runpod_model=_required_env("CBRAIN_RUNPOD_MODEL"),
            runpod_base_url=_required_env("CBRAIN_RUNPOD_BASE_URL"),
        )


def build_five_provider_router(settings: FiveProviderSettings) -> ModelRouter:
    """Build Anthropic, OpenAI, xAI, Google, and RunPod/vLLM routes.

    Model identifiers and the RunPod endpoint are explicit deployment inputs.
    Credentials remain environment names until each transport call begins.
    """

    return ModelRouter(
        {
            ANTHROPIC_ROUTE: AnthropicAdapter(
                model=settings.anthropic_model,
                transport=HttpsJsonModelTransport(base_url="https://api.anthropic.com"),
                headers_provider=EnvironmentCredentialHeaders(
                    "ANTHROPIC_API_KEY", "x-api-key"
                ),
            ),
            OPENAI_ROUTE: OpenAICompatibleAdapter(
                provider="openai",
                model=settings.openai_model,
                transport=HttpsJsonModelTransport(base_url="https://api.openai.com"),
                headers_provider=EnvironmentCredentialHeaders(
                    "OPENAI_API_KEY", "Authorization", "Bearer "
                ),
            ),
            XAI_ROUTE: OpenAICompatibleAdapter(
                provider="xai",
                model=settings.xai_model,
                transport=HttpsJsonModelTransport(base_url="https://api.x.ai"),
                headers_provider=EnvironmentCredentialHeaders(
                    "XAI_API_KEY", "Authorization", "Bearer "
                ),
            ),
            GOOGLE_ROUTE: GoogleAdapter(
                model=settings.google_model,
                transport=HttpsJsonModelTransport(
                    base_url="https://generativelanguage.googleapis.com"
                ),
                headers_provider=EnvironmentCredentialHeaders(
                    "GOOGLE_API_KEY", "x-goog-api-key"
                ),
            ),
            RUNPOD_ROUTE: OpenAICompatibleAdapter(
                provider="runpod",
                model=settings.runpod_model,
                transport=HttpsJsonModelTransport(base_url=settings.runpod_base_url),
                headers_provider=EnvironmentCredentialHeaders(
                    "RUNPOD_API_KEY", "Authorization", "Bearer "
                ),
            ),
        }
    )


def _required_env(name: str) -> str:
    value = os.environ.get(name, "")
    if not value.strip():
        raise ModelRoutingError(f"required model setting {name} is unavailable")
    return value


__all__ = [
    "ANTHROPIC_ROUTE",
    "FIVE_PROVIDER_ROUTES",
    "GOOGLE_ROUTE",
    "ModelRouter",
    "ModelRoutingError",
    "OPENAI_ROUTE",
    "RUNPOD_ROUTE",
    "FiveProviderSettings",
    "XAI_ROUTE",
    "build_five_provider_router",
]
