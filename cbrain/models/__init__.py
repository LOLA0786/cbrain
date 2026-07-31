"""Credential-safe model routing and provider adapters."""

from .anthropic import AnthropicAdapter
from .contracts import (
    CompletionRequest,
    Message,
    MessageRole,
    ModelAdapter,
    ModelContractError,
    ModelCredentialError,
    ModelError,
    ModelOutput,
    ModelResponseError,
    ModelTransportError,
    TextOutput,
    ToolCall,
    ToolDefinition,
)
from .google import GoogleAdapter
from .openai_compatible import OpenAICompatibleAdapter
from .router import (
    ANTHROPIC_ROUTE,
    FIVE_PROVIDER_ROUTES,
    GOOGLE_ROUTE,
    OPENAI_ROUTE,
    RUNPOD_ROUTE,
    XAI_ROUTE,
    FiveProviderSettings,
    ModelRouter,
    ModelRoutingError,
    build_five_provider_router,
)
from .transport import (
    EnvironmentCredentialHeaders,
    HeadersProvider,
    HttpsJsonModelTransport,
    JsonModelTransport,
)

__all__ = [
    "ANTHROPIC_ROUTE",
    "FIVE_PROVIDER_ROUTES",
    "GOOGLE_ROUTE",
    "OPENAI_ROUTE",
    "RUNPOD_ROUTE",
    "XAI_ROUTE",
    "AnthropicAdapter",
    "CompletionRequest",
    "EnvironmentCredentialHeaders",
    "FiveProviderSettings",
    "GoogleAdapter",
    "HeadersProvider",
    "HttpsJsonModelTransport",
    "JsonModelTransport",
    "Message",
    "MessageRole",
    "ModelAdapter",
    "ModelContractError",
    "ModelCredentialError",
    "ModelError",
    "ModelOutput",
    "ModelResponseError",
    "ModelRouter",
    "ModelRoutingError",
    "ModelTransportError",
    "OpenAICompatibleAdapter",
    "TextOutput",
    "ToolCall",
    "ToolDefinition",
    "build_five_provider_router",
]
