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
    ProviderContinuation,
    TextOutput,
    ToolCall,
    ToolDefinition,
    validate_history,
)
from .google import GoogleAdapter
from .instrumented import CompletionObservation, InstrumentedModelAdapter
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
from .usage import TokenUsage, TokenUsageTotals, UsageContractError, UsageSource

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
    "InstrumentedModelAdapter",
    "CompletionObservation",
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
    "ProviderContinuation",
    "TextOutput",
    "ToolCall",
    "ToolDefinition",
    "TokenUsage",
    "TokenUsageTotals",
    "UsageContractError",
    "UsageSource",
    "build_five_provider_router",
    "validate_history",
]
