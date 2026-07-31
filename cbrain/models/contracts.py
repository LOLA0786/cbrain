"""Framework-neutral model contracts with credential-safe tool boundaries."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Protocol

_TOOL_NAME = re.compile(r"[A-Za-z0-9_-]{1,64}")
_SENSITIVE_COMPONENTS = frozenset(
    {
        "api_key",
        "apikey",
        "auth",
        "authorization",
        "bearer",
        "credential",
        "credentials",
        "password",
        "private_key",
        "secret",
        "token",
    }
)


class ModelError(RuntimeError):
    """Base failure for model routing and provider translation."""


class ModelContractError(ModelError):
    """A neutral request or output violated the model contract."""


class ModelCredentialError(ModelError):
    """A required credential was unavailable or malformed."""


class ModelTransportError(ModelError):
    """A provider could not be reached securely."""


class ModelResponseError(ModelError):
    """A provider response was malformed or ambiguous."""


class MessageRole(StrEnum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True, slots=True)
class Message:
    role: MessageRole
    content: str
    tool_call_id: str | None = None
    tool_name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.role, MessageRole):
            raise ModelContractError("message role is invalid")
        _required_text(self.content, "message content")
        if self.role is MessageRole.TOOL:
            _required_text(self.tool_call_id, "tool_call_id")
            _tool_name(self.tool_name, "tool_name")
        elif self.tool_call_id is not None or self.tool_name is not None:
            raise ModelContractError(
                "tool_call_id and tool_name are permitted only on tool messages"
            )


@dataclass(frozen=True, slots=True)
class ToolDefinition:
    name: str
    description: str
    _input_schema_json: bytes

    def __post_init__(self) -> None:
        _tool_name(self.name, "tool name")
        _required_text(self.description, "tool description")
        schema = _restore_object(self._input_schema_json, "tool input schema")
        if schema.get("type") != "object":
            raise ModelContractError("tool input schema must have type=object")
        _reject_sensitive_schema(schema, "tool input schema")

    @classmethod
    def capture(
        cls,
        *,
        name: str,
        description: str,
        input_schema: Mapping[str, Any],
    ) -> ToolDefinition:
        return cls(
            name=name,
            description=description,
            _input_schema_json=_canonical_object(input_schema, "tool input schema"),
        )

    @property
    def input_schema(self) -> dict[str, Any]:
        return _restore_object(self._input_schema_json, "tool input schema")


@dataclass(frozen=True, slots=True)
class CompletionRequest:
    messages: tuple[Message, ...]
    tools: tuple[ToolDefinition, ...] = ()
    max_output_tokens: int = 1024
    temperature: float = 0.0

    def __post_init__(self) -> None:
        if not self.messages:
            raise ModelContractError("completion requires at least one message")
        if any(not isinstance(item, Message) for item in self.messages):
            raise ModelContractError("messages must contain Message objects")
        if any(not isinstance(item, ToolDefinition) for item in self.tools):
            raise ModelContractError("tools must contain ToolDefinition objects")
        names = [item.name for item in self.tools]
        if len(names) != len(set(names)):
            raise ModelContractError("tool names must be unique")
        if (
            isinstance(self.max_output_tokens, bool)
            or not isinstance(self.max_output_tokens, int)
            or self.max_output_tokens < 1
        ):
            raise ModelContractError("max_output_tokens must be positive")
        if (
            isinstance(self.temperature, bool)
            or not isinstance(self.temperature, (int, float))
            or not math.isfinite(float(self.temperature))
            or not 0 <= float(self.temperature) <= 2
        ):
            raise ModelContractError("temperature must be finite and between 0 and 2")


@dataclass(frozen=True, slots=True)
class ToolCall:
    call_id: str
    name: str
    _arguments_json: bytes

    def __post_init__(self) -> None:
        _required_text(self.call_id, "tool call ID")
        _tool_name(self.name, "tool call name")
        arguments = _restore_object(self._arguments_json, "tool arguments")
        _reject_sensitive_keys(arguments, "tool arguments")

    @classmethod
    def capture(
        cls,
        *,
        call_id: str,
        name: str,
        arguments: Mapping[str, Any],
    ) -> ToolCall:
        return cls(
            call_id=call_id,
            name=name,
            _arguments_json=_canonical_object(arguments, "tool arguments"),
        )

    @property
    def arguments(self) -> dict[str, Any]:
        return _restore_object(self._arguments_json, "tool arguments")


@dataclass(frozen=True, slots=True)
class TextOutput:
    text: str

    def __post_init__(self) -> None:
        _required_text(self.text, "model text")


type ModelOutput = ToolCall | TextOutput


class ModelAdapter(Protocol):
    @property
    def provider(self) -> str: ...

    @property
    def model(self) -> str: ...

    def complete(self, request: CompletionRequest) -> ModelOutput:
        """Return exactly one tool call or one text response."""


def required_mapping(value: object, path: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ModelResponseError(f"{path} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ModelResponseError(f"{path} keys must be strings")
    return value


def required_list(value: object, path: str) -> list[Any]:
    if not isinstance(value, list):
        raise ModelResponseError(f"{path} must be an array")
    return value


def required_response_text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelResponseError(f"{path} must be non-empty text")
    return value


def parse_arguments(value: object, path: str) -> dict[str, Any]:
    if isinstance(value, str):
        try:
            decoded: object = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ModelResponseError(f"{path} is not valid JSON") from exc
    else:
        decoded = value
    mapping = required_mapping(decoded, path)
    try:
        call = ToolCall.capture(
            call_id="validation", name="validation", arguments=mapping
        )
    except ModelContractError as exc:
        raise ModelResponseError(str(exc)) from exc
    return call.arguments


def _required_text(value: object, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ModelContractError(f"{path} must be non-empty text")
    return value


def _tool_name(value: object, path: str) -> str:
    text = _required_text(value, path)
    if _TOOL_NAME.fullmatch(text) is None:
        raise ModelContractError(
            f"{path} must contain only letters, digits, underscores, or hyphens"
        )
    return text


def _canonical_object(value: Mapping[str, Any], path: str) -> bytes:
    if not isinstance(value, Mapping):
        raise ModelContractError(f"{path} must be a mapping")
    _validate_json(value, path, set())
    try:
        return json.dumps(
            dict(value),
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise ModelContractError(f"{path} must contain finite JSON values") from exc


def _restore_object(value: bytes, path: str) -> dict[str, Any]:
    if not isinstance(value, bytes):
        raise ModelContractError(f"{path} snapshot must be bytes")
    try:
        restored: object = json.loads(value)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ModelContractError(f"{path} is not valid JSON") from exc
    if not isinstance(restored, dict):
        raise ModelContractError(f"{path} must be an object")
    return restored


def _validate_json(value: Any, path: str, ancestors: set[int]) -> None:
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ModelContractError(f"{path} contains a non-finite number")
        return
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in ancestors:
            raise ModelContractError(f"{path} must not contain cycles")
        ancestors.add(identity)
        try:
            for key, child in value.items():
                if not isinstance(key, str):
                    raise ModelContractError(f"{path} keys must be strings")
                _validate_json(child, f"{path}.{key}", ancestors)
        finally:
            ancestors.remove(identity)
        return
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        identity = id(value)
        if identity in ancestors:
            raise ModelContractError(f"{path} must not contain cycles")
        ancestors.add(identity)
        try:
            for child in value:
                _validate_json(child, f"{path}[]", ancestors)
        finally:
            ancestors.remove(identity)
        return
    raise ModelContractError(f"{path} contains a non-JSON value")


def _normalized_components(value: str) -> set[str]:
    snake_case = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", value)
    normalized = re.sub(r"[^a-z0-9]+", "_", snake_case.casefold()).strip("_")
    components = set(normalized.split("_")) if normalized else set()
    components.add(normalized)
    return components


def _is_sensitive_name(value: str) -> bool:
    components = _normalized_components(value)
    if components & _SENSITIVE_COMPONENTS:
        return True
    return any(item in components for item in ("api_key", "private_key"))


def _reject_sensitive_schema(value: Any, path: str) -> None:
    if isinstance(value, Mapping):
        properties = value.get("properties")
        if properties is not None:
            if not isinstance(properties, Mapping):
                raise ModelContractError(f"{path}.properties must be an object")
            for name in properties:
                if not isinstance(name, str):
                    raise ModelContractError(f"{path}.properties keys must be strings")
                if _is_sensitive_name(name):
                    raise ModelContractError(
                        f"{path} contains credential-shaped property {name!r}"
                    )
        required = value.get("required")
        if required is not None:
            if not isinstance(required, list) or any(
                not isinstance(item, str) for item in required
            ):
                raise ModelContractError(f"{path}.required must be a string array")
            for name in required:
                if _is_sensitive_name(name):
                    raise ModelContractError(
                        f"{path} requires credential-shaped property {name!r}"
                    )
        for child in value.values():
            _reject_sensitive_schema(child, path)
    elif isinstance(value, list):
        for child in value:
            _reject_sensitive_schema(child, path)


def _reject_sensitive_keys(value: Any, path: str) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ModelContractError(f"{path} keys must be strings")
            if _is_sensitive_name(key):
                raise ModelContractError(
                    f"{path} contains credential-shaped key {key!r}"
                )
            _reject_sensitive_keys(child, f"{path}.{key}")
    elif isinstance(value, list):
        for child in value:
            _reject_sensitive_keys(child, f"{path}[]")


__all__ = [
    "CompletionRequest",
    "Message",
    "MessageRole",
    "ModelAdapter",
    "ModelContractError",
    "ModelCredentialError",
    "ModelError",
    "ModelOutput",
    "ModelResponseError",
    "ModelTransportError",
    "TextOutput",
    "ToolCall",
    "ToolDefinition",
    "parse_arguments",
    "required_list",
    "required_mapping",
    "required_response_text",
]
