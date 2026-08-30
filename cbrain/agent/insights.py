"""Proposal-only insights and human-approved personalization.

Learning signals, the Insights engine, and compiled profiles never authorize
tool dispatch. Every consequential ``ActionIntent`` still enters
``GovernedRuntime`` and PrivateVault.
"""

from __future__ import annotations

import hashlib
import html
import json
import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType
from typing import Any, Protocol

from cbrain.contracts import ExecutionStatus

from .contracts import RunEventKind, RunResult, RunStatus
from .durable import profile_fingerprint
from .profile import AgentProfile

SIGNAL_SCHEMA_VERSION = 1
APPROVAL_SCHEMA_VERSION = 1
REPORT_SCHEMA = "cbrain-insights-report/v1"
DEFAULT_WINDOW_DAYS = 30
MIN_GUIDANCE_OCCURRENCES = 2
MAX_FEEDBACK_CHARS = 4_000
SECONDS_PER_DAY = 86_400

FORBIDDEN_SIGNAL_KEYS = frozenset(
    {
        "api_key",
        "args",
        "arguments",
        "authorization_reason",
        "content",
        "credential",
        "credentials",
        "output",
        "outputs",
        "password",
        "passwords",
        "private_key",
        "prompt",
        "prompts",
        "reason",
        "secret",
        "secrets",
        "task",
        "token",
        "tokens",
    }
)


class InsightsError(ValueError):
    """Insights input, signal, or report configuration is invalid."""


class PersonalizationError(ValueError):
    """Personalization approval or profile compilation failed closed."""


class SignalKind(StrEnum):
    HUMAN_CORRECTION = "human_correction"
    HUMAN_PREFERENCE = "human_preference"
    HUMAN_WORKFLOW = "human_workflow"
    STRUCTURAL_RUN_OUTCOME = "structural_run_outcome"
    GOVERNED_TOOL_REJECTION = "governed_tool_rejection"


class FindingKind(StrEnum):
    REPEATED_CORRECTION = "repeated-correction"
    REPEATED_PREFERENCE = "repeated-preference"
    REPEATED_WORKFLOW = "repeated-workflow"
    RUN_FRICTION = "run-friction"
    TOOL_FRICTION = "tool-friction"


class CandidateKind(StrEnum):
    RULE = "rule"
    SKILL = "skill"


class HumanFeedbackKind(StrEnum):
    CORRECTION = "correction"
    PREFERENCE = "preference"
    WORKFLOW = "workflow"


_HUMAN_KIND_MAP = {
    HumanFeedbackKind.CORRECTION: SignalKind.HUMAN_CORRECTION,
    HumanFeedbackKind.PREFERENCE: SignalKind.HUMAN_PREFERENCE,
    HumanFeedbackKind.WORKFLOW: SignalKind.HUMAN_WORKFLOW,
}

_FINDING_FOR_HUMAN = {
    SignalKind.HUMAN_CORRECTION: FindingKind.REPEATED_CORRECTION,
    SignalKind.HUMAN_PREFERENCE: FindingKind.REPEATED_PREFERENCE,
    SignalKind.HUMAN_WORKFLOW: FindingKind.REPEATED_WORKFLOW,
}

_CANDIDATE_FOR_HUMAN = {
    SignalKind.HUMAN_CORRECTION: CandidateKind.RULE,
    SignalKind.HUMAN_PREFERENCE: CandidateKind.RULE,
    SignalKind.HUMAN_WORKFLOW: CandidateKind.SKILL,
}

_HUMAN_SIGNAL_KINDS = frozenset(_FINDING_FOR_HUMAN)
_FRICTION_RUN_STATUSES = frozenset(
    status for status in RunStatus if status is not RunStatus.COMPLETED
)
_REJECTION_STATUSES = frozenset(
    status for status in ExecutionStatus if status is not ExecutionStatus.EXECUTED
)
_CREDENTIAL_PATTERNS = (
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
    re.compile(r"\bsk-[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bghp_[A-Za-z0-9]{36}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
    re.compile(r"(?i)\b(password|passwd|secret|api[_-]?key|token)\s*[:=]\s*\S+"),
    re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._\-+/=]{16,}"),
)


def canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(dict(payload), sort_keys=True, separators=(",", ":"))


def sha256_hex(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(payload).encode("utf-8")).hexdigest()


def positive_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise InsightsError(f"{field_name} must be a positive integer")
    return value


def finite_timestamp(value: object, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InsightsError(f"{field_name} must be a finite number")
    timestamp = float(value)
    if not math.isfinite(timestamp):
        raise InsightsError(f"{field_name} must be a finite number")
    return timestamp


def require_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise InsightsError(f"{field_name} must be a non-empty string")
    if "\x00" in value:
        raise InsightsError(f"{field_name} must not contain null bytes")
    return value.strip()


def normalize_feedback(text: str) -> str:
    return " ".join(text.split()).casefold()


def reject_credential_shaped(text: str) -> None:
    for pattern in _CREDENTIAL_PATTERNS:
        if pattern.search(text):
            raise InsightsError("feedback looks credential-shaped")


def parse_run_status(value: object) -> RunStatus:
    if isinstance(value, RunStatus):
        return value
    if not isinstance(value, str):
        raise InsightsError("run status must be a defined CBrain RunStatus")
    try:
        return RunStatus(value)
    except ValueError as exc:
        raise InsightsError("run status must be a defined CBrain RunStatus") from exc


def parse_execution_status(value: object) -> ExecutionStatus:
    if isinstance(value, ExecutionStatus):
        return value
    if not isinstance(value, str):
        raise InsightsError("execution status must be a defined CBrain ExecutionStatus")
    try:
        return ExecutionStatus(value)
    except ValueError as exc:
        raise InsightsError(
            "execution status must be a defined CBrain ExecutionStatus"
        ) from exc


def parse_human_feedback_kind(value: object) -> HumanFeedbackKind:
    if isinstance(value, HumanFeedbackKind):
        return value
    if not isinstance(value, str):
        raise InsightsError("feedback kind must be correction, preference, or workflow")
    try:
        return HumanFeedbackKind(value)
    except ValueError as exc:
        raise InsightsError(
            "feedback kind must be correction, preference, or workflow"
        ) from exc


def _reject_forbidden_keys(payload: Mapping[str, Any]) -> None:
    for key in payload:
        if key in FORBIDDEN_SIGNAL_KEYS:
            raise InsightsError(f"forbidden signal field {key!r}")


@dataclass(frozen=True, slots=True)
class LearningSignal:
    schema_version: int
    signal_id: str
    kind: SignalKind
    agent_id: str
    idempotency_key: str
    observed_at: float
    content_hash: str
    reviewer_id: str | None = None
    feedback_text: str | None = None
    run_id: str | None = None
    run_status: RunStatus | None = None
    tool_name: str | None = None
    execution_status: ExecutionStatus | None = None

    def to_mapping(self) -> Mapping[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "signal_id": self.signal_id,
            "kind": self.kind.value,
            "agent_id": self.agent_id,
            "idempotency_key": self.idempotency_key,
            "observed_at": self.observed_at,
            "content_hash": self.content_hash,
            "reviewer_id": self.reviewer_id,
            "feedback_text": self.feedback_text,
            "run_id": self.run_id,
            "run_status": None if self.run_status is None else self.run_status.value,
            "tool_name": self.tool_name,
            "execution_status": (
                None if self.execution_status is None else self.execution_status.value
            ),
        }
        _reject_forbidden_keys(payload)
        return MappingProxyType(payload)

    def to_json(self) -> str:
        return canonical_json(self.to_mapping())

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> LearningSignal:
        if not isinstance(payload, Mapping):
            raise InsightsError("learning signal must be an object")
        _reject_forbidden_keys(payload)
        try:
            kind = SignalKind(payload["kind"])
        except (KeyError, ValueError) as exc:
            raise InsightsError("learning signal kind is invalid") from exc
        schema_version = payload.get("schema_version")
        if schema_version != SIGNAL_SCHEMA_VERSION:
            raise InsightsError("unsupported learning signal schema")
        feedback_raw = payload.get("feedback_text")
        if feedback_raw is not None and not isinstance(feedback_raw, str):
            raise InsightsError("feedback_text must be a string")
        signal = cls(
            schema_version=schema_version,
            signal_id=require_text(payload.get("signal_id"), "signal_id"),
            kind=kind,
            agent_id=require_text(payload.get("agent_id"), "agent_id"),
            idempotency_key=require_text(
                payload.get("idempotency_key"), "idempotency_key"
            ),
            observed_at=finite_timestamp(payload.get("observed_at"), "observed_at"),
            content_hash=require_text(payload.get("content_hash"), "content_hash"),
            reviewer_id=_optional_text(payload.get("reviewer_id"), "reviewer_id"),
            feedback_text=feedback_raw,
            run_id=_optional_text(payload.get("run_id"), "run_id"),
            run_status=(
                None
                if payload.get("run_status") is None
                else parse_run_status(payload.get("run_status"))
            ),
            tool_name=_optional_text(payload.get("tool_name"), "tool_name"),
            execution_status=(
                None
                if payload.get("execution_status") is None
                else parse_execution_status(payload.get("execution_status"))
            ),
        )
        expected = _signal_content_hash(signal)
        if signal.content_hash != expected or signal.signal_id != expected:
            raise InsightsError("learning signal content hash mismatch")
        _validate_signal_shape(signal)
        return signal

    @classmethod
    def from_json(cls, raw: str) -> LearningSignal:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise InsightsError("learning signal is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise InsightsError("learning signal must be a JSON object")
        return cls.from_mapping(payload)


def _optional_text(value: object, field_name: str) -> str | None:
    if value is None:
        return None
    return require_text(value, field_name)


def _signal_identity_payload(signal: LearningSignal) -> dict[str, Any]:
    return {
        "schema_version": signal.schema_version,
        "kind": signal.kind.value,
        "agent_id": signal.agent_id,
        "idempotency_key": signal.idempotency_key,
        "reviewer_id": signal.reviewer_id,
        "feedback_text": signal.feedback_text,
        "run_id": signal.run_id,
        "run_status": None if signal.run_status is None else signal.run_status.value,
        "tool_name": signal.tool_name,
        "execution_status": (
            None if signal.execution_status is None else signal.execution_status.value
        ),
    }


def _signal_content_hash(signal: LearningSignal) -> str:
    return sha256_hex(_signal_identity_payload(signal))


def _validate_signal_shape(signal: LearningSignal) -> None:
    if signal.schema_version != SIGNAL_SCHEMA_VERSION:
        raise InsightsError("unsupported learning signal schema")
    if signal.kind in _HUMAN_SIGNAL_KINDS:
        if signal.reviewer_id is None or signal.feedback_text is None:
            raise InsightsError("human signals require reviewer_id and feedback_text")
        if signal.run_status is not None or signal.tool_name is not None:
            raise InsightsError("human signals must not carry runtime fields")
        if signal.execution_status is not None:
            raise InsightsError("human signals must not carry execution status")
        return
    if signal.kind is SignalKind.STRUCTURAL_RUN_OUTCOME:
        if signal.run_id is None or signal.run_status is None:
            raise InsightsError("run outcomes require run_id and run_status")
        if (
            signal.reviewer_id is not None
            or signal.feedback_text is not None
            or signal.tool_name is not None
            or signal.execution_status is not None
        ):
            raise InsightsError("run outcomes must be structural only")
        return
    if (
        signal.run_id is None
        or signal.tool_name is None
        or signal.execution_status is None
    ):
        raise InsightsError("tool rejections require run_id, tool_name, and status")
    if signal.execution_status not in _REJECTION_STATUSES:
        raise InsightsError("tool rejection status must be a governed rejection")
    if (
        signal.reviewer_id is not None
        or signal.feedback_text is not None
        or signal.run_status is not None
    ):
        raise InsightsError("tool rejections must be structural only")


def finalize_signal(
    *,
    kind: SignalKind,
    agent_id: str,
    idempotency_key: str,
    observed_at: float,
    reviewer_id: str | None = None,
    feedback_text: str | None = None,
    run_id: str | None = None,
    run_status: RunStatus | None = None,
    tool_name: str | None = None,
    execution_status: ExecutionStatus | None = None,
) -> LearningSignal:
    draft = LearningSignal(
        schema_version=SIGNAL_SCHEMA_VERSION,
        signal_id="pending",
        kind=kind,
        agent_id=require_text(agent_id, "agent_id"),
        idempotency_key=require_text(idempotency_key, "idempotency_key"),
        observed_at=finite_timestamp(observed_at, "observed_at"),
        content_hash="pending",
        reviewer_id=_optional_text(reviewer_id, "reviewer_id"),
        feedback_text=feedback_text,
        run_id=_optional_text(run_id, "run_id"),
        run_status=run_status,
        tool_name=_optional_text(tool_name, "tool_name"),
        execution_status=execution_status,
    )
    if draft.feedback_text is not None:
        text = require_text(draft.feedback_text, "feedback_text")
        if len(text) > MAX_FEEDBACK_CHARS:
            raise InsightsError("feedback_text exceeds max length")
        reject_credential_shaped(text)
        draft = LearningSignal(
            schema_version=draft.schema_version,
            signal_id=draft.signal_id,
            kind=draft.kind,
            agent_id=draft.agent_id,
            idempotency_key=draft.idempotency_key,
            observed_at=draft.observed_at,
            content_hash=draft.content_hash,
            reviewer_id=draft.reviewer_id,
            feedback_text=text,
            run_id=draft.run_id,
            run_status=draft.run_status,
            tool_name=draft.tool_name,
            execution_status=draft.execution_status,
        )
    _validate_signal_shape(draft)
    digest = _signal_content_hash(draft)
    return LearningSignal(
        schema_version=draft.schema_version,
        signal_id=digest,
        kind=draft.kind,
        agent_id=draft.agent_id,
        idempotency_key=draft.idempotency_key,
        observed_at=draft.observed_at,
        content_hash=digest,
        reviewer_id=draft.reviewer_id,
        feedback_text=draft.feedback_text,
        run_id=draft.run_id,
        run_status=draft.run_status,
        tool_name=draft.tool_name,
        execution_status=draft.execution_status,
    )


def human_signal(
    *,
    kind: HumanFeedbackKind | str,
    agent_id: str,
    reviewer_id: str,
    feedback_text: str,
    idempotency_key: str,
    observed_at: float,
    run_id: str | None = None,
) -> LearningSignal:
    parsed = parse_human_feedback_kind(kind)
    return finalize_signal(
        kind=_HUMAN_KIND_MAP[parsed],
        agent_id=agent_id,
        idempotency_key=idempotency_key,
        observed_at=observed_at,
        reviewer_id=reviewer_id,
        feedback_text=feedback_text,
        run_id=run_id,
    )


def run_outcome_signal(
    *,
    agent_id: str,
    run_id: str,
    status: RunStatus | str,
    idempotency_key: str,
    observed_at: float,
) -> LearningSignal:
    return finalize_signal(
        kind=SignalKind.STRUCTURAL_RUN_OUTCOME,
        agent_id=agent_id,
        idempotency_key=idempotency_key,
        observed_at=observed_at,
        run_id=run_id,
        run_status=parse_run_status(status),
    )


def tool_rejection_signal(
    *,
    agent_id: str,
    run_id: str,
    tool_name: str,
    execution_status: ExecutionStatus | str,
    idempotency_key: str,
    observed_at: float,
) -> LearningSignal:
    return finalize_signal(
        kind=SignalKind.GOVERNED_TOOL_REJECTION,
        agent_id=agent_id,
        idempotency_key=idempotency_key,
        observed_at=observed_at,
        run_id=run_id,
        tool_name=tool_name,
        execution_status=parse_execution_status(execution_status),
    )


@dataclass(frozen=True, slots=True)
class InsightFinding:
    finding_id: str
    kind: FindingKind
    agent_id: str
    occurrence_count: int
    evidence_ids: tuple[str, ...]
    summary: str
    grouping_key: str
    finding_hash: str
    proposes_candidate: bool

    def to_mapping(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "finding_id": self.finding_id,
                "kind": self.kind.value,
                "agent_id": self.agent_id,
                "occurrence_count": self.occurrence_count,
                "evidence_ids": list(self.evidence_ids),
                "summary": self.summary,
                "grouping_key": self.grouping_key,
                "finding_hash": self.finding_hash,
                "proposes_candidate": self.proposes_candidate,
            }
        )


@dataclass(frozen=True, slots=True)
class GuidanceCandidate:
    candidate_id: str
    candidate_kind: CandidateKind
    agent_id: str
    finding_kind: FindingKind
    title: str
    body: str
    evidence_ids: tuple[str, ...]
    occurrence_count: int
    candidate_hash: str
    skill_id: str | None = None

    def to_mapping(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "candidate_id": self.candidate_id,
                "candidate_kind": self.candidate_kind.value,
                "agent_id": self.agent_id,
                "finding_kind": self.finding_kind.value,
                "title": self.title,
                "body": self.body,
                "evidence_ids": list(self.evidence_ids),
                "occurrence_count": self.occurrence_count,
                "candidate_hash": self.candidate_hash,
                "skill_id": self.skill_id,
            }
        )


@dataclass(frozen=True, slots=True)
class InsightsSnapshot:
    schema_version: int
    generated_at: float
    window_days: int
    min_occurrences: int
    agent_id: str | None
    findings: tuple[InsightFinding, ...]
    candidates: tuple[GuidanceCandidate, ...]
    snapshot_hash: str

    def to_mapping(self) -> Mapping[str, Any]:
        return MappingProxyType(
            {
                "schema_version": self.schema_version,
                "generated_at": self.generated_at,
                "window_days": self.window_days,
                "min_occurrences": self.min_occurrences,
                "agent_id": self.agent_id,
                "findings": [dict(item.to_mapping()) for item in self.findings],
                "candidates": [dict(item.to_mapping()) for item in self.candidates],
                "snapshot_hash": self.snapshot_hash,
            }
        )


@dataclass(frozen=True, slots=True)
class ApprovalRecord:
    schema_version: int
    approval_id: str
    candidate_hash: str
    candidate_kind: CandidateKind
    agent_id: str
    reviewer_id: str
    profile_fingerprint: str
    instruction_text: str
    skill_id: str | None
    evidence_ids: tuple[str, ...]
    approved_at: float
    content_hash: str
    idempotency_key: str

    def to_mapping(self) -> Mapping[str, Any]:
        payload = {
            "schema_version": self.schema_version,
            "approval_id": self.approval_id,
            "candidate_hash": self.candidate_hash,
            "candidate_kind": self.candidate_kind.value,
            "agent_id": self.agent_id,
            "reviewer_id": self.reviewer_id,
            "profile_fingerprint": self.profile_fingerprint,
            "instruction_text": self.instruction_text,
            "skill_id": self.skill_id,
            "evidence_ids": list(self.evidence_ids),
            "approved_at": self.approved_at,
            "content_hash": self.content_hash,
            "idempotency_key": self.idempotency_key,
        }
        _reject_forbidden_keys(payload)
        return MappingProxyType(payload)

    def to_json(self) -> str:
        return canonical_json(self.to_mapping())

    @classmethod
    def from_mapping(cls, payload: Mapping[str, Any]) -> ApprovalRecord:
        if not isinstance(payload, Mapping):
            raise InsightsError("approval record must be an object")
        _reject_forbidden_keys(payload)
        try:
            candidate_kind = CandidateKind(payload["candidate_kind"])
        except (KeyError, ValueError) as exc:
            raise InsightsError("approval candidate_kind is invalid") from exc
        evidence_raw = payload.get("evidence_ids")
        if not isinstance(evidence_raw, list) or any(
            not isinstance(item, str) or not item.strip() for item in evidence_raw
        ):
            raise InsightsError("approval evidence_ids are invalid")
        schema_version = payload.get("schema_version")
        if schema_version != APPROVAL_SCHEMA_VERSION:
            raise InsightsError("unsupported approval schema")
        record = cls(
            schema_version=schema_version,
            approval_id=require_text(payload.get("approval_id"), "approval_id"),
            candidate_hash=require_text(
                payload.get("candidate_hash"), "candidate_hash"
            ),
            candidate_kind=candidate_kind,
            agent_id=require_text(payload.get("agent_id"), "agent_id"),
            reviewer_id=require_text(payload.get("reviewer_id"), "reviewer_id"),
            profile_fingerprint=require_text(
                payload.get("profile_fingerprint"), "profile_fingerprint"
            ),
            instruction_text=require_text(
                payload.get("instruction_text"), "instruction_text"
            ),
            skill_id=_optional_text(payload.get("skill_id"), "skill_id"),
            evidence_ids=tuple(evidence_raw),
            approved_at=finite_timestamp(payload.get("approved_at"), "approved_at"),
            content_hash=require_text(payload.get("content_hash"), "content_hash"),
            idempotency_key=require_text(
                payload.get("idempotency_key"), "idempotency_key"
            ),
        )
        expected = _approval_content_hash(record)
        if record.content_hash != expected or record.approval_id != expected:
            raise InsightsError("approval content hash mismatch")
        _validate_approval(record)
        return record

    @classmethod
    def from_json(cls, raw: str) -> ApprovalRecord:
        try:
            payload = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise InsightsError("approval record is not valid JSON") from exc
        if not isinstance(payload, dict):
            raise InsightsError("approval record must be a JSON object")
        return cls.from_mapping(payload)


def _approval_identity_payload(record: ApprovalRecord) -> dict[str, Any]:
    return {
        "schema_version": record.schema_version,
        "candidate_hash": record.candidate_hash,
        "candidate_kind": record.candidate_kind.value,
        "agent_id": record.agent_id,
        "reviewer_id": record.reviewer_id,
        "profile_fingerprint": record.profile_fingerprint,
        "instruction_text": record.instruction_text,
        "skill_id": record.skill_id,
        "evidence_ids": list(record.evidence_ids),
        "idempotency_key": record.idempotency_key,
    }


def _approval_content_hash(record: ApprovalRecord) -> str:
    return sha256_hex(_approval_identity_payload(record))


def _validate_approval(record: ApprovalRecord) -> None:
    if record.schema_version != APPROVAL_SCHEMA_VERSION:
        raise InsightsError("unsupported approval schema")
    require_text(record.approval_id, "approval_id")
    require_text(record.candidate_hash, "candidate_hash")
    require_text(record.agent_id, "agent_id")
    require_text(record.reviewer_id, "reviewer_id")
    require_text(record.profile_fingerprint, "profile_fingerprint")
    require_text(record.instruction_text, "instruction_text")
    require_text(record.content_hash, "content_hash")
    require_text(record.idempotency_key, "idempotency_key")
    finite_timestamp(record.approved_at, "approved_at")
    if record.candidate_kind is CandidateKind.SKILL:
        if record.skill_id is None:
            raise InsightsError("skill approvals require skill_id")
        require_text(record.skill_id, "skill_id")
    elif record.skill_id is not None:
        raise InsightsError("rule approvals must not include skill_id")
    if not record.evidence_ids:
        raise InsightsError("approval evidence_ids must be non-empty")


class LearningStore(Protocol):
    def append_signal(self, signal: LearningSignal) -> LearningSignal: ...

    def list_signals(
        self,
        *,
        agent_id: str | None = None,
        since: float | None = None,
        until: float | None = None,
    ) -> tuple[LearningSignal, ...]: ...

    def load_signals(self, signal_ids: Sequence[str]) -> tuple[LearningSignal, ...]: ...

    def append_approval(self, approval: ApprovalRecord) -> ApprovalRecord: ...

    def list_approvals(
        self,
        *,
        agent_id: str | None = None,
    ) -> tuple[ApprovalRecord, ...]: ...

    def load_approvals(
        self, approval_ids: Sequence[str]
    ) -> tuple[ApprovalRecord, ...]: ...


class LearningStoreError(ValueError):
    """Learning storage is missing, inconsistent, or cannot accept a write."""


def validate_signal_columns(
    signal: LearningSignal,
    *,
    schema_version: int,
    kind: str,
    agent_id: str,
    content_hash: str,
    observed_at: float,
    idempotency_key: str,
) -> None:
    if signal.schema_version != schema_version:
        raise LearningStoreError("schema_version column mismatch")
    if signal.kind.value != kind:
        raise LearningStoreError("kind column mismatch")
    if signal.agent_id != agent_id:
        raise LearningStoreError("agent_id column mismatch")
    if signal.content_hash != content_hash or signal.signal_id != content_hash:
        raise LearningStoreError("content_hash column mismatch")
    if signal.observed_at != observed_at:
        raise LearningStoreError("observed_at column mismatch")
    if signal.idempotency_key != idempotency_key:
        raise LearningStoreError("idempotency_key column mismatch")


def validate_approval_columns(
    approval: ApprovalRecord,
    *,
    schema_version: int,
    candidate_hash: str,
    profile_fingerprint: str,
    reviewer_id: str,
    content_hash: str,
    approved_at: float,
    idempotency_key: str,
) -> None:
    if approval.schema_version != schema_version:
        raise LearningStoreError("schema_version column mismatch")
    if approval.candidate_hash != candidate_hash:
        raise LearningStoreError("candidate_hash column mismatch")
    if approval.profile_fingerprint != profile_fingerprint:
        raise LearningStoreError("profile_fingerprint column mismatch")
    if approval.reviewer_id != reviewer_id:
        raise LearningStoreError("reviewer_id column mismatch")
    if approval.content_hash != content_hash or approval.approval_id != content_hash:
        raise LearningStoreError("content_hash column mismatch")
    if approval.approved_at != approved_at:
        raise LearningStoreError("approved_at column mismatch")
    if approval.idempotency_key != idempotency_key:
        raise LearningStoreError("idempotency_key column mismatch")


def _in_window(signal: LearningSignal, *, now: float, window_days: int) -> bool:
    start = now - (window_days * SECONDS_PER_DAY)
    return start <= signal.observed_at <= now


def _sorted_ids(signals: Sequence[LearningSignal]) -> tuple[str, ...]:
    return tuple(sorted(signal.signal_id for signal in signals))


def _finding(
    *,
    kind: FindingKind,
    agent_id: str,
    signals: Sequence[LearningSignal],
    grouping_key: str,
    summary: str,
    proposes_candidate: bool,
) -> InsightFinding:
    evidence_ids = _sorted_ids(signals)
    payload = {
        "kind": kind.value,
        "agent_id": agent_id,
        "evidence_ids": list(evidence_ids),
        "grouping_key": grouping_key,
        "occurrence_count": len(evidence_ids),
        "proposes_candidate": proposes_candidate,
    }
    digest = sha256_hex(payload)
    return InsightFinding(
        finding_id=digest,
        kind=kind,
        agent_id=agent_id,
        occurrence_count=len(evidence_ids),
        evidence_ids=evidence_ids,
        summary=summary,
        grouping_key=grouping_key,
        finding_hash=digest,
        proposes_candidate=proposes_candidate,
    )


def _candidate_from_human_group(
    signals: Sequence[LearningSignal],
) -> GuidanceCandidate:
    ordered = tuple(
        sorted(signals, key=lambda item: (item.observed_at, item.signal_id))
    )
    first = ordered[0]
    assert first.feedback_text is not None
    kind = first.kind
    finding_kind = _FINDING_FOR_HUMAN[kind]
    candidate_kind = _CANDIDATE_FOR_HUMAN[kind]
    evidence_ids = _sorted_ids(ordered)
    body = first.feedback_text
    payload = {
        "candidate_kind": candidate_kind.value,
        "agent_id": first.agent_id,
        "finding_kind": finding_kind.value,
        "body": body,
        "evidence_ids": list(evidence_ids),
        "occurrence_count": len(evidence_ids),
    }
    digest = sha256_hex(payload)
    skill_id = None if candidate_kind is CandidateKind.RULE else f"skill:{digest}"
    title = {
        FindingKind.REPEATED_CORRECTION: "Repeated correction",
        FindingKind.REPEATED_PREFERENCE: "Repeated preference",
        FindingKind.REPEATED_WORKFLOW: "Repeated workflow",
    }[finding_kind]
    return GuidanceCandidate(
        candidate_id=digest,
        candidate_kind=candidate_kind,
        agent_id=first.agent_id,
        finding_kind=finding_kind,
        title=title,
        body=body,
        evidence_ids=evidence_ids,
        occurrence_count=len(evidence_ids),
        candidate_hash=digest,
        skill_id=skill_id,
    )


class InsightsEngine:
    """Deterministic, read-only proposer of guidance and friction findings."""

    def __init__(
        self,
        *,
        window_days: int = DEFAULT_WINDOW_DAYS,
        min_occurrences: int = MIN_GUIDANCE_OCCURRENCES,
    ) -> None:
        self._window_days = positive_int(window_days, "window_days")
        requested = positive_int(min_occurrences, "min_occurrences")
        self._min_occurrences = max(requested, MIN_GUIDANCE_OCCURRENCES)

    @property
    def window_days(self) -> int:
        return self._window_days

    @property
    def min_occurrences(self) -> int:
        return self._min_occurrences

    def analyze(
        self,
        store: LearningStore,
        *,
        agent_id: str | None = None,
        now: float | None = None,
    ) -> InsightsSnapshot:
        generated_at = finite_timestamp(
            time.time() if now is None else now, "generated_at"
        )
        scoped_agent = None if agent_id is None else require_text(agent_id, "agent_id")
        since = generated_at - (self._window_days * SECONDS_PER_DAY)
        signals = tuple(
            signal
            for signal in store.list_signals(agent_id=scoped_agent, since=since)
            if _in_window(signal, now=generated_at, window_days=self._window_days)
        )
        return self._analyze_signals(
            signals, agent_id=scoped_agent, generated_at=generated_at
        )

    def _analyze_signals(
        self,
        signals: Sequence[LearningSignal],
        *,
        agent_id: str | None,
        generated_at: float,
    ) -> InsightsSnapshot:
        findings: list[InsightFinding] = []
        candidates: list[GuidanceCandidate] = []
        human_groups: dict[tuple[str, SignalKind, str], list[LearningSignal]] = {}
        run_groups: dict[tuple[str, RunStatus], list[LearningSignal]] = {}
        tool_groups: dict[tuple[str, str, ExecutionStatus], list[LearningSignal]] = {}

        for signal in signals:
            if signal.kind in _HUMAN_SIGNAL_KINDS:
                assert signal.feedback_text is not None
                key = (
                    signal.agent_id,
                    signal.kind,
                    normalize_feedback(signal.feedback_text),
                )
                human_groups.setdefault(key, []).append(signal)
            elif (
                signal.kind is SignalKind.STRUCTURAL_RUN_OUTCOME
                and signal.run_status in _FRICTION_RUN_STATUSES
            ):
                assert signal.run_status is not None
                run_groups.setdefault((signal.agent_id, signal.run_status), []).append(
                    signal
                )
            elif (
                signal.kind is SignalKind.GOVERNED_TOOL_REJECTION
                and signal.execution_status in _REJECTION_STATUSES
            ):
                assert signal.tool_name is not None
                assert signal.execution_status is not None
                tool_groups.setdefault(
                    (signal.agent_id, signal.tool_name, signal.execution_status),
                    [],
                ).append(signal)

        for (group_agent, kind, normalized), group in sorted(human_groups.items()):
            if len(group) < self._min_occurrences:
                continue
            finding_kind = _FINDING_FOR_HUMAN[kind]
            candidate = _candidate_from_human_group(group)
            findings.append(
                _finding(
                    kind=finding_kind,
                    agent_id=group_agent,
                    signals=group,
                    grouping_key=f"{kind.value}:{group_agent}:{normalized}",
                    summary=f"{finding_kind.value} observed {len(group)} times",
                    proposes_candidate=True,
                )
            )
            candidates.append(candidate)

        for (group_agent, run_status), group in sorted(
            run_groups.items(), key=lambda item: (item[0][0], item[0][1].value)
        ):
            if len(group) < self._min_occurrences:
                continue
            findings.append(
                _finding(
                    kind=FindingKind.RUN_FRICTION,
                    agent_id=group_agent,
                    signals=group,
                    grouping_key=f"run-friction:{group_agent}:{run_status.value}",
                    summary=(
                        f"run status {run_status.value} observed {len(group)} times"
                    ),
                    proposes_candidate=False,
                )
            )

        for (group_agent, tool_name, execution_status), group in sorted(
            tool_groups.items(),
            key=lambda item: (item[0][0], item[0][1], item[0][2].value),
        ):
            if len(group) < self._min_occurrences:
                continue
            findings.append(
                _finding(
                    kind=FindingKind.TOOL_FRICTION,
                    agent_id=group_agent,
                    signals=group,
                    grouping_key=(
                        f"tool-friction:{group_agent}:{tool_name}:"
                        f"{execution_status.value}"
                    ),
                    summary=(
                        f"tool {tool_name} status {execution_status.value} observed "
                        f"{len(group)} times"
                    ),
                    proposes_candidate=False,
                )
            )

        findings_tuple = tuple(
            sorted(findings, key=lambda item: (item.kind.value, item.finding_hash))
        )
        candidates_tuple = tuple(
            sorted(candidates, key=lambda item: item.candidate_hash)
        )
        snapshot_hash = sha256_hex(
            {
                "schema_version": SIGNAL_SCHEMA_VERSION,
                "window_days": self._window_days,
                "min_occurrences": self._min_occurrences,
                "agent_id": agent_id,
                "findings": [item.finding_hash for item in findings_tuple],
                "candidates": [item.candidate_hash for item in candidates_tuple],
            }
        )
        return InsightsSnapshot(
            schema_version=SIGNAL_SCHEMA_VERSION,
            generated_at=generated_at,
            window_days=self._window_days,
            min_occurrences=self._min_occurrences,
            agent_id=agent_id,
            findings=findings_tuple,
            candidates=candidates_tuple,
            snapshot_hash=snapshot_hash,
        )


class LearningRecorder:
    """Explicit, non-authoritative observation of completed foundation runs."""

    def __init__(
        self,
        store: LearningStore,
        *,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._store = store
        self._clock = clock or time.time

    def record_run(self, *, agent_id: str, result: RunResult) -> bool:
        try:
            self._record_run(agent_id=agent_id, result=result)
        except Exception:
            return False
        return True

    def _record_run(self, *, agent_id: str, result: RunResult) -> None:
        observed_at = self._clock()
        self._store.append_signal(
            run_outcome_signal(
                agent_id=agent_id,
                run_id=result.run_id,
                status=result.status,
                idempotency_key=f"run-outcome:{agent_id}:{result.run_id}",
                observed_at=observed_at,
            )
        )
        tools_by_request: dict[str, str] = {}
        rejection_index = 0
        for event in result.events:
            request_id = event.detail.get("request_id")
            if event.kind is RunEventKind.TOOL_REQUESTED:
                tool_name = event.detail.get("tool_name")
                if isinstance(request_id, str) and isinstance(tool_name, str):
                    tools_by_request[request_id] = tool_name
                continue
            if event.kind is not RunEventKind.TOOL_REJECTED:
                continue
            status_raw = event.detail.get("execution_status")
            tool_name = None
            if isinstance(request_id, str):
                tool_name = tools_by_request.get(request_id)
            if tool_name is None or status_raw is None:
                continue
            parsed_status = parse_execution_status(status_raw)
            self._store.append_signal(
                tool_rejection_signal(
                    agent_id=agent_id,
                    run_id=result.run_id,
                    tool_name=tool_name,
                    execution_status=parsed_status,
                    idempotency_key=(
                        f"tool-rejection:{agent_id}:{result.run_id}:"
                        f"{rejection_index}:{tool_name}:{parsed_status.value}"
                    ),
                    observed_at=observed_at,
                )
            )
            rejection_index += 1


def _finalize_approval(
    *,
    candidate: GuidanceCandidate,
    reviewer_id: str,
    profile_fingerprint_value: str,
    approved_at: float,
) -> ApprovalRecord:
    idempotency_key = (
        f"{reviewer_id}:{candidate.candidate_hash}:{profile_fingerprint_value}"
    )
    draft = ApprovalRecord(
        schema_version=APPROVAL_SCHEMA_VERSION,
        approval_id="pending",
        candidate_hash=candidate.candidate_hash,
        candidate_kind=candidate.candidate_kind,
        agent_id=candidate.agent_id,
        reviewer_id=reviewer_id,
        profile_fingerprint=profile_fingerprint_value,
        instruction_text=candidate.body,
        skill_id=candidate.skill_id,
        evidence_ids=candidate.evidence_ids,
        approved_at=approved_at,
        content_hash="pending",
        idempotency_key=idempotency_key,
    )
    digest = _approval_content_hash(draft)
    record = ApprovalRecord(
        schema_version=draft.schema_version,
        approval_id=digest,
        candidate_hash=draft.candidate_hash,
        candidate_kind=draft.candidate_kind,
        agent_id=draft.agent_id,
        reviewer_id=draft.reviewer_id,
        profile_fingerprint=draft.profile_fingerprint,
        instruction_text=draft.instruction_text,
        skill_id=draft.skill_id,
        evidence_ids=draft.evidence_ids,
        approved_at=draft.approved_at,
        content_hash=digest,
        idempotency_key=draft.idempotency_key,
    )
    _validate_approval(record)
    return record


class PersonalizationManager:
    """Human approval and compilation of standing personalization."""

    def __init__(
        self,
        store: LearningStore,
        *,
        allowed_approvers: frozenset[str],
        engine: InsightsEngine | None = None,
    ) -> None:
        if not isinstance(allowed_approvers, frozenset) or not allowed_approvers:
            raise PersonalizationError(
                "allowed_approvers must be a non-empty frozenset"
            )
        cleaned: set[str] = set()
        for approver in allowed_approvers:
            if not isinstance(approver, str) or not approver.strip():
                raise PersonalizationError("approver ids must be non-empty strings")
            cleaned.add(approver.strip())
        self._store = store
        self._allowed_approvers = frozenset(cleaned)
        self._engine = engine or InsightsEngine()

    def trusted_snapshot(
        self,
        *,
        agent_id: str | None = None,
        now: float | None = None,
    ) -> InsightsSnapshot:
        return self._engine.analyze(self._store, agent_id=agent_id, now=now)

    def verify_snapshot(self, snapshot: InsightsSnapshot) -> InsightsSnapshot:
        trusted = self._engine.analyze(
            self._store,
            agent_id=snapshot.agent_id,
            now=snapshot.generated_at,
        )
        if trusted.snapshot_hash != snapshot.snapshot_hash:
            raise PersonalizationError("snapshot is fabricated or stale")
        if (
            trusted.candidates != snapshot.candidates
            or trusted.findings != snapshot.findings
        ):
            raise PersonalizationError("snapshot is fabricated or stale")
        return trusted

    def approve(
        self,
        candidate: GuidanceCandidate,
        *,
        reviewer_id: str,
        profile: AgentProfile,
        approved_at: float | None = None,
        snapshot: InsightsSnapshot | None = None,
    ) -> ApprovalRecord:
        actor = require_text(reviewer_id, "reviewer_id")
        if actor not in self._allowed_approvers:
            raise PersonalizationError("reviewer is not allowlisted")
        if not isinstance(candidate, GuidanceCandidate):
            raise PersonalizationError("approval requires a guidance candidate")
        trusted = self.trusted_snapshot(agent_id=candidate.agent_id, now=approved_at)
        if snapshot is not None:
            self.verify_snapshot(snapshot)
        match = next(
            (
                item
                for item in trusted.candidates
                if item.candidate_hash == candidate.candidate_hash
            ),
            None,
        )
        if match is None:
            raise PersonalizationError("candidate is fabricated, changed, or stale")
        if (
            match.body != candidate.body
            or match.evidence_ids != candidate.evidence_ids
            or match.candidate_kind != candidate.candidate_kind
            or match.agent_id != candidate.agent_id
            or match.skill_id != candidate.skill_id
        ):
            raise PersonalizationError("candidate is fabricated, changed, or stale")
        evidence = self._store.load_signals(match.evidence_ids)
        if any(item.kind not in _HUMAN_SIGNAL_KINDS for item in evidence):
            raise PersonalizationError(
                "candidate evidence is not explicit human feedback"
            )
        fingerprint = profile_fingerprint(profile)
        if profile.agent_id != match.agent_id:
            raise PersonalizationError("candidate agent_id does not match profile")
        record = _finalize_approval(
            candidate=match,
            reviewer_id=actor,
            profile_fingerprint_value=fingerprint,
            approved_at=finite_timestamp(
                time.time() if approved_at is None else approved_at,
                "approved_at",
            ),
        )
        return self._store.append_approval(record)

    def compile_profile(
        self,
        profile: AgentProfile,
        *,
        active_skills: frozenset[str] = frozenset(),
    ) -> AgentProfile:
        if not isinstance(active_skills, frozenset):
            raise PersonalizationError("active_skills must be a frozenset")
        for skill_id in active_skills:
            if not isinstance(skill_id, str) or not skill_id.strip():
                raise PersonalizationError("active skill ids must be non-empty strings")
        fingerprint = profile_fingerprint(profile)
        approvals = tuple(
            approval
            for approval in self._store.list_approvals(agent_id=profile.agent_id)
            if approval.profile_fingerprint == fingerprint
        )
        # Reconstruct trusted approval bodies from storage, never caller snapshots.
        loaded = self._store.load_approvals(
            tuple(approval.approval_id for approval in approvals)
        )
        if loaded != approvals:
            raise PersonalizationError("stored approvals could not be reconstructed")
        approved_skills = {
            approval.skill_id
            for approval in loaded
            if approval.candidate_kind is CandidateKind.SKILL and approval.skill_id
        }
        unknown = sorted(active_skills - approved_skills)
        if unknown:
            raise PersonalizationError(f"unknown or unapproved skills: {unknown}")
        rules = tuple(
            sorted(
                (
                    approval
                    for approval in loaded
                    if approval.candidate_kind is CandidateKind.RULE
                ),
                key=lambda item: item.candidate_hash,
            )
        )
        skills = tuple(
            sorted(
                (
                    approval
                    for approval in loaded
                    if approval.candidate_kind is CandidateKind.SKILL
                    and approval.skill_id in active_skills
                ),
                key=lambda item: item.candidate_hash,
            )
        )
        sections = [profile.instructions.rstrip()]
        if rules:
            sections.append("## Approved personalization rules")
            sections.extend(f"- {approval.instruction_text}" for approval in rules)
        if skills:
            sections.append("## Activated skills")
            for approval in skills:
                sections.append(f"### {approval.skill_id}")
                sections.append(approval.instruction_text)
        metadata = dict(profile.metadata or {})
        metadata["personalization"] = {
            "base_profile_fingerprint": fingerprint,
            "approved_rule_hashes": [item.candidate_hash for item in rules],
            "active_skill_ids": [item.skill_id for item in skills],
        }
        return AgentProfile(
            agent_id=profile.agent_id,
            instructions="\n\n".join(sections),
            model_route=profile.model_route,
            permitted_tools=profile.permitted_tools,
            max_model_turns=profile.max_model_turns,
            max_tool_calls=profile.max_tool_calls,
            timeout_seconds=profile.timeout_seconds,
            limits=profile.limits,
            metadata=metadata,
        )


def build_report(
    store: LearningStore,
    *,
    agent_id: str | None = None,
    window_days: int = DEFAULT_WINDOW_DAYS,
    min_occurrences: int = MIN_GUIDANCE_OCCURRENCES,
    now: float | None = None,
) -> Mapping[str, Any]:
    engine = InsightsEngine(window_days=window_days, min_occurrences=min_occurrences)
    snapshot = engine.analyze(store, agent_id=agent_id, now=now)
    approvals = store.list_approvals(agent_id=agent_id)
    return MappingProxyType(
        {
            "schema": REPORT_SCHEMA,
            "generated_at": snapshot.generated_at,
            "window_days": snapshot.window_days,
            "min_occurrences": snapshot.min_occurrences,
            "agent_id": snapshot.agent_id,
            "signal_count": len(
                store.list_signals(
                    agent_id=agent_id,
                    since=snapshot.generated_at
                    - (snapshot.window_days * SECONDS_PER_DAY),
                )
            ),
            "approval_count": len(approvals),
            "findings": [dict(item.to_mapping()) for item in snapshot.findings],
            "candidates": [dict(item.to_mapping()) for item in snapshot.candidates],
            "snapshot_hash": snapshot.snapshot_hash,
        }
    )


def render_report_json(report: Mapping[str, Any]) -> str:
    return json.dumps(dict(report), indent=2, sort_keys=True) + "\n"


def encode_json_for_html(payload: Mapping[str, Any]) -> str:
    return (
        json.dumps(dict(payload), sort_keys=True)
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )


def render_report_html(report: Mapping[str, Any]) -> str:
    findings = report.get("findings")
    candidates = report.get("candidates")
    if not isinstance(findings, list) or not isinstance(candidates, list):
        raise InsightsError("report findings and candidates must be lists")
    rows: list[str] = []
    for finding in findings:
        if not isinstance(finding, Mapping):
            raise InsightsError("report finding is invalid")
        rows.append(
            "<li>{} — {} ({})</li>".format(
                html.escape(str(finding.get("kind", "")), quote=True),
                html.escape(str(finding.get("summary", "")), quote=True),
                html.escape(str(finding.get("occurrence_count", "")), quote=True),
            )
        )
    candidate_rows: list[str] = []
    for candidate in candidates:
        if not isinstance(candidate, Mapping):
            raise InsightsError("report candidate is invalid")
        candidate_rows.append(
            "<li>{}: {}</li>".format(
                html.escape(str(candidate.get("title", "")), quote=True),
                html.escape(str(candidate.get("body", "")), quote=True),
            )
        )
    window = html.escape(str(report.get("window_days", "")), quote=True)
    agent = html.escape(str(report.get("agent_id") or "all"), quote=True)
    findings_list = "".join(rows) or "<li>None</li>"
    candidates_list = "".join(candidate_rows) or "<li>None</li>"
    embedded = encode_json_for_html(report)
    return (
        "<!DOCTYPE html>\n"
        '<html lang="en">\n'
        "<head>\n"
        '  <meta charset="utf-8">\n'
        "  <title>CBrain Insights Report</title>\n"
        "</head>\n"
        "<body>\n"
        "  <h1>CBrain Insights Report</h1>\n"
        f"  <p>Window: {window} days</p>\n"
        f"  <p>Agent: {agent}</p>\n"
        "  <h2>Findings</h2>\n"
        f"  <ul>{findings_list}</ul>\n"
        "  <h2>Guidance candidates</h2>\n"
        f"  <ul>{candidates_list}</ul>\n"
        f'  <script type="application/json" id="insights-data">{embedded}</script>\n'
        "</body>\n"
        "</html>\n"
    )


__all__ = [
    "APPROVAL_SCHEMA_VERSION",
    "DEFAULT_WINDOW_DAYS",
    "FORBIDDEN_SIGNAL_KEYS",
    "MAX_FEEDBACK_CHARS",
    "MIN_GUIDANCE_OCCURRENCES",
    "REPORT_SCHEMA",
    "SIGNAL_SCHEMA_VERSION",
    "ApprovalRecord",
    "CandidateKind",
    "FindingKind",
    "GuidanceCandidate",
    "HumanFeedbackKind",
    "InsightFinding",
    "InsightsEngine",
    "InsightsError",
    "InsightsSnapshot",
    "LearningRecorder",
    "LearningSignal",
    "LearningStore",
    "LearningStoreError",
    "PersonalizationError",
    "PersonalizationManager",
    "SignalKind",
    "build_report",
    "canonical_json",
    "encode_json_for_html",
    "finalize_signal",
    "finite_timestamp",
    "human_signal",
    "parse_execution_status",
    "parse_human_feedback_kind",
    "parse_run_status",
    "positive_int",
    "reject_credential_shaped",
    "render_report_html",
    "render_report_json",
    "run_outcome_signal",
    "sha256_hex",
    "tool_rejection_signal",
    "validate_approval_columns",
    "validate_signal_columns",
]
