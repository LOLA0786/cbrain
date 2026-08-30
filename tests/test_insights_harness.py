"""Adversarial tests for the insights and hyperpersonalized harness."""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Mapping
from typing import Any

import pytest

from cbrain.agent import (
    AgentProfile,
    InsightsEngine,
    InsightsError,
    LearningRecorder,
    PersonalizationError,
    PersonalizationManager,
    RunLimits,
    RunStatus,
)
from cbrain.agent.contracts import RunEvent, RunEventKind, RunInput, RunResult
from cbrain.agent.durable import RunStoreError, new_running_record, profile_fingerprint
from cbrain.agent.durable_loop import resume_durable_run
from cbrain.agent.insights import (
    FORBIDDEN_SIGNAL_KEYS,
    ApprovalRecord,
    CandidateKind,
    FindingKind,
    GuidanceCandidate,
    InsightsSnapshot,
    LearningSignal,
    LearningStoreError,
    SignalKind,
    build_report,
    encode_json_for_html,
    human_signal,
    render_report_html,
    render_report_json,
    run_outcome_signal,
    tool_rejection_signal,
)
from cbrain.agent.insights_cli import main as insights_main
from cbrain.agent.insights_store import (
    InMemoryLearningStore,
    ReadOnlyLearningStore,
    SQLiteLearningStore,
)
from cbrain.contracts import ExecutionStatus
from cbrain.models import Message, MessageRole

NOW = 1_700_000_000.0
AGENT_ID = "ops-agent"
REVIEWER = "alice"


def _profile(**overrides: Any) -> AgentProfile:
    payload: dict[str, Any] = {
        "agent_id": AGENT_ID,
        "instructions": "Answer operational questions with concise evidence.",
        "model_route": "local",
        "permitted_tools": frozenset({"lookup_status"}),
        "max_model_turns": 6,
        "max_tool_calls": 3,
        "timeout_seconds": 120.0,
        "limits": RunLimits(),
        "metadata": {"team": "platform"},
    }
    payload.update(overrides)
    return AgentProfile(**payload)


def _human(
    store: InMemoryLearningStore | SQLiteLearningStore,
    *,
    text: str,
    key: str,
    kind: str = "correction",
    observed_at: float = NOW,
    reviewer_id: str = REVIEWER,
    agent_id: str = AGENT_ID,
) -> LearningSignal:
    signal = human_signal(
        kind=kind,
        agent_id=agent_id,
        reviewer_id=reviewer_id,
        feedback_text=text,
        idempotency_key=key,
        observed_at=observed_at,
    )
    return store.append_signal(signal)


def _manager(
    store: InMemoryLearningStore | SQLiteLearningStore,
    *,
    approvers: frozenset[str] = frozenset({REVIEWER}),
) -> PersonalizationManager:
    return PersonalizationManager(store, allowed_approvers=approvers)


def _run_result(
    *,
    status: RunStatus = RunStatus.COMPLETED,
    run_id: str = "run-1",
    events: tuple[RunEvent, ...] = (),
    metadata: Mapping[str, Any] | None = None,
    final_text: str | None = "done",
) -> RunResult:
    return RunResult(
        run_id=run_id,
        status=status,
        final_text=final_text,
        tool_calls=0,
        model_turns=1,
        events=events,
        metadata=metadata or {},
    )


def test_repeated_correction_proposes_a_rule() -> None:
    store = InMemoryLearningStore()
    _human(store, text="Always cite ticket IDs.", key="c1")
    _human(store, text="Always cite ticket IDs.", key="c2")
    snapshot = InsightsEngine().analyze(store, agent_id=AGENT_ID, now=NOW)
    assert [item.kind for item in snapshot.findings] == [
        FindingKind.REPEATED_CORRECTION
    ]
    assert len(snapshot.candidates) == 1
    candidate = snapshot.candidates[0]
    assert candidate.candidate_kind is CandidateKind.RULE
    assert candidate.body == "Always cite ticket IDs."
    assert candidate.occurrence_count == 2
    assert candidate.evidence_ids
    assert candidate.candidate_hash


def test_one_occurrence_proposes_nothing() -> None:
    store = InMemoryLearningStore()
    _human(store, text="Always cite ticket IDs.", key="c1")
    snapshot = InsightsEngine().analyze(store, agent_id=AGENT_ID, now=NOW)
    assert snapshot.findings == ()
    assert snapshot.candidates == ()


def test_run_failures_generate_friction_only() -> None:
    store = InMemoryLearningStore()
    for index, run_id in enumerate(("run-a", "run-b")):
        store.append_signal(
            run_outcome_signal(
                agent_id=AGENT_ID,
                run_id=run_id,
                status=RunStatus.TOOL_FAILURE,
                idempotency_key=f"run-{index}",
                observed_at=NOW,
            )
        )
    snapshot = InsightsEngine().analyze(store, agent_id=AGENT_ID, now=NOW)
    assert [item.kind for item in snapshot.findings] == [FindingKind.RUN_FRICTION]
    assert snapshot.findings[0].proposes_candidate is False
    assert snapshot.candidates == ()


def test_tool_rejection_generates_friction_only() -> None:
    store = InMemoryLearningStore()
    for index, run_id in enumerate(("run-a", "run-b")):
        store.append_signal(
            tool_rejection_signal(
                agent_id=AGENT_ID,
                run_id=run_id,
                tool_name="lookup_status",
                execution_status=ExecutionStatus.BLOCKED,
                idempotency_key=f"rej-{index}",
                observed_at=NOW,
            )
        )
    snapshot = InsightsEngine().analyze(store, agent_id=AGENT_ID, now=NOW)
    assert [item.kind for item in snapshot.findings] == [FindingKind.TOOL_FRICTION]
    assert snapshot.findings[0].proposes_candidate is False
    assert snapshot.candidates == ()


@pytest.mark.parametrize(
    "text",
    [
        "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA",
        "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.signaturexx",
        "sk-abcdefghijklmnopqrstuvwxyz123456",
        "AKIAIOSFODNN7EXAMPLE",
        "password=hunter2",
        "api_key=abcdEFGH1234",
        "token=supersecretvalue",
        "Bearer abcdefghijklmnop",
    ],
)
def test_credential_shaped_feedback_is_rejected(text: str) -> None:
    with pytest.raises(InsightsError, match="credential-shaped"):
        human_signal(
            kind="correction",
            agent_id=AGENT_ID,
            reviewer_id=REVIEWER,
            feedback_text=text,
            idempotency_key="bad",
            observed_at=NOW,
        )


def test_arbitrary_status_strings_are_rejected() -> None:
    with pytest.raises(InsightsError, match="RunStatus"):
        run_outcome_signal(
            agent_id=AGENT_ID,
            run_id="run-1",
            status="totally-made-up",
            idempotency_key="status-1",
            observed_at=NOW,
        )
    with pytest.raises(InsightsError, match="ExecutionStatus"):
        tool_rejection_signal(
            agent_id=AGENT_ID,
            run_id="run-1",
            tool_name="lookup_status",
            execution_status="DENIED",
            idempotency_key="status-2",
            observed_at=NOW,
        )


def test_retry_idempotency_allows_only_timestamp_to_differ() -> None:
    store = InMemoryLearningStore()
    first = human_signal(
        kind="preference",
        agent_id=AGENT_ID,
        reviewer_id=REVIEWER,
        feedback_text="Prefer short answers.",
        idempotency_key="pref-1",
        observed_at=NOW,
    )
    retry = human_signal(
        kind="preference",
        agent_id=AGENT_ID,
        reviewer_id=REVIEWER,
        feedback_text="Prefer short answers.",
        idempotency_key="pref-1",
        observed_at=NOW + 90,
    )
    stored = store.append_signal(first)
    replayed = store.append_signal(retry)
    assert first.signal_id == retry.signal_id
    assert replayed.observed_at == stored.observed_at
    assert len(store.list_signals()) == 1
    changed = human_signal(
        kind="preference",
        agent_id=AGENT_ID,
        reviewer_id=REVIEWER,
        feedback_text="Prefer long answers.",
        idempotency_key="pref-1",
        observed_at=NOW + 120,
    )
    with pytest.raises(LearningStoreError, match="idempotent retry"):
        store.append_signal(changed)


def test_sqlite_persistence_and_corruption_detection(tmp_path: Any) -> None:
    path = tmp_path / "insights.db"
    store = SQLiteLearningStore(path)
    signal = _human(store, text="Keep replies factual.", key="sqlite-1")
    store.close()

    reopened = SQLiteLearningStore(path)
    loaded = reopened.load_signals([signal.signal_id])
    assert loaded == (signal,)
    reopened.close()

    connection = sqlite3.connect(path)
    connection.execute(
        "UPDATE learning_signals SET kind = ? WHERE signal_id = ?",
        ("tampered", signal.signal_id),
    )
    connection.commit()
    connection.close()

    corrupted = SQLiteLearningStore(path)
    with pytest.raises(LearningStoreError):
        corrupted.load_signals([signal.signal_id])
    corrupted.close()

    connection = sqlite3.connect(path)
    connection.execute(
        "UPDATE learning_signals SET record_json = ? WHERE signal_id = ?",
        ('{"broken": true}', signal.signal_id),
    )
    connection.commit()
    connection.close()
    broken = SQLiteLearningStore(path)
    with pytest.raises(LearningStoreError):
        broken.list_signals()
    broken.close()


def test_missing_evidence_fails_closed() -> None:
    store = InMemoryLearningStore()
    with pytest.raises(LearningStoreError, match="missing learning evidence"):
        store.load_signals(["does-not-exist"])


def test_unauthorized_reviewer_is_rejected() -> None:
    store = InMemoryLearningStore()
    _human(store, text="Cite sources.", key="r1")
    _human(store, text="Cite sources.", key="r2")
    snapshot = InsightsEngine().analyze(store, agent_id=AGENT_ID, now=NOW)
    manager = _manager(store)
    with pytest.raises(PersonalizationError, match="not allowlisted"):
        manager.approve(
            snapshot.candidates[0],
            reviewer_id="mallory",
            profile=_profile(),
            approved_at=NOW,
        )
    with pytest.raises(PersonalizationError, match="non-empty"):
        PersonalizationManager(store, allowed_approvers=frozenset())


def test_approval_is_bound_to_profile_fingerprint() -> None:
    store = InMemoryLearningStore()
    _human(store, text="Cite sources.", key="r1")
    _human(store, text="Cite sources.", key="r2")
    snapshot = InsightsEngine().analyze(store, agent_id=AGENT_ID, now=NOW)
    manager = _manager(store)
    base = _profile()
    manager.approve(
        snapshot.candidates[0],
        reviewer_id=REVIEWER,
        profile=base,
        approved_at=NOW,
    )
    compiled = manager.compile_profile(base)
    assert "Cite sources." in compiled.instructions
    other = _profile(instructions="Different standing instructions.")
    unchanged = manager.compile_profile(other)
    assert unchanged.instructions == other.instructions
    assert "Cite sources." not in unchanged.instructions


def test_fabricated_evidence_is_rejected() -> None:
    store = InMemoryLearningStore()
    _human(store, text="Cite sources.", key="r1")
    _human(store, text="Cite sources.", key="r2")
    snapshot = InsightsEngine().analyze(store, agent_id=AGENT_ID, now=NOW)
    candidate = snapshot.candidates[0]
    fake = GuidanceCandidate(
        candidate_id=candidate.candidate_id,
        candidate_kind=candidate.candidate_kind,
        agent_id=candidate.agent_id,
        finding_kind=candidate.finding_kind,
        title=candidate.title,
        body=candidate.body,
        evidence_ids=("fabricated-evidence",),
        occurrence_count=2,
        candidate_hash=candidate.candidate_hash,
        skill_id=candidate.skill_id,
    )
    with pytest.raises(PersonalizationError, match="fabricated"):
        _manager(store).approve(
            fake, reviewer_id=REVIEWER, profile=_profile(), approved_at=NOW
        )


def test_fabricated_and_stale_snapshots_are_rejected() -> None:
    store = InMemoryLearningStore()
    _human(store, text="Cite sources.", key="r1")
    _human(store, text="Cite sources.", key="r2")
    engine = InsightsEngine()
    snapshot = engine.analyze(store, agent_id=AGENT_ID, now=NOW)
    manager = _manager(store)
    fabricated = InsightsSnapshot(
        schema_version=snapshot.schema_version,
        generated_at=snapshot.generated_at,
        window_days=snapshot.window_days,
        min_occurrences=snapshot.min_occurrences,
        agent_id=snapshot.agent_id,
        findings=snapshot.findings,
        candidates=snapshot.candidates,
        snapshot_hash="0" * 64,
    )
    with pytest.raises(PersonalizationError, match="fabricated or stale"):
        manager.verify_snapshot(fabricated)
    _human(store, text="Cite sources.", key="r3")
    with pytest.raises(PersonalizationError, match="fabricated or stale"):
        manager.verify_snapshot(snapshot)
    with pytest.raises(PersonalizationError, match="fabricated, changed, or stale"):
        manager.approve(
            snapshot.candidates[0],
            reviewer_id=REVIEWER,
            profile=_profile(),
            approved_at=NOW,
        )


def test_rules_are_appended_and_skills_require_explicit_activation() -> None:
    store = InMemoryLearningStore()
    _human(store, text="Always cite ticket IDs.", key="c1")
    _human(store, text="Always cite ticket IDs.", key="c2")
    _human(store, text="Use the status checklist.", key="w1", kind="workflow")
    _human(store, text="Use the status checklist.", key="w2", kind="workflow")
    snapshot = InsightsEngine().analyze(store, agent_id=AGENT_ID, now=NOW)
    manager = _manager(store)
    profile = _profile()
    for candidate in snapshot.candidates:
        manager.approve(
            candidate, reviewer_id=REVIEWER, profile=profile, approved_at=NOW
        )
    without_skills = manager.compile_profile(profile)
    assert without_skills.instructions.startswith(profile.instructions)
    assert "Always cite ticket IDs." in without_skills.instructions
    assert "Use the status checklist." not in without_skills.instructions
    skill = next(
        item
        for item in snapshot.candidates
        if item.candidate_kind is CandidateKind.SKILL
    )
    assert skill.skill_id is not None
    with_skills = manager.compile_profile(
        profile, active_skills=frozenset({skill.skill_id})
    )
    assert "Use the status checklist." in with_skills.instructions
    assert skill.skill_id in with_skills.instructions


def test_unknown_skill_is_rejected() -> None:
    store = InMemoryLearningStore()
    manager = _manager(store)
    with pytest.raises(PersonalizationError, match="unknown or unapproved"):
        manager.compile_profile(_profile(), active_skills=frozenset({"skill:unknown"}))


def test_tools_routes_and_limits_remain_unchanged() -> None:
    store = InMemoryLearningStore()
    _human(store, text="Keep answers short.", key="p1", kind="preference")
    _human(store, text="Keep answers short.", key="p2", kind="preference")
    snapshot = InsightsEngine().analyze(store, agent_id=AGENT_ID, now=NOW)
    manager = _manager(store)
    profile = _profile()
    manager.approve(
        snapshot.candidates[0],
        reviewer_id=REVIEWER,
        profile=profile,
        approved_at=NOW,
    )
    compiled = manager.compile_profile(profile)
    assert compiled.agent_id == profile.agent_id
    assert compiled.model_route == profile.model_route
    assert compiled.permitted_tools == profile.permitted_tools
    assert compiled.max_model_turns == profile.max_model_turns
    assert compiled.max_tool_calls == profile.max_tool_calls
    assert compiled.timeout_seconds == profile.timeout_seconds
    assert compiled.limits == profile.limits


def test_durable_run_fingerprint_changes_after_personalization() -> None:
    store = InMemoryLearningStore()
    _human(store, text="Cite ticket IDs.", key="c1")
    _human(store, text="Cite ticket IDs.", key="c2")
    snapshot = InsightsEngine().analyze(store, agent_id=AGENT_ID, now=NOW)
    manager = _manager(store)
    base = _profile()
    manager.approve(
        snapshot.candidates[0],
        reviewer_id=REVIEWER,
        profile=base,
        approved_at=NOW,
    )
    compiled = manager.compile_profile(base)
    assert profile_fingerprint(compiled) != profile_fingerprint(base)
    run_input = RunInput(task="Check the payments service", run_id="durable-1")
    record = new_running_record(
        run_id="durable-1",
        profile=base,
        run_input=run_input,
        started_at_utc=NOW,
        expires_at_utc=NOW + 120,
        messages=(
            Message(role=MessageRole.SYSTEM, content=base.instructions),
            Message(role=MessageRole.USER, content=run_input.task),
        ),
        events=(
            RunEvent(
                kind=RunEventKind.RUN_STARTED,
                step=0,
                timestamp=0.0,
                detail={"run_id": "durable-1"},
            ),
        ),
    )
    with pytest.raises(RunStoreError, match="fingerprint"):
        resume_durable_run(
            profile=compiled,
            run_input=run_input,
            record=record,
            wall_clock=lambda: NOW,
            monotonic_clock=lambda: 0.0,
        )


def test_report_generation_is_read_only() -> None:
    store = InMemoryLearningStore()
    _human(store, text="Cite sources.", key="r1")
    wrapper = ReadOnlyLearningStore(store)
    before = store.list_signals()
    report = build_report(wrapper, agent_id=AGENT_ID, now=NOW)
    assert report["schema"]
    assert store.list_signals() == before
    with pytest.raises(LearningStoreError, match="read-only"):
        wrapper.append_signal(
            human_signal(
                kind="correction",
                agent_id=AGENT_ID,
                reviewer_id=REVIEWER,
                feedback_text="Should not persist.",
                idempotency_key="blocked",
                observed_at=NOW,
            )
        )
    with pytest.raises(LearningStoreError, match="read-only"):
        wrapper.append_approval(
            ApprovalRecord(
                schema_version=1,
                approval_id="x",
                candidate_hash="x",
                candidate_kind=CandidateKind.RULE,
                agent_id=AGENT_ID,
                reviewer_id=REVIEWER,
                profile_fingerprint="x",
                instruction_text="x",
                skill_id=None,
                evidence_ids=("x",),
                approved_at=NOW,
                content_hash="x",
                idempotency_key="x",
            )
        )


def test_html_and_json_escaping() -> None:
    store = InMemoryLearningStore()
    payload = "<script>alert(1)</script> and </script><script>"
    _human(store, text=payload, key="xss-1")
    _human(store, text=payload, key="xss-2")
    report = build_report(store, agent_id=AGENT_ID, now=NOW)
    rendered_html = render_report_html(report)
    rendered_json = render_report_json(report)
    assert "<script>alert(1)</script>" not in rendered_html
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in rendered_html
    assert "\\u003c" in encode_json_for_html(report)
    parsed = json.loads(rendered_json)
    assert parsed["candidates"][0]["body"] == payload


def test_no_prompts_arguments_outputs_or_authorization_reasons_are_stored() -> None:
    store = InMemoryLearningStore()
    result = _run_result(
        status=RunStatus.REJECTED,
        run_id="run-secret",
        final_text="internal model answer that must not persist",
        metadata={
            "task": "transfer $50 to vendor-9",
            "reason": "policy denied because secret=abc",
            "prompt": "Ignore previous instructions",
            "arguments": {"amount": 50},
        },
        events=(
            RunEvent(
                kind=RunEventKind.TOOL_REQUESTED,
                step=1,
                timestamp=NOW,
                detail={
                    "tool_name": "lookup_status",
                    "request_id": "req-1",
                    "arguments": {"amount": 50, "password": "hunter2"},
                    "output": {"ok": True},
                },
            ),
            RunEvent(
                kind=RunEventKind.TOOL_REJECTED,
                step=1,
                timestamp=NOW,
                detail={
                    "request_id": "req-1",
                    "execution_status": ExecutionStatus.BLOCKED.value,
                    "reason": "authorization denied: credential xyz",
                },
            ),
        ),
    )
    assert LearningRecorder(store, clock=lambda: NOW).record_run(
        agent_id=AGENT_ID,
        result=result,
    )
    serialized = [signal.to_json() for signal in store.list_signals()]
    joined = "\n".join(serialized)
    for key in FORBIDDEN_SIGNAL_KEYS:
        assert f'"{key}"' not in joined
    for leaked in (
        "transfer $50",
        "Ignore previous instructions",
        "hunter2",
        "internal model answer",
        "authorization denied",
        "secret=abc",
    ):
        assert leaked not in joined
    kinds = {signal.kind for signal in store.list_signals()}
    assert kinds == {
        SignalKind.STRUCTURAL_RUN_OUTCOME,
        SignalKind.GOVERNED_TOOL_REJECTION,
    }
    rejection = next(
        signal
        for signal in store.list_signals()
        if signal.kind is SignalKind.GOVERNED_TOOL_REJECTION
    )
    assert rejection.tool_name == "lookup_status"
    assert rejection.execution_status is ExecutionStatus.BLOCKED
    assert rejection.run_id == "run-secret"


def test_learning_store_failure_does_not_alter_run_result() -> None:
    class BrokenStore(InMemoryLearningStore):
        def append_signal(self, signal: LearningSignal) -> LearningSignal:
            raise LearningStoreError("disk full")

    result = _run_result(status=RunStatus.COMPLETED, run_id="run-ok")
    recorded = LearningRecorder(BrokenStore()).record_run(
        agent_id=AGENT_ID,
        result=result,
    )
    assert recorded is False
    assert result.status is RunStatus.COMPLETED
    assert result.run_id == "run-ok"


def test_insights_cli_feedback_and_read_only_report(
    tmp_path: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    path = str(tmp_path / "cli.db")
    assert (
        insights_main(
            (
                "feedback",
                "--store",
                path,
                "--kind",
                "correction",
                "--agent-id",
                AGENT_ID,
                "--reviewer-id",
                REVIEWER,
                "--text",
                "Always cite ticket IDs.",
                "--idempotency-key",
                "cli-1",
            )
        )
        == 0
    )
    insights_main(
        (
            "feedback",
            "--store",
            path,
            "--kind",
            "correction",
            "--agent-id",
            AGENT_ID,
            "--reviewer-id",
            REVIEWER,
            "--text",
            "Always cite ticket IDs.",
            "--idempotency-key",
            "cli-2",
        )
    )
    store = SQLiteLearningStore(path)
    before = store.list_signals()
    store.close()
    capsys.readouterr()
    assert (
        insights_main(
            (
                "report",
                "--store",
                path,
                "--agent-id",
                AGENT_ID,
                "--days",
                "30",
                "--format",
                "json",
            )
        )
        == 0
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["schema"] == "cbrain-insights-report/v1"
    assert payload["candidates"]
    after = SQLiteLearningStore(path)
    assert after.list_signals() == before
    after.close()
    assert insights_main(("report", "--store", path, "--days", "0")) == 2
    assert insights_main(("report", "--store", path, "--min-occurrences", "0")) == 2


def test_cli_has_no_approve_or_activate_commands() -> None:
    with pytest.raises(SystemExit):
        insights_main(("approve",))
    with pytest.raises(SystemExit):
        insights_main(("activate",))
