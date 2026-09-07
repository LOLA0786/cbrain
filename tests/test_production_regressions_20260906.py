"""Production regression cases from the September 2026 source audit.

All model, gateway, and verification doubles remain inside tests.
These checks do not substitute for real PrivateVault conformance or live APIs.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from typing import Any

import pytest
from knowledge_fakes import DeterministicEmbeddingProvider, RuleBasedExtractor

from cbrain import ActionIntent, ExecutionStatus, GovernedExecution, GovernedRuntime
from cbrain.agent import (
    AgentProfile,
    FoundationAgent,
    GovernedTool,
    RunInput,
    RunStatus,
    ToolRegistry,
)
from cbrain.agent.durable import DurableRunState, profile_fingerprint
from cbrain.agent.insights import PersonalizationManager, ReviewerPrincipal
from cbrain.agent.insights_store import InMemoryLearningStore
from cbrain.agent.store_memory import InMemoryRunStore
from cbrain.knowledge import KnowledgeRuntime, RetrievalQuery, SourceDocument
from cbrain.models import CompletionRequest, ModelRouter, TextOutput, ToolCall

EPOCH = 1_783_000_000.0


class Clock:
    def __init__(self) -> None:
        self.now = 10.0
        self.cancelled = False


class ProposalModel:
    provider = "audit-test"
    model = "scripted"

    def __init__(self, during_call: Callable[[], None] | None = None) -> None:
        self.calls = 0
        self.during_call = during_call

    def complete(self, request: CompletionRequest) -> ToolCall | TextOutput:
        self.calls += 1
        if self.calls == 1:
            if self.during_call is not None:
                self.during_call()
            return ToolCall.capture(call_id="call-1", name="lookup", arguments={})
        return TextOutput("Finished.")


class RecordingGateway:
    independent_execution = False

    def __init__(self) -> None:
        self.actions: list[ActionIntent] = []

    def decide_and_execute(
        self, action: ActionIntent, handler: Any
    ) -> GovernedExecution:
        self.actions.append(action)
        return GovernedExecution(
            status=ExecutionStatus.EXECUTED,
            request_id=action.request_id,
            tool_executed=True,
            reason="audit_test_only",
            output=handler(action.arguments),
        )


def profile(*, required: bool = False) -> AgentProfile:
    return AgentProfile(
        agent_id="audit-agent",
        instructions="Look up a status and report it.",
        model_route="test",
        permitted_tools=frozenset({"lookup"}),
        max_model_turns=3,
        max_tool_calls=1,
        timeout_seconds=5.0,
        knowledge_required_for_tools=required,
    )


def agent(
    *,
    clock: Clock,
    gateway: RecordingGateway,
    required: bool = False,
    store: InMemoryRunStore | None = None,
    knowledge: Any = None,
    during_call: Callable[[], None] | None = None,
    route: str = "test",
) -> FoundationAgent:
    tool = GovernedTool(
        name="lookup",
        capability="status.read",
        description="Read status",
        input_schema={"type": "object", "properties": {}},
    )
    return FoundationAgent(
        profile=replace(profile(required=required), model_route=route),
        runtime=GovernedRuntime(gateway),
        model_router=ModelRouter({"test": ProposalModel(during_call)}),
        tools=ToolRegistry([tool]),
        handlers={"lookup": lambda args: {"status": "ok"}},
        clock=lambda: clock.now,
        wall_clock=lambda: EPOCH,
        cancelled=lambda: clock.cancelled,
        run_store=store,
        knowledge=knowledge,
    )


@pytest.mark.parametrize("durable", [False, True])
def test_required_knowledge_cannot_be_omitted(durable: bool) -> None:
    gateway = RecordingGateway()
    result = agent(
        clock=Clock(),
        gateway=gateway,
        required=True,
        store=InMemoryRunStore() if durable else None,
    ).run(RunInput(task="Check status"))
    assert not gateway.actions
    assert result.status is RunStatus.TOOL_FAILURE
    assert result.metadata["reason"] == "KNOWLEDGE_UNAVAILABLE"


@pytest.mark.parametrize("durable", [False, True])
@pytest.mark.parametrize("stop", ["deadline", "cancel"])
def test_no_new_dispatch_after_model_call_crosses_stop_boundary(
    durable: bool,
    stop: str,
) -> None:
    clock = Clock()
    gateway = RecordingGateway()

    def stop_during_call() -> None:
        if stop == "deadline":
            clock.now += 6.0
        else:
            clock.cancelled = True

    result = agent(
        clock=clock,
        gateway=gateway,
        during_call=stop_during_call,
        store=InMemoryRunStore() if durable else None,
    ).run(RunInput(task="Check status"))
    assert not gateway.actions
    expected = RunStatus.TIMED_OUT if stop == "deadline" else RunStatus.CANCELLED
    assert result.status is expected


@pytest.mark.parametrize("durable", [False, True])
def test_action_timestamp_is_wall_clock_time(durable: bool) -> None:
    gateway = RecordingGateway()
    result = agent(
        clock=Clock(),
        gateway=gateway,
        store=InMemoryRunStore() if durable else None,
    ).run(RunInput(task="Check status"))
    assert result.status is RunStatus.COMPLETED
    assert len(gateway.actions) == 1
    assert gateway.actions[0].timestamp == EPOCH


class SimulatedCrash(BaseException):
    pass


class CrashBeforeClaimStore(InMemoryRunStore):
    def __init__(self) -> None:
        super().__init__()
        self.crashed = False

    def claim_tool_dispatch(self, run_id: str, *, expected_version: int) -> Any:
        if not self.crashed:
            self.crashed = True
            raise SimulatedCrash()
        return super().claim_tool_dispatch(run_id, expected_version=expected_version)


def document(
    *,
    collection: str = "a",
    principals: frozenset[str] = frozenset({"alice"}),
    text: str = "Alice works at Acme",
    provenance: str = "operator-upload",
) -> SourceDocument:
    return SourceDocument(
        tenant_id="tenant",
        collection_id=collection,
        source_id="source",
        media_type="text/plain",
        text=text,
        acl_principals=principals,
        created_at=EPOCH,
        provenance=provenance,
    )


def query(*, collection: str = "a", principal: str = "alice") -> RetrievalQuery:
    return RetrievalQuery(
        tenant_id="tenant",
        collection_id=collection,
        principal_id=principal,
        text="Alice",
        top_k=4,
        created_at=EPOCH,
    )



def knowledge_runtime() -> KnowledgeRuntime:
    return KnowledgeRuntime(
        embeddings=DeterministicEmbeddingProvider(),
        extractor=RuleBasedExtractor(),
    )


def test_prepared_resume_rechecks_required_knowledge_before_claim() -> None:
    store = CrashBeforeClaimStore()
    gateway = RecordingGateway()
    knowledge = knowledge_runtime()
    knowledge.ingest(document())
    run_input = RunInput(
        task="Check status",
        run_id="prepared-knowledge",
        metadata={"tenant_id": "tenant", "collection_id": "a", "principal_id": "alice"},
    )
    with pytest.raises(SimulatedCrash):
        agent(
            clock=Clock(),
            gateway=gateway,
            store=store,
            knowledge=knowledge,
            required=True,
        ).run(run_input)
    assert (
        store.load(run_input.run_id or "").durable_state
        is DurableRunState.TOOL_PREPARED
    )
    knowledge.store.fail_reads = True
    result = agent(
        clock=Clock(), gateway=gateway, store=store, knowledge=knowledge, required=True
    ).run(run_input)
    assert not gateway.actions
    assert result.status is RunStatus.TOOL_FAILURE
    assert result.metadata["reason"] == "KNOWLEDGE_UNAVAILABLE"


def test_knowledge_requirement_changes_durable_fingerprint() -> None:
    assert profile_fingerprint(profile(required=False)) != profile_fingerprint(
        profile(required=True)
    )


class ReviewerVerifier:
    def verify(self, principal: ReviewerPrincipal) -> ReviewerPrincipal:
        return principal


def test_personalization_preserves_required_knowledge() -> None:
    manager = PersonalizationManager(
        InMemoryLearningStore(),
        reviewer_verifier=ReviewerVerifier(),
    )
    compiled = manager.compile_profile(profile(required=True))
    assert compiled.knowledge_required_for_tools is True


def test_acl_revocation_applies_when_text_has_not_changed() -> None:
    runtime = knowledge_runtime()
    first = runtime.ingest(document(principals=frozenset({"alice", "bob"})))
    assert runtime.retrieve(query(principal="bob")).hits
    second = runtime.ingest(document(principals=frozenset({"alice"})))
    assert not runtime.retrieve(query(principal="bob")).hits
    assert second.source_revision > first.source_revision
    assert runtime.retrieve(query(principal="alice")).hits


def test_provenance_change_creates_new_revision_without_text_change() -> None:
    runtime = knowledge_runtime()
    first = runtime.ingest(document(provenance="old-origin"))
    second = runtime.ingest(document(provenance="corrected-origin"))
    assert second.source_revision > first.source_revision
    assert runtime.retrieve(query()).hits[0].citation.provenance == "corrected-origin"


def test_same_source_in_separate_collections_does_not_overwrite_chunks() -> None:
    runtime = knowledge_runtime()
    first = runtime.ingest(document(collection="a"))
    second = runtime.ingest(document(collection="b"))
    assert set(first.chunk_ids).isdisjoint(second.chunk_ids)
    assert runtime.retrieve(query(collection="a")).hits
    assert runtime.retrieve(query(collection="b")).hits


def test_same_source_graph_nodes_are_tenant_keyed_not_collection_keyed() -> None:
    runtime = knowledge_runtime()
    first = runtime.ingest(document(collection="a"))
    second = runtime.ingest(document(collection="b"))
    assert set(first.chunk_ids).isdisjoint(second.chunk_ids)
    left = runtime.store.nodes_for_chunks(first.chunk_ids)
    right = runtime.store.nodes_for_chunks(second.chunk_ids)
    assert not left
    assert right


@pytest.mark.parametrize("durable", [False, True])
def test_unknown_model_route_returns_structured_failure(durable: bool) -> None:
    gateway = RecordingGateway()
    result = agent(
        clock=Clock(),
        gateway=gateway,
        route="missing",
        store=InMemoryRunStore() if durable else None,
    ).run(RunInput(task="Check status"))
    assert result.status is RunStatus.MODEL_FAILURE
    assert not gateway.actions


def test_claim_verification_uses_actual_claim_time() -> None:
    from test_privatevault_claim import Report, Store, _authorization, _binding

    from cbrain.adapters.privatevault_claim import (
        PrivateVaultAuthorizationClaimCoordinator,
    )
    from cbrain.adapters.privatevault_consumption import (
        PrivateVaultAuthorizationUseFactory,
    )
    from cbrain.adapters.privatevault_execution import (
        AgentDNAVerifiers,
        PrivateVaultAgentDNAVerifier,
    )

    times: list[object] = []

    def verifier(*args: object, **kwargs: object) -> object:
        times.append(kwargs.get("at_time"))
        return Report()

    coordinator = PrivateVaultAuthorizationClaimCoordinator(
        verifier=PrivateVaultAgentDNAVerifier(
            AgentDNAVerifiers(verifier, verifier, verifier)
        ),
        use_factory=PrivateVaultAuthorizationUseFactory(lambda _: "sha256:" + "0" * 64),
        consumption_store=Store([]),
    )
    actual_time = "2026-09-06T12:00:00Z"
    coordinator.verify_and_claim(
        authorization=_authorization(),
        trust_bundle={"trusted": True},
        binding=_binding(),
        claimed_at=actual_time,
    )
    assert times == [actual_time]
