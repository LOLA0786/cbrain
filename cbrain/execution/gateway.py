"""The concrete execution gateway.

This is the component `GovernedRuntime` has been calling into a Protocol for.
It turns a PrivateVault ALLOW into an actual execution by walking the full
evidence chain, and refuses to return EXECUTED unless every stage verified.

Failure classification is the load-bearing part. Anything that fails before the
tool handler is entered is a proven pre-execution failure (CONTROL_FAILURE,
retryable). Anything that fails once the handler may have run is INDETERMINATE
and never retryable, because the effect may already exist.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, Protocol, cast

from cbrain.adapters.privatevault import (
    PrivateVaultDecision,
    PrivateVaultVerdict,
)
from cbrain.adapters.privatevault_claim import (
    PrivateVaultAuthorizationClaimCoordinator,
)
from cbrain.adapters.privatevault_execution import (
    ExecutionAuthorizationBinding,
    PrivateVaultAgentDNAVerifier,
)
from cbrain.contracts import ActionIntent, ExecutionStatus, GovernedExecution
from cbrain.dispatch import PreparedDispatch
from cbrain.execution.transport import (
    DispatchTransport,
    HandlerNotInvoked,
)
from cbrain.ports import ToolHandler


class ExecutionGatewayError(RuntimeError):
    """The gateway could not complete a governed execution."""


@dataclass(frozen=True, slots=True)
class PlannedDispatch:
    """Everything the permit must commit to, built before authorization."""

    action: Mapping[str, Any]
    prepared: PreparedDispatch


class DispatchPlanner(Protocol):
    def plan(self, action: ActionIntent) -> PlannedDispatch:
        """Translate a governed intent into exact outbound bytes."""


class DecisionClient(Protocol):
    def decide(self, action: ActionIntent) -> PrivateVaultDecision:
        """Return the authoritative PrivateVault decision."""


@dataclass(frozen=True, slots=True)
class IssuedAuthorization:
    authorization: Mapping[str, Any]
    trust_bundle: Mapping[str, Any]
    binding_digests: Mapping[str, Any]


class AuthorizationIssuer(Protocol):
    def issue(
        self,
        *,
        action: ActionIntent,
        decision: PrivateVaultDecision,
        planned: PlannedDispatch,
    ) -> IssuedAuthorization:
        """Mint a signed, single-use execution authorization."""


class ClosureWriter(Protocol):
    def seal(
        self,
        *,
        authorization: Mapping[str, Any],
        witness: Mapping[str, Any],
        trust_bundle: Mapping[str, Any],
        dispatch_outcome: str,
        response_status: str,
        response_bytes: bytes,
        effect_state: str,
        idempotency_key_digest: str,
    ) -> Mapping[str, Any]:
        """Build and sign the closure record."""


class PrivateVaultExecutionGateway:
    """Authorize, dispatch once, witness, close, and return evidence state."""

    def __init__(
        self,
        *,
        decision_client: DecisionClient,
        planner: DispatchPlanner,
        issuer: AuthorizationIssuer,
        claim_coordinator: PrivateVaultAuthorizationClaimCoordinator,
        verifier: PrivateVaultAgentDNAVerifier,
        transport: DispatchTransport,
        closure_writer: ClosureWriter,
        clock: Callable[[], str],
        witness_id_factory: Callable[[ActionIntent], str],
    ) -> None:
        self._decision_client = decision_client
        self._planner = planner
        self._issuer = issuer
        self._claim_coordinator = claim_coordinator
        self._verifier = verifier
        self._transport = transport
        self._closure_writer = closure_writer
        self._clock = clock
        self._witness_id_factory = witness_id_factory

    def decide_and_execute(
        self,
        action: ActionIntent,
        handler: ToolHandler,
    ) -> GovernedExecution:
        handler_entered = False

        try:
            decision = self._decision_client.decide(action)
        except Exception as exc:
            return _control_failure(
                action,
                f"decision_unavailable:{type(exc).__name__}",
            )

        if decision.verdict is PrivateVaultVerdict.BLOCK:
            return GovernedExecution(
                status=ExecutionStatus.BLOCKED,
                request_id=action.request_id,
                tool_executed=False,
                reason=f"privatevault_block:{decision.triggered_by}",
                decision_id=_decision_id(decision),
            )

        if decision.verdict is PrivateVaultVerdict.REQUIRE_APPROVAL:
            return GovernedExecution(
                status=ExecutionStatus.REVIEW_REQUIRED,
                request_id=action.request_id,
                tool_executed=False,
                reason=f"privatevault_review:{decision.triggered_by}",
                decision_id=_decision_id(decision),
            )

        if decision.verdict is not PrivateVaultVerdict.ALLOW:
            return _control_failure(action, "unknown_verdict")

        try:
            planned = self._planner.plan(action)
            issued = self._issuer.issue(
                action=action,
                decision=decision,
                planned=planned,
            )
            binding = _bind(action, planned, issued)
            verified_authorization = self._claim_coordinator.verify_and_claim(
                authorization=issued.authorization,
                trust_bundle=issued.trust_bundle,
                binding=binding,
            )
        except Exception as exc:
            return _control_failure(
                action,
                f"authorization_refused:{type(exc).__name__}",
            )

        # The permit is now consumed. Every path below is post-claim.
        def run_handler(arguments: Mapping[str, Any]) -> Any:
            nonlocal handler_entered
            handler_entered = True
            return handler(arguments)

        transport = _rebind_handler(self._transport, run_handler)

        try:
            result = transport.dispatch(
                authorization=issued.authorization,
                trust_bundle=issued.trust_bundle,
                action=planned.action,
                prepared=planned.prepared,
                witness_id=self._witness_id_factory(action),
                observed_at=self._clock(),
                attempt=1,
            )
        except HandlerNotInvoked as exc:
            return _control_failure(
                action,
                f"dispatch_refused:{type(exc).__name__}",
                decision_id=_decision_id(decision),
            )
        except Exception as exc:
            return _indeterminate(
                action,
                f"dispatch_failed:{type(exc).__name__}",
                handler_entered,
                decision_id=_decision_id(decision),
            )

        try:
            verified_dispatch = self._verifier.verify_dispatch(
                verified_authorization=verified_authorization,
                witness=result.witness,
            )
            closure = self._closure_writer.seal(
                authorization=issued.authorization,
                witness=result.witness,
                trust_bundle=issued.trust_bundle,
                dispatch_outcome=result.dispatch_outcome,
                response_status=result.response_status,
                response_bytes=result.response_bytes,
                effect_state=result.effect_state,
                idempotency_key_digest=(
                    planned.prepared.dispatch["idempotency_key_digest"]
                ),
            )
            self._verifier.verify_closure(
                verified_dispatch=verified_dispatch,
                closure=closure,
            )
        except Exception as exc:
            return _indeterminate(
                action,
                f"closure_unproven:{type(exc).__name__}",
                handler_entered,
                decision_id=_decision_id(decision),
            )

        if not handler_entered:
            return _indeterminate(
                action,
                "closure_without_handler_entry",
                handler_entered,
                decision_id=_decision_id(decision),
            )

        return GovernedExecution(
            status=ExecutionStatus.EXECUTED,
            request_id=action.request_id,
            tool_executed=True,
            reason="execution_closed",
            decision_id=_decision_id(decision),
            output=result.output,
        )


def _rebind_handler(
    transport: DispatchTransport,
    runner: Callable[[Mapping[str, Any]], Any],
) -> DispatchTransport:
    """Give the transport this request's single-entry handler."""

    bound = getattr(transport, "with_handler", None)
    if callable(bound):
        rebound = bound(runner)
        return cast(DispatchTransport, rebound)

    raise ExecutionGatewayError("dispatch transport cannot bind a single-entry handler")


def _bind(
    action: ActionIntent,
    planned: PlannedDispatch,
    issued: IssuedAuthorization,
) -> ExecutionAuthorizationBinding:
    digests = issued.binding_digests

    return ExecutionAuthorizationBinding.capture(
        request_id=action.request_id,
        action=planned.action,
        prepared_dispatch=planned.prepared,
        decision_receipt_digest=digests["decision_receipt_digest"],
        authority_receipt_digest=digests["authority_receipt_digest"],
        approval_artifact_digest=digests.get("approval_artifact_digest"),
        state_snapshot_digest=digests["state_snapshot_digest"],
        policy_bundle_digest=digests["policy_bundle_digest"],
        obligations_digest=digests["obligations_digest"],
        at_time=digests["at_time"],
    )


def _decision_id(decision: PrivateVaultDecision) -> str | None:
    try:
        value = decision.record.get("decision_id")
    except Exception:
        return None

    return value if isinstance(value, str) and value else None


def _control_failure(
    action: ActionIntent,
    reason: str,
    decision_id: str | None = None,
) -> GovernedExecution:
    return GovernedExecution(
        status=ExecutionStatus.CONTROL_FAILURE,
        request_id=action.request_id,
        tool_executed=False,
        reason=reason,
        decision_id=decision_id,
        retryable=True,
    )


def _indeterminate(
    action: ActionIntent,
    reason: str,
    handler_entered: bool,
    decision_id: str | None = None,
) -> GovernedExecution:
    if not handler_entered:
        return _control_failure(action, reason, decision_id)

    return GovernedExecution(
        status=ExecutionStatus.INDETERMINATE,
        request_id=action.request_id,
        tool_executed=None,
        reason=reason,
        decision_id=decision_id,
        retryable=False,
    )


__all__ = [
    "AuthorizationIssuer",
    "ClosureWriter",
    "DecisionClient",
    "DispatchPlanner",
    "ExecutionGatewayError",
    "IssuedAuthorization",
    "PlannedDispatch",
    "PrivateVaultExecutionGateway",
]
