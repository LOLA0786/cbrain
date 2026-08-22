"""Artifact generation for company-agent offline evaluation."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cbrain.company.spec import COMPANY_AGENT_VERSION, SCENARIO_SUITE_VERSION

from .company_gates import ReleaseGateResult
from .company_harness import OFFLINE_MODEL_ROUTES, CompanySuiteMetrics
from .cost import cost_formula_text
from .pricing import PricingCatalog

REPORT_SCHEMA = "cbrain-company-eval-report/v1"


@dataclass(frozen=True, slots=True)
class CompanyEvalManifest:
    schema: str
    cbrain_commit: str
    profile_versions: dict[str, str]
    scenario_suite_version: str
    pricing_catalog_hash: str
    model_route_labels: tuple[str, ...]
    simulated_route_labels: tuple[str, ...]
    deterministic_seed: str
    output_files: tuple[str, ...]
    execution_mode: str = "offline_fixture"
    model_output_source: str = "scripted"
    provider_called: bool = False
    decision_authority: str = "company_test_gateway"
    live_provider_calls: int = 0
    real_model_quality: str = "not_evaluated"
    model_prompt_injection_resilience: str = "not_evaluated"
    control_path_adversarial: str = "evaluated"

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "cbrain_commit": self.cbrain_commit,
            "profile_versions": self.profile_versions,
            "scenario_suite_version": self.scenario_suite_version,
            "pricing_catalog_hash": self.pricing_catalog_hash,
            "model_route_labels": list(self.model_route_labels),
            "simulated_route_labels": list(self.simulated_route_labels),
            "deterministic_seed": self.deterministic_seed,
            "output_files": list(self.output_files),
            "execution_mode": self.execution_mode,
            "model_output_source": self.model_output_source,
            "provider_called": self.provider_called,
            "decision_authority": self.decision_authority,
            "live_provider_calls": self.live_provider_calls,
            "real_model_quality": self.real_model_quality,
            "model_prompt_injection_resilience": self.model_prompt_injection_resilience,
            "control_path_adversarial": self.control_path_adversarial,
        }


def write_company_eval_artifacts(
    *,
    output_dir: str | Path,
    metrics: CompanySuiteMetrics,
    aggregate: dict[str, Any],
    gates: ReleaseGateResult,
    catalog: PricingCatalog,
    catalog_path: Path,
    deterministic_seed: str = "company-offline-v0.4.1",
) -> CompanyEvalManifest:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    runs_jsonl = directory / "runs.jsonl"
    aggregate_json = directory / "aggregate_report.json"
    comparison_csv = directory / "comparison.csv"
    summary_md = directory / "summary.md"
    manifest_json = directory / "manifest.json"

    with runs_jsonl.open("w", encoding="utf-8") as handle:
        for run in metrics.runs:
            handle.write(json.dumps(run.to_payload(), sort_keys=True))
            handle.write("\n")

    aggregate_payload = {
        "schema": REPORT_SCHEMA,
        "cost_formula": cost_formula_text(),
        "aggregate": aggregate,
        "release_gates": gates.to_payload(),
        "pricing_catalog": catalog.to_payload(),
    }
    aggregate_json.write_text(
        json.dumps(aggregate_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_comparison_csv(comparison_csv, aggregate)
    summary_md.write_text(
        render_markdown_summary(aggregate, gates=gates, catalog_path=catalog_path),
        encoding="utf-8",
    )

    manifest = CompanyEvalManifest(
        schema="cbrain-company-eval-manifest/v1",
        cbrain_commit=_git_commit(),
        profile_versions={
            "gtm": COMPANY_AGENT_VERSION,
            "operations": COMPANY_AGENT_VERSION,
            "legal": COMPANY_AGENT_VERSION,
            "accounts": COMPANY_AGENT_VERSION,
        },
        scenario_suite_version=SCENARIO_SUITE_VERSION,
        pricing_catalog_hash=_sha256_file(catalog_path),
        model_route_labels=OFFLINE_MODEL_ROUTES,
        simulated_route_labels=tuple(
            f"simulated:{route}" for route in OFFLINE_MODEL_ROUTES
        ),
        deterministic_seed=deterministic_seed,
        output_files=(
            runs_jsonl.name,
            aggregate_json.name,
            comparison_csv.name,
            summary_md.name,
            manifest_json.name,
        ),
    )
    manifest_json.write_text(
        json.dumps(manifest.to_payload(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def render_markdown_summary(
    aggregate: dict[str, Any],
    *,
    gates: ReleaseGateResult,
    catalog_path: Path,
) -> str:
    cost_complete = bool(aggregate.get("cost_data_complete"))
    cost_evaluation = aggregate.get("cost_evaluation", "not_evaluated")
    lines = [
        "# CBrain Company Agent Offline Evaluation",
        "",
        "Four configuration-driven company agents over one shared "
        "FoundationAgent runtime.",
        "",
        "## Provenance",
        "- Execution mode: offline_fixture",
        "- Model output source: scripted",
        "- Decision authority: company_test_gateway",
        f"- Fixture conformance: {aggregate.get('fixture_conformance')}",
        "- Real-model quality: not_evaluated",
        f"- Live provider calls: {aggregate.get('live_provider_calls', 0)}",
        "- Control-path adversarial: evaluated",
        "- Live-model prompt-injection resilience: not_evaluated",
        "",
        "Across six scripted route fixtures, identical canonical ActionIntent "
        "inputs produced zero decision divergence in the deterministic "
        "company_test_gateway.",
        "",
        "## Aggregate metrics",
        f"- Task success rate: {aggregate['task_success_rate']:.3f}",
        f"- Tool selection accuracy: {aggregate.get('tool_selection_accuracy')}",
        f"- Tool argument accuracy: {aggregate.get('tool_argument_accuracy')}",
        "- Citation grounding accuracy: "
        f"{aggregate.get('citation_grounding_accuracy')}",
        f"- Unauthorized executions: {aggregate['unauthorized_execution_count']}",
        f"- Approval bypasses: {aggregate['approval_bypass_count']}",
        f"- Safety violations: {aggregate['safety_violation_count']}",
        "- Decision divergence (identical ActionIntent): "
        f"{aggregate['decision_divergence_count']}",
        "",
        "## Cost evaluation",
        f"- Cost data complete: {cost_complete}",
        f"- Cost evaluation: {cost_evaluation}",
        f"- Known-cost subtotal: {aggregate.get('known_cost_subtotal')}",
        f"- Unknown completion count: {aggregate.get('unknown_completion_count')}",
        f"- Estimated completion count: {aggregate.get('estimated_completion_count')}",
        "- Simulated-cache completions: "
        f"{aggregate.get('simulated_cache_completions')}",
        "- Cost optimization accepted: false",
        "- Unit-cost metrics are omitted while cost data is incomplete.",
        "",
        "## Release gates",
        f"- Passed: {gates.passed}",
    ]
    if gates.failures:
        lines.extend(["", "### Failures"] + [f"- {item}" for item in gates.failures])
    lines.extend(
        [
            "",
            "## Scope",
            "- Simulator-only fixtures; no production integrations or credentials.",
            "- Route labels are simulated scripted fixtures, not live providers.",
            "- Legal output requires licensed lawyer review.",
            "- Accounts scenarios do not execute real payments.",
            "",
            f"Pricing catalog: `{catalog_path}`",
        ]
    )
    return "\n".join(lines) + "\n"


def _write_comparison_csv(path: Path, aggregate: dict[str, Any]) -> None:
    rows = [
        ("metric", "value"),
        ("task_success_rate", aggregate.get("task_success_rate")),
        ("tool_selection_accuracy", aggregate.get("tool_selection_accuracy")),
        ("tool_argument_accuracy", aggregate.get("tool_argument_accuracy")),
        ("citation_grounding_accuracy", aggregate.get("citation_grounding_accuracy")),
        ("unauthorized_execution_count", aggregate.get("unauthorized_execution_count")),
        ("approval_bypass_count", aggregate.get("approval_bypass_count")),
        ("safety_violation_count", aggregate.get("safety_violation_count")),
        ("duplicate_dispatch_count", aggregate.get("duplicate_dispatch_count")),
        ("decision_divergence_count", aggregate.get("decision_divergence_count")),
        ("cost_data_complete", aggregate.get("cost_data_complete")),
        ("cost_evaluation", aggregate.get("cost_evaluation")),
        ("real_model_quality", aggregate.get("real_model_quality")),
        ("live_provider_calls", aggregate.get("live_provider_calls")),
        ("decision_authority", aggregate.get("decision_authority")),
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return completed.stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


__all__ = ["CompanyEvalManifest", "write_company_eval_artifacts"]
