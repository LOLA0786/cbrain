"""Artifact generation for agent evaluation and cost reports."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from cbrain.adapters.gbrain import GBRAIN_UPSTREAM_COMMIT, GBRAIN_VERSION

from .agent_harness import AgentSuiteMetrics
from .cost import cost_formula_text
from .pricing import PricingCatalog

REPORT_SCHEMA = "cbrain-agent-eval-report/v1"


@dataclass(frozen=True, slots=True)
class AgentEvalManifest:
    schema: str
    configuration_hashes: dict[str, str]
    pricing_catalog_hash: str
    pinned_components: dict[str, str]
    output_files: tuple[str, ...]

    def to_payload(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "configuration_hashes": self.configuration_hashes,
            "pricing_catalog_hash": self.pricing_catalog_hash,
            "pinned_components": self.pinned_components,
            "output_files": list(self.output_files),
        }


def write_agent_eval_artifacts(
    *,
    output_dir: str | Path,
    comparison: dict[str, Any],
    catalog: PricingCatalog,
    catalog_path: Path,
    baseline: AgentSuiteMetrics,
    optimized: AgentSuiteMetrics,
) -> AgentEvalManifest:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    runs_jsonl = directory / "runs.jsonl"
    aggregate_json = directory / "aggregate_report.json"
    comparison_csv = directory / "comparison.csv"
    summary_md = directory / "summary.md"
    manifest_json = directory / "manifest.json"

    with runs_jsonl.open("w", encoding="utf-8") as handle:
        for metrics in (baseline, optimized):
            for run in metrics.runs:
                handle.write(json.dumps(run.to_payload(), sort_keys=True))
                handle.write("\n")

    aggregate_payload = {
        "schema": REPORT_SCHEMA,
        "cost_formula": cost_formula_text(),
        "comparison": comparison,
        "pricing_catalog": catalog.to_payload(),
    }
    aggregate_json.write_text(
        json.dumps(aggregate_payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _write_comparison_csv(comparison_csv, comparison)
    summary_md.write_text(
        render_markdown_summary(comparison, catalog_path=catalog_path),
        encoding="utf-8",
    )

    catalog_hash = _sha256_file(catalog_path)
    manifest = AgentEvalManifest(
        schema="cbrain-agent-eval-manifest/v1",
        configuration_hashes={
            "baseline": _sha256_text(
                json.dumps(comparison["baseline"], sort_keys=True)
            ),
            "optimized": _sha256_text(
                json.dumps(comparison["optimized"], sort_keys=True)
            ),
        },
        pricing_catalog_hash=catalog_hash,
        pinned_components=pinned_component_metadata(),
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
    comparison: dict[str, Any],
    *,
    catalog_path: Path,
) -> str:
    baseline = comparison["baseline"]
    optimized = comparison["optimized"]
    lines = [
        "# CBrain Agent Evaluation Summary",
        "",
        "This report compares baseline and optimized offline agent evaluation runs.",
        "",
        "## Baseline",
        f"- Task success rate: {baseline['task_success_rate']:.3f}",
        f"- Safety violation rate: {baseline['safety_violation_rate']:.3f}",
        f"- Cost per successful task: {baseline['cost_per_successful_task']}",
        f"- Total cost status: {baseline['total_cost']['status']}",
        f"- Known cost subtotal: {baseline['total_cost'].get('known_cost_subtotal')}",
        f"- Cost data complete: {baseline['cost_data_complete']}",
        f"- Simulated cache completions: "
        f"{baseline['usage'].get('simulated_cache_completions', 0)}",
        "",
        "## Optimized",
        f"- Task success rate: {optimized['task_success_rate']:.3f}",
        f"- Safety violation rate: {optimized['safety_violation_rate']:.3f}",
        f"- Cost per successful task: {optimized['cost_per_successful_task']}",
        f"- Total cost status: {optimized['total_cost']['status']}",
        f"- Known cost subtotal: {optimized['total_cost'].get('known_cost_subtotal')}",
        f"- Cost data complete: {optimized['cost_data_complete']}",
        f"- Simulated cache completions: "
        f"{optimized['usage'].get('simulated_cache_completions', 0)}",
        "",
        "## Trade-off",
        f"- Optimization decision: {comparison.get('optimization_decision')}",
        f"- Optimization decision reason: "
        f"{comparison.get('optimization_decision_reason')}",
        f"- Optimization accepted: {comparison['optimization_accepted']}",
        f"- Cost per successful task improvement ratio: "
        f"{comparison['cost_per_successful_task_improvement_ratio']}",
        "",
        "## Pricing provenance",
        f"- Catalog file: `{catalog_path}`",
        f"- Cost formula: {cost_formula_text()}",
        "",
        "Unknown usage and missing pricing are reported as `cost_unknown`, never zero.",
        "Simulated cache savings use `cached_input_accounting=simulated_assumption`.",
        "Provider-measured, estimated, assumed-cache, and unknown "
        "usage remain distinct.",
    ]
    return "\n".join(lines) + "\n"


def pinned_component_metadata() -> dict[str, str]:
    return {
        "hermes": "cbrain-hermes launcher with mandatory cbrain_guard pre-tool hook",
        "gbrain_version": GBRAIN_VERSION,
        "gbrain_upstream_commit": GBRAIN_UPSTREAM_COMMIT,
        "privatevault": "decision authority via PrivateVault adapters",
    }


def _write_comparison_csv(path: Path, comparison: dict[str, Any]) -> None:
    rows = [
        ("metric", "baseline", "optimized"),
        (
            "task_success_rate",
            comparison["baseline"]["task_success_rate"],
            comparison["optimized"]["task_success_rate"],
        ),
        (
            "safety_violation_rate",
            comparison["baseline"]["safety_violation_rate"],
            comparison["optimized"]["safety_violation_rate"],
        ),
        (
            "cost_per_successful_task",
            comparison["baseline"]["cost_per_successful_task"],
            comparison["optimized"]["cost_per_successful_task"],
        ),
        (
            "total_cost",
            comparison["baseline"]["total_cost"]["total_cost"],
            comparison["optimized"]["total_cost"]["total_cost"],
        ),
        (
            "total_cost_status",
            comparison["baseline"]["total_cost"]["status"],
            comparison["optimized"]["total_cost"]["status"],
        ),
        (
            "cost_data_complete",
            comparison["baseline"]["cost_data_complete"],
            comparison["optimized"]["cost_data_complete"],
        ),
        (
            "known_cost_subtotal",
            comparison["baseline"]["total_cost"]["known_cost_subtotal"],
            comparison["optimized"]["total_cost"]["known_cost_subtotal"],
        ),
        (
            "optimization_decision",
            comparison.get("optimization_decision"),
            comparison.get("optimization_decision"),
        ),
        (
            "optimization_decision_reason",
            comparison.get("optimization_decision_reason"),
            comparison.get("optimization_decision_reason"),
        ),
        (
            "optimization_accepted",
            comparison["optimization_accepted"],
            comparison["optimization_accepted"],
        ),
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    return f"sha256:{digest}"


def _sha256_text(value: str) -> str:
    digest = hashlib.sha256(value.encode("utf-8")).hexdigest()
    return f"sha256:{digest}"


__all__ = [
    "AgentEvalManifest",
    "REPORT_SCHEMA",
    "render_markdown_summary",
    "write_agent_eval_artifacts",
]
