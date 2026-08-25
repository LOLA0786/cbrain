"""Offline demo for company-agent profiles."""

from __future__ import annotations

from cbrain.company.kinds import MATRIX_AGENT_KINDS
from cbrain.company.profiles import all_company_profiles
from cbrain.evaluation.company_gates import evaluate_release_gates
from cbrain.evaluation.company_harness import CompanyEvalHarness
from cbrain.evaluation.company_suites import all_company_eval_cases, plan_payload
from cbrain.evaluation.pricing import default_pricing_catalog_path, load_pricing_catalog


def main() -> int:
    pricing = load_pricing_catalog(default_pricing_catalog_path())
    cases = all_company_eval_cases()
    metrics = CompanyEvalHarness(
        catalog=pricing,
        cases=cases,
        model_routes=("offline",),
    ).run()
    gates = evaluate_release_gates(metrics)
    plan = plan_payload()

    print("CBrain company-agent offline demo")
    print("profiles:", ", ".join(spec.agent_id for spec in all_company_profiles()))
    print("total scenarios:", plan["total_cases"])
    print()
    print(f"{'agent':<12} {'category':<28} {'count':>5}")
    print("-" * 48)
    for agent, categories in sorted(plan["counts_by_agent"].items()):
        for category, count in sorted(categories.items()):
            print(f"{agent:<12} {category:<28} {count:>5}")
    print()
    print(f"task success rate: {metrics.task_success_rate:.3f}")
    print(f"release gates passed: {gates.passed}")
    for kind in MATRIX_AGENT_KINDS:
        agent_runs = [run for run in metrics.runs if run.agent_kind == kind.value]
        successes = sum(1 for run in agent_runs if run.success)
        print(f"  {kind.value}: {successes}/{len(agent_runs)}")
    return 0 if gates.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
