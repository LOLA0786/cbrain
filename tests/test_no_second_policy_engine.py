"""Adapters, execution, and the agent loop must not import the eval gateway."""

from __future__ import annotations

import ast
from pathlib import Path

FORBIDDEN_ROOTS = (
    Path("cbrain/adapters"),
    Path("cbrain/execution"),
    Path("cbrain/agent"),
)
FORBIDDEN_MODULES = (
    "cbrain.evaluation.company_gateway",
    "cbrain.company.governance",
)


def test_production_packages_do_not_import_company_test_gateway() -> None:
    offenders: list[str] = []
    for root in FORBIDDEN_ROOTS:
        for path in root.rglob("*.py"):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name in FORBIDDEN_MODULES or alias.name.startswith(
                            tuple(f"{module}." for module in FORBIDDEN_MODULES)
                        ):
                            offenders.append(f"{path}: import {alias.name}")
                if isinstance(node, ast.ImportFrom) and node.module:
                    if node.module in FORBIDDEN_MODULES or node.module.startswith(
                        tuple(f"{module}." for module in FORBIDDEN_MODULES)
                    ):
                        offenders.append(f"{path}: from {node.module}")
                    if node.module.endswith("company_gateway") or node.module.endswith(
                        "governance"
                    ):
                        names = {alias.name for alias in node.names}
                        if "CompanyRiskGateway" in names:
                            offenders.append(
                                f"{path}: from {node.module} import CompanyRiskGateway"
                            )
    assert offenders == []
