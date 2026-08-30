"""CLI for recording sanitized human feedback and read-only insights reports."""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from .insights import (
    DEFAULT_WINDOW_DAYS,
    MIN_GUIDANCE_OCCURRENCES,
    InsightsError,
    LearningStoreError,
    build_report,
    human_signal,
    parse_human_feedback_kind,
    positive_int,
    render_report_html,
    render_report_json,
)
from .insights_store import ReadOnlyLearningStore, SQLiteLearningStore


def main(arguments: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="cbrain-insights",
        description=(
            "Record sanitized human feedback and generate read-only insights "
            "reports. This CLI never approves or activates guidance."
        ),
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    feedback = subcommands.add_parser(
        "feedback",
        help="Record a sanitized human correction, preference, or workflow",
    )
    feedback.add_argument("--store", required=True, help="SQLite learning store path")
    feedback.add_argument(
        "--kind",
        required=True,
        choices=("correction", "preference", "workflow"),
    )
    feedback.add_argument("--agent-id", required=True)
    feedback.add_argument("--reviewer-id", required=True)
    feedback.add_argument("--text", required=True)
    feedback.add_argument("--idempotency-key", required=True)
    feedback.add_argument("--run-id")

    report = subcommands.add_parser(
        "report",
        help="Generate a read-only JSON or HTML insights report",
    )
    report.add_argument("--store", required=True, help="SQLite learning store path")
    report.add_argument("--agent-id")
    report.add_argument("--days", type=int, default=DEFAULT_WINDOW_DAYS)
    report.add_argument(
        "--min-occurrences",
        type=int,
        default=MIN_GUIDANCE_OCCURRENCES,
    )
    report.add_argument("--format", choices=("json", "html"), default="json")
    report.add_argument("--output", help="Optional file path; defaults to stdout")

    parsed = parser.parse_args(arguments)
    try:
        if parsed.command == "feedback":
            return _write_feedback(parsed)
        return _write_report(parsed)
    except (InsightsError, LearningStoreError) as exc:
        print(str(exc), file=sys.stderr)
        return 2


def _write_feedback(parsed: argparse.Namespace) -> int:
    store = SQLiteLearningStore(parsed.store)
    try:
        signal = human_signal(
            kind=parse_human_feedback_kind(parsed.kind),
            agent_id=parsed.agent_id,
            reviewer_id=parsed.reviewer_id,
            feedback_text=parsed.text,
            idempotency_key=parsed.idempotency_key,
            observed_at=time.time(),
            run_id=parsed.run_id,
        )
        stored = store.append_signal(signal)
    finally:
        store.close()
    print(stored.signal_id)
    return 0


def _write_report(parsed: argparse.Namespace) -> int:
    days = positive_int(parsed.days, "days")
    min_occurrences = positive_int(parsed.min_occurrences, "min-occurrences")
    store = SQLiteLearningStore(parsed.store)
    try:
        report = build_report(
            ReadOnlyLearningStore(store),
            agent_id=parsed.agent_id,
            window_days=days,
            min_occurrences=min_occurrences,
        )
        rendered = (
            render_report_html(report)
            if parsed.format == "html"
            else render_report_json(report)
        )
    finally:
        store.close()
    if parsed.output:
        Path(parsed.output).write_text(rendered, encoding="utf-8")
    else:
        sys.stdout.write(rendered)
    return 0


__all__ = ["main"]
