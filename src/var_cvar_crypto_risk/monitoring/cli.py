"""One-shot monitoring CLI for cron, launchd, or another external scheduler."""

from __future__ import annotations

import argparse
from datetime import date, datetime
import json
import os
from typing import Sequence
from uuid import UUID

from .database import create_monitoring_engine, create_session_factory
from .live_update import LiveMonitoringService
from .providers import default_provider_registry
from .repository import SqlAlchemyUnitOfWork


def _date(value: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected YYYY-MM-DD") from exc


def _timestamp(value: str) -> datetime:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("expected an ISO-8601 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise argparse.ArgumentTypeError("timestamp must include a timezone")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qcprf-monitor",
        description="Run one idempotent portfolio-monitoring update.",
    )
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--experiment-id", type=UUID)
    target.add_argument("--all-active", action="store_true")
    parser.add_argument("--requested-cutoff", type=_date)
    parser.add_argument("--as-of", type=_timestamp)
    parser.add_argument(
        "--database-url",
        help="Optional SQLAlchemy URL; prefer QCPRF_MONITORING_DATABASE_URL.",
    )
    parser.add_argument(
        "--code-version",
        default=os.getenv("QCPRF_CODE_VERSION", "working-tree"),
    )
    parser.add_argument("--calculation-version", default="valuation-v1")
    return parser


def _result_payload(result) -> dict:
    return {
        "experiment_id": str(result.experiment_id),
        "run_id": str(result.run_id),
        "status": result.final_status.value,
        "requested_cutoff": result.requested_cutoff.isoformat(),
        "actual_cutoff": (
            result.actual_cutoff.isoformat() if result.actual_cutoff else None
        ),
        "actual_source": result.actual_source,
        "processed_dates": result.processed_dates,
        "evaluated_forecasts": result.evaluated_forecasts,
        "warning_count": result.warning_count,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """Execute once and return a scheduler-friendly process status."""
    args = build_parser().parse_args(argv)
    engine = create_monitoring_engine(args.database_url)
    session_factory = create_session_factory(engine)

    def uow_factory():
        return SqlAlchemyUnitOfWork(session_factory)

    service = LiveMonitoringService(uow_factory, default_provider_registry())
    try:
        if args.all_active:
            result = service.update_all_active(
                requested_cutoff=args.requested_cutoff,
                as_of=args.as_of,
                code_version=args.code_version,
                calculation_version=args.calculation_version,
            )
            payload = {
                "completed": [_result_payload(item) for item in result.completed],
                "failed_experiment_ids": [
                    str(item) for item in result.failed_experiment_ids
                ],
            }
            exit_code = 1 if result.failed_experiment_ids else 0
        else:
            result = service.update_experiment(
                args.experiment_id,
                requested_cutoff=args.requested_cutoff,
                as_of=args.as_of,
                code_version=args.code_version,
                calculation_version=args.calculation_version,
            )
            payload = _result_payload(result)
            exit_code = 0
        print(json.dumps(payload, sort_keys=True))
        return exit_code
    finally:
        engine.dispose()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
