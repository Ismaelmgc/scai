"""CLI for Massive data layer.

Usage:
    python -m app.data.massive audit
    python -m app.data.massive backfill-reference
    python -m app.data.massive backfill-daily-bars \
        --start 2020-01-01 --end 2026-01-01 --adjusted true
    python -m app.data.massive backfill-corporate-actions --start 2010-01-01
    python -m app.data.massive update-daily
    python -m app.data.massive update-snapshots
    python -m app.data.massive validate
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date


def _parse_date(s: str) -> date:
    return date.fromisoformat(s)


def cmd_audit(args: argparse.Namespace) -> None:
    from app.data.massive.jobs import audit
    result = audit()
    print(json.dumps(result, indent=2, default=str))


def cmd_backfill_reference(args: argparse.Namespace) -> None:
    from app.data.massive.jobs import backfill_reference
    result = backfill_reference(
        start_date=args.start if hasattr(args, "start") and args.start else None,
        end_date=args.end if hasattr(args, "end") and args.end else None,
    )
    print(f"Backfill reference complete: {result}")


def cmd_backfill_daily_bars(args: argparse.Namespace) -> None:
    from app.data.massive.jobs import backfill_daily_bars
    result = backfill_daily_bars(
        start_date=args.start,
        end_date=args.end,
        adjusted=args.adjusted,
        use_grouped=args.grouped,
    )
    print(f"Backfill daily bars complete: {result}")


def cmd_backfill_corporate_actions(args: argparse.Namespace) -> None:
    from app.data.massive.jobs import backfill_corporate_actions
    result = backfill_corporate_actions(
        start_date=args.start,
        end_date=args.end if args.end else None,
    )
    print(f"Backfill corporate actions complete: {result}")


def cmd_update_daily(args: argparse.Namespace) -> None:
    from app.data.massive.jobs import update_daily
    result = update_daily()
    print(f"Daily update complete: {result}")


def cmd_update_snapshots(args: argparse.Namespace) -> None:
    from app.data.massive.jobs import update_snapshots
    count = update_snapshots()
    print(f"Snapshots updated: {count} tickers")


def cmd_validate(args: argparse.Namespace) -> None:
    from app.data.massive.jobs import validate_data
    issues = validate_data()
    has_issues = any(v for v in issues.values())
    for check, problems in issues.items():
        status = "FAIL" if problems else "PASS"
        print(f"  [{status}] {check}")
        for p in problems:
            print(f"         - {p}")
    if has_issues:
        sys.exit(1)
    print("\nAll validations passed.")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m app.data.massive",
        description="Massive data layer CLI — backfill, update, validate",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # audit
    sub.add_parser("audit", help="Show current data inventory")

    # backfill-reference
    p = sub.add_parser("backfill-reference",
                       help="Backfill reference data (tickers, exchanges, etc.)")
    p.add_argument("--start", type=_parse_date, default=None)
    p.add_argument("--end", type=_parse_date, default=None)

    # backfill-daily-bars
    p = sub.add_parser("backfill-daily-bars", help="Backfill daily OHLCV bars")
    p.add_argument("--start", type=_parse_date, default=date(2020, 1, 1))
    p.add_argument("--end", type=_parse_date, default=None)
    p.add_argument("--adjusted", type=lambda x: x.lower() == "true", default=True)
    p.add_argument("--grouped", action="store_true", help="Use grouped daily (1 call per date)")

    # backfill-corporate-actions
    p = sub.add_parser("backfill-corporate-actions", help="Backfill splits and dividends")
    p.add_argument("--start", type=_parse_date, default=date(2010, 1, 1))
    p.add_argument("--end", type=_parse_date, default=None)

    # update-daily
    sub.add_parser("update-daily",
                   help="Incremental daily update (bars + snapshots + corporate actions)")

    # update-snapshots
    sub.add_parser("update-snapshots", help="Fetch current market snapshots")

    # validate
    sub.add_parser("validate", help="Run data quality validations")

    args = parser.parse_args()

    commands = {
        "audit": cmd_audit,
        "backfill-reference": cmd_backfill_reference,
        "backfill-daily-bars": cmd_backfill_daily_bars,
        "backfill-corporate-actions": cmd_backfill_corporate_actions,
        "update-daily": cmd_update_daily,
        "update-snapshots": cmd_update_snapshots,
        "validate": cmd_validate,
    }

    commands[args.command](args)


if __name__ == "__main__":
    main()
