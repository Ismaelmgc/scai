"""SCAI CLI — entry point for daily pipeline + web dashboard.

Usage:
    scai run          # Run daily pipeline (both strategies)
    scai run --dry-run
    scai run --force-retrain
    scai web          # Start web dashboard
    scai web --port 8080
    scai monitor      # Run intraday trailing stop check
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]  # SCAI/


def _env() -> dict[str, str]:
    """Build env dict with DYLD and PYTHONPATH."""
    env = os.environ.copy()
    env["DYLD_LIBRARY_PATH"] = str(ROOT / ".local" / "lib")
    env["PYTHONPATH"] = str(ROOT / "src")
    # Windows console defaults to cp1252 and crashes on the Unicode glyphs
    # (▸ ✓ → …) the pipeline prints. Force UTF-8 stdout/stderr. Harmless on macOS.
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def cmd_run(args: argparse.Namespace) -> None:
    """Run the daily pipeline (both baseline + adaptive strategies)."""
    script = str(ROOT / "scripts" / "daily_pipeline.py")
    cmd = [sys.executable, script]
    if args.force_retrain:
        cmd.append("--force-retrain")
    if args.dry_run:
        cmd.append("--dry-run")
    if args.skip_download:
        cmd.append("--skip-download")
    if args.skip_features:
        cmd.append("--skip-features")
    if args.capital != 1000.0:
        cmd.extend(["--capital", str(args.capital)])

    result = subprocess.run(cmd, env=_env(), cwd=str(ROOT))
    sys.exit(result.returncode)


def cmd_web(args: argparse.Namespace) -> None:
    """Start the web dashboard."""
    import uvicorn
    os.environ["DYLD_LIBRARY_PATH"] = str(ROOT / ".local" / "lib")
    sys.path.insert(0, str(ROOT / "src"))
    uvicorn.run(
        "app.web.server:app",
        host="0.0.0.0",
        port=args.port,
        reload=False,
    )


def cmd_monitor(args: argparse.Namespace) -> None:
    """Run intraday trailing stop monitor."""
    script = str(ROOT / "scripts" / "intraday_monitor.py")
    cmd = [sys.executable, script]
    if args.force:
        cmd.append("--force")
    result = subprocess.run(cmd, env=_env(), cwd=str(ROOT))
    sys.exit(result.returncode)


def app() -> None:
    parser = argparse.ArgumentParser(
        prog="scai",
        description="SCAI — Small Cap AI Trading Platform",
    )
    sub = parser.add_subparsers(dest="command")

    # scai run
    p_run = sub.add_parser("run", help="Run daily pipeline")
    p_run.add_argument("--force-retrain", action="store_true")
    p_run.add_argument("--dry-run", action="store_true")
    p_run.add_argument("--skip-download", action="store_true")
    p_run.add_argument("--skip-features", action="store_true")
    p_run.add_argument("--capital", type=float, default=1000.0)

    # scai web
    p_web = sub.add_parser("web", help="Start web dashboard")
    p_web.add_argument("--port", type=int, default=8501)

    # scai monitor
    p_mon = sub.add_parser("monitor", help="Intraday trailing stop check")
    p_mon.add_argument("--force", action="store_true")

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "web":
        cmd_web(args)
    elif args.command == "monitor":
        cmd_monitor(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    app()
