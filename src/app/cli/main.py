"""SCAI CLI — local entry point for the small-cap daily pipeline + web dashboard.

Usage:
    scai run          # small-cap daily pipeline (data, features, retrain, signals)
    scai run --dry-run
    scai run --force-retrain
    scai web          # local web dashboard (http://localhost:8501)
    scai web --port 8080

The production books (liquidcap, btc_trend, bounce, illiquid) run in GitHub
Actions; see .github/workflows/.
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
    """Run the small-cap daily pipeline."""
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


def app() -> None:
    parser = argparse.ArgumentParser(
        prog="scai",
        description="SCAI — algorithmic trading research platform",
    )
    sub = parser.add_subparsers(dest="command")

    # scai run
    p_run = sub.add_parser("run", help="Run daily pipeline")
    p_run.add_argument("--force-retrain", action="store_true")
    p_run.add_argument("--dry-run", action="store_true")
    p_run.add_argument("--skip-download", action="store_true")
    p_run.add_argument("--skip-features", action="store_true")

    # scai web
    p_web = sub.add_parser("web", help="Start web dashboard")
    p_web.add_argument("--port", type=int, default=8501)

    args = parser.parse_args()

    if args.command == "run":
        cmd_run(args)
    elif args.command == "web":
        cmd_web(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    app()
