"""SCAI Web Dashboard — serves the login shell + pipeline-trigger API.

The dashboard itself is client-rendered: the page ships with NO trade data, logs
in via Supabase Auth, then fetches the render-ready `dashboard_view` (RLS allows
only authenticated reads) and paints KPIs/chart/tables in the browser. Both the
live server and the static Pages snapshot serve the same shell.
"""
from __future__ import annotations

import base64
import os
import subprocess
import sys
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from app.data import supabase_store  # noqa: E402
from app.data.free_sources import finnhub  # noqa: E402

app = FastAPI(title="SCAI Dashboard")
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))

PAPER_TRADING_DIR = ROOT / "data" / "paper_trading"
_STATIC = Path(__file__).parent / "static"
LOGO_PATH = _STATIC / "scai_bull.png"        # header mark: thin gold line-art bull
FAVICON_PATH = _STATIC / "scai_favicon.png"  # browser tab: gold tile (legible at 16px)

_pipeline_proc: subprocess.Popen | None = None


def _asset_data_uri(path: Path) -> str:
    """Base64 PNG data URI for a brand asset, embedded inline so the static
    Pages snapshot stays a single self-contained file. "" if the asset is missing.
    """
    if not path.exists():
        return ""
    return f"data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}"


def shell_context(static_mode: bool) -> dict:
    """Template context for the login shell (no trade data ever ships here)."""
    supabase_url, supabase_anon_key = supabase_store.public_config()
    return {
        "static_mode": static_mode,
        "supabase_url": supabase_url,
        "supabase_anon_key": supabase_anon_key,
        # Live prices stream client-side via the Finnhub WebSocket (post-login).
        "finnhub_token": finnhub.public_token(),
        "logo": _asset_data_uri(LOGO_PATH),
        "favicon": _asset_data_uri(FAVICON_PATH),
    }


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse(
        request=request, name="dashboard.html",
        context=shell_context(static_mode=False),
    )


@app.post("/api/run-pipeline", response_class=JSONResponse)
async def run_pipeline():
    global _pipeline_proc

    if _pipeline_proc is not None and _pipeline_proc.poll() is None:
        return JSONResponse(
            {"status": "running", "message": "Pipeline ya en ejecución"},
            status_code=409,
        )

    script = str(ROOT / "scripts" / "daily_pipeline.py")
    python = str(ROOT / ".venv" / "bin" / "python")
    log_path = PAPER_TRADING_DIR / "logs"
    log_path.mkdir(parents=True, exist_ok=True)

    env = {
        "PYTHONPATH": str(ROOT / "src"),
        "DYLD_LIBRARY_PATH": str(ROOT / ".local" / "lib"),
        "PATH": f"{ROOT / '.venv' / 'bin'}:/usr/local/bin:/usr/bin:/bin",
        "HOME": os.environ.get("HOME", ""),
    }
    # Load .env file for API keys
    env_file = ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, val = line.partition("=")
                env[key.strip()] = val.strip()

    # Kept open intentionally: handed to Popen for the subprocess lifetime.
    stdout_log = open(log_path / "web_pipeline_stdout.log", "w")  # noqa: SIM115
    stderr_log = open(log_path / "web_pipeline_stderr.log", "w")  # noqa: SIM115

    _pipeline_proc = subprocess.Popen(
        [python, script],
        cwd=str(ROOT),
        env=env,
        stdout=stdout_log,
        stderr=stderr_log,
    )

    return {"status": "started", "pid": _pipeline_proc.pid}


@app.get("/api/pipeline-status", response_class=JSONResponse)
async def pipeline_status():
    global _pipeline_proc

    if _pipeline_proc is None:
        return {"status": "idle"}

    rc = _pipeline_proc.poll()
    if rc is None:
        # Read log tail for progress
        log_file = PAPER_TRADING_DIR / "logs" / "web_pipeline_stdout.log"
        tail = ""
        if log_file.exists():
            lines = log_file.read_text().strip().splitlines()
            tail = "\n".join(lines[-10:])
        return {"status": "running", "pid": _pipeline_proc.pid, "log_tail": tail}

    log_file = PAPER_TRADING_DIR / "logs" / "web_pipeline_stdout.log"
    tail = ""
    if log_file.exists():
        lines = log_file.read_text().strip().splitlines()
        tail = "\n".join(lines[-25:])

    _pipeline_proc = None
    return {"status": "finished", "exit_code": rc, "log_tail": tail}
