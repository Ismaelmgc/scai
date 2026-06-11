"""Reset paper trading to start with V3 (May 19, 2026 = "yesterday").

- Archives current V2 paper-trading state to data/paper_trading/archive_v2/
- Creates fresh portfolio.json with initial_capital=1000, no positions
- Clears live signal_history.parquet, daily_log.jsonl
- Keeps signal_history_backtest.parquet (used for meta-learning features)
- Keeps model_registry.json (already updated by V3 training)
"""
from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path

PT = Path("data/paper_trading")
ARCHIVE = PT / "archive_v2"


def main() -> None:
    ARCHIVE.mkdir(parents=True, exist_ok=True)
    moved = []
    for fname in [
        "portfolio.json",
        "signal_history.parquet",
        "daily_log.jsonl",
        "trades.parquet",
        "signals_2026-05-19.parquet",
        "signals_2026-05-20.parquet",
    ]:
        src = PT / fname
        if src.exists():
            shutil.move(str(src), str(ARCHIVE / fname))
            moved.append(fname)
    print(f"Archived: {moved}")

    fresh = {
        "initial_capital": 1000.0,
        "cash": 1000.0,
        "positions": [],
        "pending_signals": [],
        "closed_trades": [],
        "version": "v3",
        "created_at": date.today().isoformat(),
    }
    (PT / "portfolio.json").write_text(json.dumps(fresh, indent=2))
    print(f"Fresh portfolio: capital=1000, no positions, version=v3")
    print(f"Listo en {PT / 'portfolio.json'}")


if __name__ == "__main__":
    main()
