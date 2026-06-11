"""Reporting module – generate performance reports and visualisations."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from app.backtest import BacktestResult
from app.utils import ensure_dir, get_logger

log = get_logger(__name__)


def generate_text_report(result: BacktestResult) -> str:
    """Generate a plain-text summary report."""
    m = result.metrics
    lines = [
        "=" * 60,
        "SCAI BACKTEST REPORT",
        "=" * 60,
        "",
        "── Performance Summary ──",
        f"  Total Return:        {m.get('total_return', 0):.2%}",
        f"  CAGR:                {m.get('cagr', 0):.2%}",
        f"  Annual Volatility:   {m.get('annual_volatility', 0):.2%}",
        f"  Sharpe Ratio:        {m.get('sharpe_ratio', 0):.2f}",
        f"  Sortino Ratio:       {m.get('sortino_ratio', 0):.2f}",
        f"  Max Drawdown:        {m.get('max_drawdown', 0):.2%}",
        f"  Calmar Ratio:        {m.get('calmar_ratio', 0):.2f}",
        f"  # Trades:            {m.get('n_trades', 0)}",
        f"  Total Days:          {m.get('total_days', 0)}",
        "",
    ]

    # Year-by-year
    if not result.performance_by_year.empty:
        lines.append("── Performance by Year ──")
        lines.append(result.performance_by_year.to_string())
        lines.append("")

    lines.append("=" * 60)
    return "\n".join(lines)


def generate_html_report(result: BacktestResult, output_path: Path | str) -> Path:
    """Generate an HTML report with embedded charts (matplotlib)."""
    output_path = Path(output_path)
    ensure_dir(output_path.parent)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        # Fallback: write text report as HTML
        report_text = generate_text_report(result)
        html = f"<html><body><pre>{report_text}</pre></body></html>"
        output_path.write_text(html)
        return output_path

    fig, axes = plt.subplots(3, 1, figsize=(14, 12), gridspec_kw={"height_ratios": [3, 1, 1]})

    # 1. Portfolio value
    ax = axes[0]
    pv = result.portfolio_values
    ax.plot(pv.index, pv.values, linewidth=1.5, color="#2196F3")
    ax.set_title("Portfolio Value", fontsize=14)
    ax.set_ylabel("USD")
    ax.grid(True, alpha=0.3)

    # 2. Drawdown
    ax = axes[1]
    cum = (1 + result.daily_returns).cumprod()
    dd = cum / cum.cummax() - 1
    ax.fill_between(dd.index, dd.values, 0, color="#F44336", alpha=0.4)
    ax.set_title("Drawdown", fontsize=14)
    ax.set_ylabel("Drawdown")
    ax.grid(True, alpha=0.3)

    # 3. Rolling Sharpe (60d)
    ax = axes[2]
    rolling_sharpe = (
        result.daily_returns.rolling(60).mean()
        / result.daily_returns.rolling(60).std()
        * np.sqrt(252)
    )
    ax.plot(rolling_sharpe.index, rolling_sharpe.values, linewidth=1, color="#4CAF50")
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("Rolling 60-day Sharpe Ratio", fontsize=14)
    ax.set_ylabel("Sharpe")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    chart_path = output_path.parent / "backtest_chart.png"
    fig.savefig(chart_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Build HTML
    metrics_html = generate_text_report(result).replace("\n", "<br>")
    html = f"""<!DOCTYPE html>
<html>
<head><title>SCAI Backtest Report</title>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; margin: 40px; background: #fafafa; }}
  .metrics {{
    background: white;
    padding: 20px;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,.1);
  }}
  img {{ max-width: 100%; margin-top: 20px; }}
</style>
</head>
<body>
  <h1>SCAI Backtest Report</h1>
  <div class="metrics"><pre>{metrics_html}</pre></div>
  <img src="backtest_chart.png" alt="Backtest Chart">
</body>
</html>"""
    output_path.write_text(html)
    log.info("report_generated", path=str(output_path))
    return output_path
