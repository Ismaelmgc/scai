"""Centralised application settings backed by env vars and .env file.

Uses pydantic-settings so every value can be overridden via environment
variables (uppercase, prefixed SCAI_).
"""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppMode(StrEnum):
    DEMO = "demo"
    PRODUCTION = "production"


class MarketDataProvider(StrEnum):
    POLYGON = "polygon"
    TIINGO = "tiingo"
    ALPACA = "alpaca"
    ALPHA_VANTAGE = "alpha_vantage"


class Settings(BaseSettings):
    """Root configuration loaded once at startup."""

    model_config = SettingsConfigDict(
        env_prefix="SCAI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ── General ─────────────────────────────────────────────
    env: Literal["development", "production"] = "development"
    mode: AppMode = AppMode.DEMO
    data_dir: Path = Path("./data")
    log_level: str = "INFO"
    seed: int = 42

    # ── Market-data provider ────────────────────────────────
    market_data_provider: MarketDataProvider = MarketDataProvider.POLYGON
    polygon_api_key: str = ""
    tiingo_api_key: str = ""
    alpaca_api_key: str = ""
    alpaca_secret_key: str = ""
    alpha_vantage_api_key: str = ""

    # ── SEC EDGAR ───────────────────────────────────────────
    sec_edgar_user_agent: str = "scai-platform research@example.com"

    # ── MLflow ──────────────────────────────────────────────
    mlflow_tracking_uri: str = "./mlruns"

    # ── Universe defaults ───────────────────────────────────
    min_market_cap: float = 50_000_000          # USD 50 M
    max_market_cap: float = 2_000_000_000       # USD 2 B
    min_price: float = 1.0                      # USD
    min_adv_usd: float = 200_000                # min average daily dollar volume
    min_trading_days: int = 60
    exclude_otc: bool = True
    exclude_adrs: bool = True

    # ── Model defaults ──────────────────────────────────────
    horizons: list[int] = Field(default_factory=lambda: [1, 5, 10, 20])
    default_horizon: int = 5

    # ── Trading cost assumptions ────────────────────────────
    commission_bps: float = 5.0         # 5 bps per side
    slippage_bps: float = 10.0          # 10 bps estimated
    spread_bps: float = 15.0            # for small caps
    max_participation_rate: float = 0.05  # max 5 % of ADV

    # ── Backtest ────────────────────────────────────────────
    backtest_start: str = "2015-01-01"
    backtest_end: str = "2024-12-31"
    walk_forward_train_months: int = 36
    walk_forward_step_months: int = 3

    @property
    def raw_dir(self) -> Path:
        return self.data_dir / "raw"

    @property
    def interim_dir(self) -> Path:
        return self.data_dir / "interim"

    @property
    def processed_dir(self) -> Path:
        return self.data_dir / "processed"


# Singleton-ish accessor ─────────────────────────────────────
_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings()
    return _settings
