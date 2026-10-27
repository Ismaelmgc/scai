"""Centralised application settings backed by env vars and .env file.

Uses pydantic-settings so every value can be overridden via environment
variables (uppercase, prefixed SCAI_), e.g. SCAI_POLYGON_API_KEY, SCAI_SEED.
Only settings the code actually reads live here; per-book constants (costs,
hold periods, thresholds) are frozen in each book's script next to the rule
they belong to.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Root configuration loaded once at startup."""

    model_config = SettingsConfigDict(
        env_prefix="SCAI_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    data_dir: Path = Path("./data")
    seed: int = 42

    # Polygon.io ("massive" client) — small-cap daily bars + reference data
    polygon_api_key: str = ""

    # Small-cap universe discovery (scripts/refresh_universe.py)
    min_market_cap: float = 50_000_000          # USD 50 M
    max_market_cap: float = 2_000_000_000       # USD 2 B
    exclude_otc: bool = True

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
