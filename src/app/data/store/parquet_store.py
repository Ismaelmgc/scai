"""Parquet-backed local data store with DuckDB query layer.

Design decisions:
  - One parquet file per data domain (ohlcv, fundamentals, universe, …).
  - DuckDB is used *only* for ad-hoc analytical queries on top of parquet
    files – the parquet files are the source of truth.
  - Every write is append-or-replace keyed on (ticker, date).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from app.config import get_settings
from app.utils import ensure_dir, get_logger

log = get_logger(__name__)


class ParquetStore:
    """Thin wrapper around parquet files with DuckDB query support."""

    def __init__(self, base_dir: Path | None = None) -> None:
        self._base = base_dir or get_settings().processed_dir
        ensure_dir(self._base)
        self._con = duckdb.connect(database=":memory:")

    def _path(self, domain: str) -> Path:
        return self._base / f"{domain}.parquet"

    # ── Write ───────────────────────────────────────────────
    def upsert(self, domain: str, df: pd.DataFrame, key_cols: list[str] | None = None) -> None:
        """Append or replace rows in a parquet file.

        If the file exists and *key_cols* is given, existing rows matching
        the key are dropped before appending.
        """
        path = self._path(domain)
        if path.exists() and key_cols:
            existing = pd.read_parquet(path)
            # Remove rows whose keys are in the new data
            merge_key = df[key_cols].drop_duplicates()
            existing = existing.merge(merge_key, on=key_cols, how="left", indicator=True)
            existing = existing[existing["_merge"] == "left_only"].drop(columns=["_merge"])
            df = pd.concat([existing, df], ignore_index=True)
        elif path.exists():
            existing = pd.read_parquet(path)
            df = pd.concat([existing, df], ignore_index=True)

        table = pa.Table.from_pandas(df)
        pq.write_table(table, path, compression="snappy")
        log.info("parquet_upserted", domain=domain, rows=len(df))

    def write(self, domain: str, df: pd.DataFrame) -> None:
        """Overwrite parquet file for a domain."""
        path = self._path(domain)
        table = pa.Table.from_pandas(df)
        pq.write_table(table, path, compression="snappy")
        log.info("parquet_written", domain=domain, rows=len(df))

    # ── Read ────────────────────────────────────────────────
    def read(self, domain: str) -> pd.DataFrame:
        path = self._path(domain)
        if not path.exists():
            log.warning("parquet_not_found", domain=domain)
            return pd.DataFrame()
        return pd.read_parquet(path)

    def exists(self, domain: str) -> bool:
        return self._path(domain).exists()

    # ── DuckDB queries ──────────────────────────────────────
    def query(self, sql: str, params: dict[str, Any] | None = None) -> pd.DataFrame:
        """Run arbitrary SQL against registered parquet files.

        Use ``read_parquet('path')`` in SQL to reference files, or register
        domains first with :meth:`register`.
        """
        return self._con.execute(sql, (params or {}) if params else []).fetchdf()

    def register(self, domain: str, alias: str | None = None) -> None:
        """Register a parquet file as a DuckDB view."""
        path = self._path(domain)
        name = alias or domain
        if path.exists():
            self._con.execute(
                f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM read_parquet('{path}')"
            )

    # ── Point-in-time convenience ───────────────────────────
    def read_as_of(self, domain: str, as_of_date: str, date_col: str = "date") -> pd.DataFrame:
        """Return rows where date_col <= as_of_date."""
        df = self.read(domain)
        if df.empty:
            return df
        df[date_col] = pd.to_datetime(df[date_col])
        return df[df[date_col] <= pd.Timestamp(as_of_date)].copy()
