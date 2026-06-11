"""Flat Files support for bulk historical data.

Massive/Polygon Flat Files provide bulk historical data via S3-compatible download.
These are more efficient than REST for large historical backfills.

Datasets:
  - us_stocks_sip/day_aggs_v1 (daily OHLCV, unadjusted)
  - us_stocks_sip/minute_aggs_v1 (minute bars, unadjusted)
  - us_stocks_sip/trades_v1 (tick-level trades)
  - us_stocks_sip/quotes_v1 (tick-level quotes)

IMPORTANT: Flat Files are UNADJUSTED. Apply corporate actions before using in backtests.
"""

from __future__ import annotations

import gzip
import io
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Iterator

import pandas as pd

from app.utils import get_logger

log = get_logger(__name__)

# Flat file base URL pattern
_FF_BASE_URL = "https://files.polygon.io/flat-files"

# Dataset paths
DATASETS = {
    "day_aggs": "us_stocks_sip/day_aggs_v1",
    "minute_aggs": "us_stocks_sip/minute_aggs_v1",
    "trades": "us_stocks_sip/trades_v1",
    "quotes": "us_stocks_sip/quotes_v1",
}


class FlatFilesAPI:
    """Manager for Massive Flat Files download, decompression, and conversion.

    Flat Files are available as .csv.gz partitioned by date.
    This module handles download, dedup, and conversion to parquet.
    """

    def __init__(
        self,
        data_dir: Path | str = "data/flat_files",
        api_key: str | None = None,
    ) -> None:
        self._data_dir = Path(data_dir)
        self._data_dir.mkdir(parents=True, exist_ok=True)

        self._key = api_key or os.environ.get("MASSIVE_API_KEY") or os.environ.get("SCAI_POLYGON_API_KEY", "")
        if not self._key:
            log.warning("flat_files_no_key", msg="No API key configured for flat files")

    def _get_file_url(self, dataset: str, file_date: date) -> str:
        """Construct the flat file URL for a given dataset and date."""
        ds_path = DATASETS.get(dataset, dataset)
        date_str = file_date.strftime("%Y-%m-%d")
        return f"{_FF_BASE_URL}/{ds_path}/{date_str}.csv.gz"

    def _local_csv_path(self, dataset: str, file_date: date) -> Path:
        """Local path for downloaded compressed file."""
        return self._data_dir / dataset / f"{file_date.isoformat()}.csv.gz"

    def _local_parquet_path(self, dataset: str, file_date: date) -> Path:
        """Local path for converted parquet file."""
        return self._data_dir / dataset / f"{file_date.isoformat()}.parquet"

    def is_downloaded(self, dataset: str, file_date: date) -> bool:
        """Check if a flat file has already been downloaded and converted."""
        return self._local_parquet_path(dataset, file_date).exists()

    def download_file(
        self,
        dataset: str,
        file_date: date,
        *,
        force: bool = False,
    ) -> Path | None:
        """Download a single flat file and convert to parquet.

        Returns path to parquet file, or None if download failed.
        Skips if already exists and force=False.
        """
        import httpx

        parquet_path = self._local_parquet_path(dataset, file_date)
        if parquet_path.exists() and not force:
            log.info("flat_file_exists", dataset=dataset, date=file_date.isoformat())
            return parquet_path

        url = self._get_file_url(dataset, file_date)
        csv_path = self._local_csv_path(dataset, file_date)
        csv_path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with httpx.Client(timeout=120) as client:
                resp = client.get(url, params={"apiKey": self._key})
                if resp.status_code == 404:
                    log.info("flat_file_not_found", dataset=dataset, date=file_date.isoformat())
                    return None
                if resp.status_code == 403:
                    log.warning("flat_file_forbidden", dataset=dataset, msg="Plan may not include flat files")
                    return None
                resp.raise_for_status()

            # Write compressed file
            csv_path.write_bytes(resp.content)

            # Decompress and convert to parquet
            df = self._decompress_to_df(csv_path)
            if df is not None and not df.empty:
                parquet_path.parent.mkdir(parents=True, exist_ok=True)
                df.to_parquet(parquet_path, index=False, engine="pyarrow")
                log.info("flat_file_converted", dataset=dataset, date=file_date.isoformat(), rows=len(df))
                # Remove compressed file to save space
                csv_path.unlink(missing_ok=True)
                return parquet_path

        except Exception as e:
            log.error("flat_file_download_failed", dataset=dataset, date=file_date.isoformat(), error=str(e))

        return None

    def _decompress_to_df(self, csv_gz_path: Path) -> pd.DataFrame | None:
        """Decompress .csv.gz and parse into DataFrame."""
        try:
            with gzip.open(csv_gz_path, "rt") as f:
                df = pd.read_csv(f)
            return df
        except Exception as e:
            log.error("decompress_failed", path=str(csv_gz_path), error=str(e))
            return None

    def download_range(
        self,
        dataset: str,
        start_date: date,
        end_date: date,
        *,
        force: bool = False,
    ) -> list[Path]:
        """Download flat files for a date range. Skips weekends and existing files."""
        paths = []
        current = start_date
        while current <= end_date:
            # Skip weekends (no market data)
            if current.weekday() < 5:
                path = self.download_file(dataset, current, force=force)
                if path:
                    paths.append(path)
            current += timedelta(days=1)
        log.info("download_range_complete", dataset=dataset, files=len(paths))
        return paths

    def read_date(self, dataset: str, file_date: date) -> pd.DataFrame | None:
        """Read a single date's parquet file."""
        path = self._local_parquet_path(dataset, file_date)
        if not path.exists():
            return None
        return pd.read_parquet(path)

    def read_range(
        self,
        dataset: str,
        start_date: date,
        end_date: date,
    ) -> pd.DataFrame:
        """Read and concatenate parquet files for a date range."""
        frames = []
        current = start_date
        while current <= end_date:
            if current.weekday() < 5:
                df = self.read_date(dataset, current)
                if df is not None:
                    frames.append(df)
            current += timedelta(days=1)

        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)

    def list_available_dates(self, dataset: str) -> list[date]:
        """List dates for which we have local parquet files."""
        ds_dir = self._data_dir / dataset
        if not ds_dir.exists():
            return []
        dates = []
        for f in sorted(ds_dir.glob("*.parquet")):
            try:
                dates.append(date.fromisoformat(f.stem))
            except ValueError:
                continue
        return dates

    def get_manifest(self) -> dict[str, dict]:
        """Get a manifest of all downloaded flat files."""
        manifest = {}
        for ds_name in DATASETS:
            dates = self.list_available_dates(ds_name)
            manifest[ds_name] = {
                "count": len(dates),
                "first": dates[0].isoformat() if dates else None,
                "last": dates[-1].isoformat() if dates else None,
            }
        return manifest
