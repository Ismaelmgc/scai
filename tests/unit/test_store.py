"""Tests for the parquet store."""

import pandas as pd

from app.data.store.parquet_store import ParquetStore


def test_write_and_read(tmp_path):
    store = ParquetStore(base_dir=tmp_path)
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    store.write("test_domain", df)
    result = store.read("test_domain")
    assert len(result) == 3
    assert list(result.columns) == ["a", "b"]


def test_upsert(tmp_path):
    store = ParquetStore(base_dir=tmp_path)
    df1 = pd.DataFrame({"id": [1, 2], "value": [10, 20]})
    store.upsert("test_upsert", df1, key_cols=["id"])

    df2 = pd.DataFrame({"id": [2, 3], "value": [25, 30]})
    store.upsert("test_upsert", df2, key_cols=["id"])

    result = store.read("test_upsert")
    assert len(result) == 3  # id=1 from first, id=2 replaced, id=3 new
    assert result[result["id"] == 2]["value"].iloc[0] == 25


def test_read_nonexistent(tmp_path):
    store = ParquetStore(base_dir=tmp_path)
    result = store.read("nonexistent")
    assert result.empty


def test_read_as_of(tmp_path):
    store = ParquetStore(base_dir=tmp_path)
    df = pd.DataFrame({
        "date": pd.to_datetime(["2023-01-01", "2023-06-01", "2024-01-01"]),
        "value": [1, 2, 3],
    })
    store.write("test_as_of", df)
    result = store.read_as_of("test_as_of", "2023-07-01")
    assert len(result) == 2
