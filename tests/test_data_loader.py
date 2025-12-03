"""
Tests for multi-source data loader utilities.
"""

from pathlib import Path

import pandas as pd
import pytest
from sqlalchemy import create_engine

from src.data_loader import (
    load_csv_from_path,
    load_sql_query,
    load_database_table,
    ensure_sample_sqlite,
    DataLoaderError,
)

SAMPLE_CSV = Path("sample_data/marketing_campaign_performance_sample.csv")


@pytest.fixture(scope="module")
def sample_csv_exists():
    assert SAMPLE_CSV.exists(), "Sample CSV missing"
    return SAMPLE_CSV


@pytest.fixture()
def temp_sqlite(tmp_path, sample_csv_exists):
    db_path = tmp_path / "test_marketing.db"
    conn_str = ensure_sample_sqlite(sample_csv_exists, db_path)
    return conn_str


def test_load_csv_from_path(sample_csv_exists):
    df = load_csv_from_path(str(sample_csv_exists))
    assert not df.empty
    assert set(["Date", "Campaign"]).issubset(df.columns)


def test_load_sql_query(temp_sqlite):
    df = load_sql_query(temp_sqlite, "SELECT * FROM marketing_data LIMIT 50")
    assert len(df) <= 50
    assert "Impressions" in df.columns


def test_load_database_table(temp_sqlite):
    df = load_database_table(temp_sqlite, "marketing_data", limit=25)
    assert len(df) <= 25
    assert "Clicks" in df.columns


def test_load_sql_query_invalid_connection():
    with pytest.raises(DataLoaderError):
        load_sql_query("sqlite:///non_existent.db", "SELECT 1")

