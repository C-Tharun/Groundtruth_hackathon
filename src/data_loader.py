"""
Utilities for loading marketing data from multiple sources (CSV, SQL, databases).
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd
from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine


class DataLoaderError(Exception):
    """Custom exception for data loading failures."""


def _ensure_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        raise DataLoaderError("No data returned from source.")
    return df


def load_csv_from_buffer(buffer) -> pd.DataFrame:
    """Load CSV data from an uploaded file buffer."""
    try:
        return _ensure_dataframe(pd.read_csv(buffer))
    except Exception as exc:
        raise DataLoaderError(f"Failed to read CSV: {exc}") from exc


def load_csv_from_path(path: str) -> pd.DataFrame:
    """Load CSV data from disk."""
    try:
        return _ensure_dataframe(pd.read_csv(path))
    except FileNotFoundError as exc:
        raise DataLoaderError(f"CSV file not found: {path}") from exc
    except Exception as exc:
        raise DataLoaderError(f"Failed to read CSV {path}: {exc}") from exc


def _create_engine(connection_string: str) -> Engine:
    if not connection_string:
        raise DataLoaderError("Connection string is required.")
    try:
        return create_engine(connection_string)
    except Exception as exc:
        raise DataLoaderError(f"Invalid connection string: {exc}") from exc


def load_sql_query(connection_string: str, query: str) -> pd.DataFrame:
    """Execute an arbitrary SQL query and return the result."""
    if not query or not query.strip():
        raise DataLoaderError("SQL query is required.")
    engine = _create_engine(connection_string)
    try:
        with engine.connect() as conn:
            df = pd.read_sql(text(query), conn)
        return _ensure_dataframe(df)
    except Exception as exc:
        raise DataLoaderError(f"SQL query failed: {exc}") from exc


def load_database_table(connection_string: str, table: str, limit: Optional[int] = None) -> pd.DataFrame:
    """Load all rows (or a limited subset) from a database table."""
    if not table:
        raise DataLoaderError("Table name is required.")
    engine = _create_engine(connection_string)
    try:
        with engine.connect() as conn:
            if limit and limit > 0:
                query = text(f"SELECT * FROM {table} LIMIT :limit")
                df = pd.read_sql(query, conn, params={"limit": limit})
            else:
                df = pd.read_sql_table(table, conn)
        return _ensure_dataframe(df)
    except Exception as exc:
        raise DataLoaderError(f"Failed to load table '{table}': {exc}") from exc


def ensure_sample_sqlite(csv_path: Path, sqlite_path: Path, table_name: str = "marketing_data") -> str:
    """
    Create a SQLite database from the sample CSV if it doesn't exist.
    
    Returns SQLAlchemy connection string.
    """
    sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    connection_string = f"sqlite:///{sqlite_path}"
    if not sqlite_path.exists():
        df = load_csv_from_path(str(csv_path))
        engine = _create_engine(connection_string)
        with engine.begin() as conn:
            df.to_sql(table_name, conn, index=False, if_exists='replace')
    return connection_string

