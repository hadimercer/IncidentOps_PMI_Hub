from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import pandas as pd
import psycopg2
import pytest
from psycopg2.extras import execute_values

from pipeline import load_csv


REPO_ROOT = Path(__file__).resolve().parents[1]
SQL_FILES = [REPO_ROOT / "scripts/01_create_schema.sql", REPO_ROOT / "scripts/03_seed_reference.sql", REPO_ROOT / "scripts/02_views.sql"]
RAW_INSERT_SQL = """
INSERT INTO im.event_raw (
    event_key, case_id, variant, priority, reporter, ts, event, issue_type,
    resolver, report_channel, short_description, customer_satisfaction
)
VALUES %s
ON CONFLICT (event_key) DO NOTHING
"""


def apply_repo_sql(conn: psycopg2.extensions.connection) -> None:
    with conn.cursor() as cur:
        for sql_file in SQL_FILES:
            cur.execute(sql_file.read_text(encoding="utf-8"))
    conn.commit()


def _insert_events(conn: psycopg2.extensions.connection, rows: Iterable[dict[str, object]]) -> None:
    frame = pd.DataFrame(list(rows), columns=load_csv.TARGET_COLUMNS)
    frame["ts"] = pd.to_datetime(frame["ts"], utc=True)
    frame["event_key"] = frame.apply(load_csv.build_event_key, axis=1)
    ordered = frame[load_csv.DB_COLUMNS]
    records = load_csv.dataframe_to_records(ordered)
    with conn.cursor() as cur:
        execute_values(cur, RAW_INSERT_SQL, records)
    conn.commit()


@pytest.fixture(scope="session")
def test_db_url() -> str:
    database_url = os.getenv("TEST_DATABASE_URL")
    if not database_url:
        pytest.skip("TEST_DATABASE_URL is not set; skipping Postgres integration tests.")
    return database_url


@pytest.fixture()
def db_conn(test_db_url: str):
    with psycopg2.connect(test_db_url) as conn:
        with conn.cursor() as cur:
            cur.execute("DROP SCHEMA IF EXISTS im CASCADE")
        conn.commit()
        apply_repo_sql(conn)
        try:
            yield conn
        finally:
            with conn.cursor() as cur:
                cur.execute("DROP SCHEMA IF EXISTS im CASCADE")
            conn.commit()


@pytest.fixture()
def insert_events():
    return _insert_events

