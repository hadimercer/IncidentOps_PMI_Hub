from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from pipeline import load_csv


FIXTURE_CSV = Path(__file__).resolve().parent / "fixtures/incident_loader_fixture.csv"


def test_resolve_column_mapping_supports_aliases() -> None:
    mapping = load_csv.resolve_column_mapping(
        [
            "incident_number",
            "process_variant",
            "severity",
            "reported_by",
            "timestamp",
            "event_name",
            "category",
            "assigned_to",
            "channel",
            "summary",
            "csat",
        ]
    )

    assert mapping == {
        "case_id": "incident_number",
        "variant": "process_variant",
        "priority": "severity",
        "reporter": "reported_by",
        "ts": "timestamp",
        "event": "event_name",
        "issue_type": "category",
        "resolver": "assigned_to",
        "report_channel": "channel",
        "short_description": "summary",
        "customer_satisfaction": "csat",
    }


def test_standardize_allows_missing_csat() -> None:
    chunk = pd.DataFrame(
        {
            "case_id": ["INC-1"],
            "variant": ["Happy path"],
            "priority": ["High"],
            "reporter": ["Ava"],
            "ts": ["2024-01-01 09:00:00"],
            "event": ["Ticket created"],
            "issue_type": ["Access"],
            "resolver": ["L1"],
            "report_channel": ["Portal"],
            "short_description": ["Password reset"],
            "customer_satisfaction": [""],
        }
    )
    column_mapping = {column: column for column in load_csv.TARGET_COLUMNS}

    validated, dropped = load_csv.standardize_and_validate_chunk(chunk, column_mapping)

    assert dropped == 0
    assert len(validated) == 1
    assert validated.iloc[0]["customer_satisfaction"] != validated.iloc[0]["customer_satisfaction"]
    assert validated.iloc[0]["event_key"]


def test_standardize_drops_missing_required_rows() -> None:
    chunk = pd.DataFrame(
        {
            "case_id": ["INC-1", ""],
            "variant": ["Happy path", "Broken path"],
            "priority": ["High", "Low"],
            "reporter": ["Ava", "Ben"],
            "ts": ["2024-01-01 09:00:00", "2024-01-01 10:00:00"],
            "event": ["Ticket created", "Ticket closed"],
            "issue_type": ["Access", "Access"],
            "resolver": ["L1", "L1"],
            "report_channel": ["Portal", "Portal"],
            "short_description": ["Password reset", "Password reset"],
            "customer_satisfaction": ["", "4.0"],
        }
    )
    column_mapping = {column: column for column in load_csv.TARGET_COLUMNS}

    validated, dropped = load_csv.standardize_and_validate_chunk(chunk, column_mapping)

    assert dropped == 1
    assert len(validated) == 1
    assert validated.iloc[0]["case_id"] == "INC-1"


def test_run_reports_duplicate_rows_when_reloading_same_file(db_conn, test_db_url: str, monkeypatch, capsys) -> None:
    monkeypatch.setenv("DATABASE_URL_DIRECT", test_db_url)
    args = argparse.Namespace(
        path=str(FIXTURE_CSV),
        truncate=False,
        delimiter=";",
        chunksize=100,
        dry_run=False,
    )

    assert load_csv.run(args) == 0
    first_output = capsys.readouterr().out

    with db_conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM im.event_raw")
        assert cur.fetchone()[0] == 4

    assert load_csv.run(args) == 0
    second_output = capsys.readouterr().out

    with db_conn.cursor() as cur:
        cur.execute("SELECT COUNT(*) FROM im.event_raw")
        assert cur.fetchone()[0] == 4

    assert "rows_duplicate_skipped: 1" in first_output
    assert "rows_inserted: 4" in first_output
    assert "rows_duplicate_skipped: 5" in second_output
    assert "rows_inserted: 0" in second_output

