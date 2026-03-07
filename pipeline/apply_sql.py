"""Apply SQL scripts to Postgres and verify core objects."""

from __future__ import annotations

import os
from pathlib import Path

import psycopg2
from dotenv import load_dotenv


SQL_FILES = [
    "scripts/01_create_schema.sql",
    "scripts/03_seed_reference.sql",
    "scripts/02_views.sql",
]

OBJECT_CHECKS = [
    ("im.event_raw", "r"),
    ("im.event_stage", "r"),
    ("im.sla_policy", "r"),
    ("im.event_catalog", "r"),
    ("im.event_alias", "r"),
    ("im.v_event_normalized", "v"),
    ("im.v_event_seq", "v"),
    ("im.v_case", "v"),
    ("im.v_case_sla", "v"),
    ("im.v_variant_summary", "v"),
    ("im.v_transition_summary", "v"),
    ("im.v_transition_by_variant", "v"),
    ("im.v_dwell_by_event", "v"),
    ("im.v_pingpong_cases", "v"),
    ("im.v_handoff_summary", "v"),
    ("im.v_worst_handoff_cases", "v"),
    ("im.v_closure_compliance", "v"),
    ("im.v_cx_summary", "v"),
    ("im.v_cx_breakdown", "v"),
    ("im.v_problem_candidate_cases", "v"),
    ("im.v_problem_candidates", "v"),
    ("im.v_problem_candidate_top_cases", "v"),
    ("im.v_channel_summary", "v"),
    ("im.v_channel_issue_summary", "v"),
    ("im.v_resolution_level", "v"),
    ("im.v_fcr_cases", "v"),
    ("im.v_fcr_summary", "v"),
    ("im.v_kb_enablement_candidates", "v"),
]

SMOKE_QUERIES = [
    ("executive_kpi", "SELECT * FROM im.v_case_sla LIMIT 1"),
    ("variant_summary", "SELECT * FROM im.v_variant_summary LIMIT 1"),
    ("range_query", "SELECT min(created_at) AS min_created, max(closed_at) AS max_closed FROM im.v_case"),
    ("transition_summary", "SELECT * FROM im.v_transition_summary LIMIT 1"),
    ("transition_by_variant", "SELECT * FROM im.v_transition_by_variant LIMIT 1"),
    ("dwell_by_event", "SELECT * FROM im.v_dwell_by_event LIMIT 1"),
    ("handoff_summary", "SELECT * FROM im.v_handoff_summary LIMIT 1"),
    ("pingpong_cases", "SELECT * FROM im.v_pingpong_cases LIMIT 1"),
    ("worst_handoff_cases", "SELECT * FROM im.v_worst_handoff_cases LIMIT 1"),
    ("cx_summary", "SELECT * FROM im.v_cx_summary LIMIT 1"),
    ("cx_breakdown", "SELECT * FROM im.v_cx_breakdown LIMIT 1"),
    ("closure_compliance", "SELECT * FROM im.v_closure_compliance LIMIT 1"),
    ("problem_candidates", "SELECT * FROM im.v_problem_candidates LIMIT 1"),
    ("problem_candidate_cases", "SELECT * FROM im.v_problem_candidate_cases LIMIT 1"),
    ("problem_candidate_top_cases", "SELECT * FROM im.v_problem_candidate_top_cases LIMIT 1"),
    ("channel_summary", "SELECT * FROM im.v_channel_summary LIMIT 1"),
    ("channel_issue_summary", "SELECT * FROM im.v_channel_issue_summary LIMIT 1"),
    ("resolution_level", "SELECT * FROM im.v_resolution_level LIMIT 1"),
    ("fcr_cases", "SELECT * FROM im.v_fcr_cases LIMIT 1"),
    ("fcr_summary", "SELECT * FROM im.v_fcr_summary LIMIT 1"),
    ("kb_enablement", "SELECT * FROM im.v_kb_enablement_candidates LIMIT 1"),
]


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _require_database_url() -> str:
    load_dotenv(dotenv_path=_repo_root() / ".env")
    db_url = os.getenv("DATABASE_URL_DIRECT")
    if not db_url:
        print("FAIL missing required environment variable: DATABASE_URL_DIRECT")
        raise SystemExit(2)
    return db_url


def _apply_sql_files(conn: psycopg2.extensions.connection) -> bool:
    base = _repo_root()
    success = True

    with conn.cursor() as cur:
        for rel_path in SQL_FILES:
            file_path = base / rel_path
            try:
                sql_text = file_path.read_text(encoding="utf-8")
            except OSError as exc:
                print(f"FAIL read {rel_path}: {exc}")
                success = False
                break

            try:
                cur.execute(sql_text)
                conn.commit()
                print(f"OK applied {rel_path}")
            except Exception as exc:
                conn.rollback()
                print(f"FAIL apply {rel_path}: {exc}")
                success = False
                break

    return success


def _relation_exists(cur: psycopg2.extensions.cursor, object_name: str, expected_kind: str) -> bool:
    schema_name, rel_name = object_name.split(".", 1)
    cur.execute(
        """
        SELECT EXISTS (
            SELECT 1
            FROM pg_catalog.pg_class c
            JOIN pg_catalog.pg_namespace n
              ON n.oid = c.relnamespace
            WHERE n.nspname = %s
              AND c.relname = %s
              AND c.relkind = %s
        )
        """,
        (schema_name, rel_name, expected_kind),
    )
    return bool(cur.fetchone()[0])


def _verify_objects(conn: psycopg2.extensions.connection) -> bool:
    success = True

    with conn.cursor() as cur:
        for object_name, expected_kind in OBJECT_CHECKS:
            exists = _relation_exists(cur, object_name, expected_kind)
            status = "OK" if exists else "FAIL"
            print(f"{status} exists {object_name}")
            if not exists:
                success = False

        for table_name in ("im.sla_policy", "im.event_catalog", "im.event_alias"):
            table_exists = _relation_exists(cur, table_name, "r")
            if not table_exists:
                print(f"FAIL row_count {table_name}: table missing")
                success = False
                continue

            cur.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = int(cur.fetchone()[0])
            print(f"OK row_count {table_name}={row_count}")
            if row_count == 0:
                print(f"FAIL row_count {table_name}: expected seeded rows")
                success = False

        try:
            cur.execute(
                "SELECT COUNT(*) FROM im.v_event_normalized WHERE event_code IS NULL"
            )
            unmapped_count = int(cur.fetchone()[0])
            print(f"OK unmapped_event_count={unmapped_count}")
            if unmapped_count > 0:
                print("FAIL unmapped raw events detected in im.v_event_normalized")
                success = False
        except Exception as exc:
            print(f"FAIL unmapped event query: {exc}")
            success = False

        for smoke_name, sql_text in SMOKE_QUERIES:
            try:
                cur.execute(sql_text)
                cur.fetchone()
                print(f"OK smoke_query {smoke_name}")
            except Exception as exc:
                print(f"FAIL smoke_query {smoke_name}: {exc}")
                success = False

    return success


def main() -> int:
    db_url = _require_database_url()

    try:
        with psycopg2.connect(db_url) as conn:
            applied = _apply_sql_files(conn)
            verified = _verify_objects(conn) if applied else False
            return 0 if (applied and verified) else 1
    except Exception as exc:
        print(f"FAIL database connection or execution error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
