"""Apply SQL scripts to Postgres and verify core objects."""

from __future__ import annotations

from pathlib import Path

import psycopg2
from dotenv import load_dotenv

import os


SQL_FILES = [
    "scripts/01_create_schema.sql",
    "scripts/03_seed_reference.sql",
    "scripts/02_views.sql",
]

OBJECT_CHECKS = [
    ("im.event_raw", "r"),      # ordinary table
    ("im.sla_policy", "r"),     # ordinary table
    ("im.event_catalog", "r"),  # ordinary table
    ("im.v_event_seq", "v"),    # view
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
            except Exception as exc:  # psycopg2 emits several subclasses
                conn.rollback()
                print(f"FAIL apply {rel_path}: {exc}")
                success = False
                break

    return success


def _verify_objects(conn: psycopg2.extensions.connection) -> bool:
    success = True

    with conn.cursor() as cur:
        for object_name, expected_kind in OBJECT_CHECKS:
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
            exists = bool(cur.fetchone()[0])
            status = "OK" if exists else "FAIL"
            print(f"{status} exists {object_name}")
            if not exists:
                success = False

        for table_name in ("im.sla_policy", "im.event_catalog"):
            schema_name, rel_name = table_name.split(".", 1)
            cur.execute(
                """
                SELECT EXISTS (
                    SELECT 1
                    FROM pg_catalog.pg_class c
                    JOIN pg_catalog.pg_namespace n
                      ON n.oid = c.relnamespace
                    WHERE n.nspname = %s
                      AND c.relname = %s
                      AND c.relkind = 'r'
                )
                """,
                (schema_name, rel_name),
            )
            table_exists = bool(cur.fetchone()[0])
            if not table_exists:
                print(f"FAIL row_count {table_name}: table missing")
                success = False
                continue

            cur.execute(f"SELECT COUNT(*) FROM {table_name}")
            row_count = int(cur.fetchone()[0])
            print(f"OK row_count {table_name}={row_count}")

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
