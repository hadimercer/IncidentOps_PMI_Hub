"""Load Incident Management CSV data into im.event_raw."""

from __future__ import annotations

import argparse
import os
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import psycopg2
from dotenv import load_dotenv
from psycopg2.extras import execute_values


TARGET_COLUMNS: List[str] = [
    "case_id",
    "variant",
    "priority",
    "reporter",
    "ts",
    "event",
    "issue_type",
    "resolver",
    "report_channel",
    "short_description",
    "customer_satisfaction",
]

REQUIRED_COLUMNS: List[str] = [col for col in TARGET_COLUMNS if col != "resolver"]

TEXT_COLUMNS: List[str] = [
    "case_id",
    "variant",
    "priority",
    "reporter",
    "event",
    "issue_type",
    "resolver",
    "report_channel",
    "short_description",
]

COLUMN_ALIASES: Dict[str, List[str]] = {
    "case_id": ["caseid", "incident_id", "incident_number", "ticket_id", "ticket_number", "id"],
    "variant": ["process_variant", "flow_variant", "path", "trace_variant"],
    "priority": ["severity", "incident_priority", "urgency_level"],
    "reporter": ["reported_by", "caller", "requester", "opened_by", "submitted_by"],
    "ts": [
        "timestamp",
        "time",
        "event_time",
        "event_timestamp",
        "datetime",
        "date_time",
        "created_at",
        "reported_at",
        "occurred_at",
    ],
    "event": ["event_name", "activity", "action", "status", "stage", "step"],
    "issue_type": ["issue_category", "category", "incident_type", "type"],
    "resolver": ["assignee", "assigned_to", "resolved_by", "handler", "owner", "resolver_group"],
    "report_channel": ["channel", "source_channel", "intake_channel", "contact_channel", "reported_via"],
    "short_description": ["description", "summary", "title", "short_desc", "brief_description"],
    "customer_satisfaction": ["csat", "satisfaction", "customer_rating", "sat_score", "rating"],
}

INSERT_SQL = """
INSERT INTO im.event_raw (
    case_id, variant, priority, reporter, ts, event, issue_type, resolver,
    report_channel, short_description, customer_satisfaction
)
VALUES %s
"""


@dataclass
class LoadStats:
    total_input_rows: int = 0
    total_valid_rows: int = 0
    total_dropped_rows: int = 0
    case_ids: set = field(default_factory=set)
    event_counts: Counter = field(default_factory=Counter)
    min_ts: Optional[pd.Timestamp] = None
    max_ts: Optional[pd.Timestamp] = None

    def update(self, validated_df: pd.DataFrame, input_rows: int, dropped_rows: int) -> None:
        self.total_input_rows += input_rows
        self.total_valid_rows += len(validated_df)
        self.total_dropped_rows += dropped_rows

        if validated_df.empty:
            return

        self.case_ids.update(validated_df["case_id"].astype(str).tolist())
        self.event_counts.update(validated_df["event"].astype(str).tolist())

        chunk_min = validated_df["ts"].min()
        chunk_max = validated_df["ts"].max()
        if self.min_ts is None or chunk_min < self.min_ts:
            self.min_ts = chunk_min
        if self.max_ts is None or chunk_max > self.max_ts:
            self.max_ts = chunk_max

    def print_summary(self, dry_run: bool) -> None:
        total_label = "would_insert" if dry_run else "inserted"
        min_ts_str = self.min_ts.isoformat() if self.min_ts is not None else "n/a"
        max_ts_str = self.max_ts.isoformat() if self.max_ts is not None else "n/a"

        print(f"rows_{total_label}: {self.total_valid_rows}")
        print(f"rows_dropped_invalid: {self.total_dropped_rows}")
        print(f"distinct_case_id_count: {len(self.case_ids)}")
        print(f"min_ts: {min_ts_str}")
        print(f"max_ts: {max_ts_str}")
        print(f"distinct_event_count: {len(self.event_counts)}")
        print("top_5_events:")
        if not self.event_counts:
            print("  (none)")
            return

        for event_name, event_count in self.event_counts.most_common(5):
            print(f"  {event_name}: {event_count}")


def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def normalize_column_name(name: str) -> str:
    normalized = name.strip().lower()
    normalized = normalized.replace(" ", "_")
    normalized = re.sub(r"[^a-z0-9_]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized


def resolve_column_mapping(source_columns: Sequence[str]) -> Dict[str, Optional[str]]:
    normalized_to_source: Dict[str, str] = {}
    for col in source_columns:
        normalized_to_source[normalize_column_name(col)] = col

    mapping: Dict[str, Optional[str]] = {}
    missing_required: List[str] = []

    for target in TARGET_COLUMNS:
        candidates = [target] + COLUMN_ALIASES.get(target, [])
        source_match = None
        for candidate in candidates:
            candidate_key = normalize_column_name(candidate)
            if candidate_key in normalized_to_source:
                source_match = normalized_to_source[candidate_key]
                break

        mapping[target] = source_match
        if source_match is None and target in REQUIRED_COLUMNS:
            missing_required.append(target)

    if missing_required:
        missing_str = ", ".join(missing_required)
        raise ValueError(f"missing required source columns after mapping: {missing_str}")

    return mapping


def parse_timestamps(series: pd.Series) -> pd.Series:
    raw = series.astype("string").str.strip()
    raw = raw.replace("", pd.NA)

    parsed = pd.to_datetime(raw, errors="coerce", dayfirst=True, utc=True)
    fallback_mask = parsed.isna() & raw.notna()
    if fallback_mask.any():
        fallback = pd.to_datetime(raw[fallback_mask], errors="coerce", dayfirst=False, utc=True)
        parsed.loc[fallback_mask] = fallback

    return parsed


def standardize_and_validate_chunk(
    chunk: pd.DataFrame,
    column_mapping: Dict[str, Optional[str]],
) -> Tuple[pd.DataFrame, int]:
    standardized = pd.DataFrame(index=chunk.index)

    for target in TARGET_COLUMNS:
        source_col = column_mapping.get(target)
        standardized[target] = chunk[source_col] if source_col else pd.NA

    for col in TEXT_COLUMNS:
        standardized[col] = standardized[col].astype("string").str.strip()
        standardized[col] = standardized[col].replace("", pd.NA)

    standardized["ts"] = parse_timestamps(standardized["ts"])
    standardized["customer_satisfaction"] = pd.to_numeric(
        standardized["customer_satisfaction"],
        errors="coerce",
    )

    valid_mask = standardized[REQUIRED_COLUMNS].notna().all(axis=1)
    valid_mask &= standardized["customer_satisfaction"].notna()
    valid_rows = standardized.loc[valid_mask, TARGET_COLUMNS].copy()
    dropped_rows = int((~valid_mask).sum())

    return valid_rows, dropped_rows


def dataframe_to_records(df: pd.DataFrame) -> List[Tuple[object, ...]]:
    df_for_insert = df.where(pd.notna(df), None).copy()

    if "ts" in df_for_insert.columns:
        if (
            pd.api.types.is_datetime64_any_dtype(df_for_insert["ts"])
            or pd.api.types.is_datetime64tz_dtype(df_for_insert["ts"])
        ):
            df_for_insert["ts"] = df_for_insert["ts"].apply(
                lambda value: value.to_pydatetime() if isinstance(value, pd.Timestamp) else value
            )
        else:
            df_for_insert["ts"] = df_for_insert["ts"].apply(
                lambda value: None if pd.isna(value) else value
            )

    def _to_python_scalar(value: object) -> object:
        if pd.isna(value):
            return None
        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime()
        if hasattr(value, "item") and not isinstance(value, (str, bytes)):
            try:
                return value.item()
            except Exception:
                return value
        return value

    records: List[Tuple[object, ...]] = []

    for row in df_for_insert.itertuples(index=False, name=None):
        records.append(tuple(_to_python_scalar(value) for value in row))

    return records


def iter_csv_chunks(path: Path, delimiter: str, chunksize: int) -> Iterable[pd.DataFrame]:
    return pd.read_csv(
        path,
        sep=delimiter,
        chunksize=chunksize,
        dtype=str,
        keep_default_na=False,
    )


def load_database_url() -> str:
    load_dotenv(dotenv_path=repo_root() / ".env")
    database_url = os.getenv("DATABASE_URL_DIRECT")
    if not database_url:
        raise RuntimeError("DATABASE_URL_DIRECT is not set")
    return database_url


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Load Incident Management CSV into im.event_raw.")
    parser.add_argument(
        "--path",
        default="data/local/Incident_Management_CSV.csv",
        help="Path to CSV file (default: data/local/Incident_Management_CSV.csv).",
    )
    parser.add_argument(
        "--truncate",
        action="store_true",
        help="Truncate im.event_raw before inserting.",
    )
    parser.add_argument(
        "--delimiter",
        default=";",
        help="CSV delimiter (default: ';').",
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=5000,
        help="Chunk size for CSV processing and insert batch size (default: 5000).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print stats without inserting rows.",
    )
    return parser.parse_args()


def run(args: argparse.Namespace) -> int:
    if args.chunksize <= 0:
        print("FAIL chunksize must be > 0")
        return 1

    csv_path = Path(args.path)
    if not csv_path.is_absolute():
        csv_path = repo_root() / csv_path

    if not csv_path.exists():
        print(f"FAIL csv file not found: {csv_path}")
        return 1

    try:
        chunk_iter = iter_csv_chunks(csv_path, args.delimiter, args.chunksize)
        first_chunk = next(chunk_iter)
    except StopIteration:
        print(f"FAIL csv file is empty: {csv_path}")
        return 1
    except Exception as exc:
        print(f"FAIL could not read csv: {exc}")
        return 1

    try:
        column_mapping = resolve_column_mapping(first_chunk.columns.tolist())
    except Exception as exc:
        print(f"FAIL column mapping: {exc}")
        return 1

    print("column_mapping:")
    for target in TARGET_COLUMNS:
        source = column_mapping.get(target)
        if source is None and target == "resolver":
            print(f"  {target} <- (missing in source, loading NULL)")
        else:
            print(f"  {target} <- {source}")

    stats = LoadStats()

    def process_chunk(chunk: pd.DataFrame, cursor: Optional[psycopg2.extensions.cursor]) -> None:
        validated_df, dropped_rows = standardize_and_validate_chunk(chunk, column_mapping)
        stats.update(validated_df=validated_df, input_rows=len(chunk), dropped_rows=dropped_rows)

        if cursor is None or validated_df.empty:
            return

        records = dataframe_to_records(validated_df)
        execute_values(cursor, INSERT_SQL, records, page_size=args.chunksize)

    if args.dry_run:
        try:
            process_chunk(first_chunk, cursor=None)
            for chunk in chunk_iter:
                process_chunk(chunk, cursor=None)
        except Exception as exc:
            print(f"FAIL validation: {exc}")
            return 1

        stats.print_summary(dry_run=True)
        return 0

    try:
        database_url = load_database_url()
        with psycopg2.connect(database_url) as conn:
            with conn.cursor() as cur:
                if args.truncate:
                    cur.execute("TRUNCATE TABLE im.event_raw")
                    print("OK truncated im.event_raw")

                process_chunk(first_chunk, cursor=cur)
                for chunk in chunk_iter:
                    process_chunk(chunk, cursor=cur)
    except Exception as exc:
        print(f"FAIL database load: {exc}")
        return 1

    stats.print_summary(dry_run=False)
    return 0


def main() -> int:
    args = parse_args()
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
