-- IncidentOps PMI Hub - schema and table creation
-- Idempotent DDL for Postgres

CREATE SCHEMA IF NOT EXISTS im;

-- Raw event log table; columns mirror inbound CSV structure.
CREATE TABLE IF NOT EXISTS im.event_raw (
    case_id text NOT NULL,
    variant text,
    priority text,
    reporter text,
    ts timestamptz NOT NULL,
    event text NOT NULL,
    issue_type text,
    resolver text,
    report_channel text,
    short_description text,
    customer_satisfaction numeric(5,2),
    event_key text
);

ALTER TABLE im.event_raw
    ADD COLUMN IF NOT EXISTS event_key text;

UPDATE im.event_raw
SET event_key = md5(
    concat_ws(
        '|',
        COALESCE(case_id, ''),
        COALESCE(variant, ''),
        COALESCE(priority, ''),
        COALESCE(reporter, ''),
        COALESCE(to_char(ts AT TIME ZONE 'UTC', 'YYYY-MM-DD"T"HH24:MI:SS.US"Z"'), ''),
        COALESCE(event, ''),
        COALESCE(issue_type, ''),
        COALESCE(resolver, ''),
        COALESCE(report_channel, ''),
        COALESCE(short_description, ''),
        COALESCE(replace(to_char(customer_satisfaction, 'FM9999999999999990D00'), ',', '.'), '')
    )
)
WHERE event_key IS NULL
   OR event_key = '';

WITH duplicate_rows AS (
    SELECT
        ctid,
        ROW_NUMBER() OVER (
            PARTITION BY event_key
            ORDER BY ts, ctid
        ) AS row_num
    FROM im.event_raw
    WHERE event_key IS NOT NULL
)
DELETE FROM im.event_raw er
USING duplicate_rows dup
WHERE er.ctid = dup.ctid
  AND dup.row_num > 1;

ALTER TABLE im.event_raw
    ALTER COLUMN event_key SET NOT NULL;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'uq_event_raw_event_key'
          AND conrelid = 'im.event_raw'::regclass
    ) THEN
        ALTER TABLE im.event_raw
            ADD CONSTRAINT uq_event_raw_event_key UNIQUE (event_key);
    END IF;
END
$$;

CREATE INDEX IF NOT EXISTS idx_event_raw_case_ts
    ON im.event_raw (case_id, ts);

CREATE INDEX IF NOT EXISTS idx_event_raw_case_ts_key
    ON im.event_raw (case_id, ts, event_key);

CREATE INDEX IF NOT EXISTS idx_event_raw_event
    ON im.event_raw (event);

CREATE INDEX IF NOT EXISTS idx_event_raw_variant
    ON im.event_raw (variant);

CREATE INDEX IF NOT EXISTS idx_event_raw_issue_type
    ON im.event_raw (issue_type);

CREATE INDEX IF NOT EXISTS idx_event_raw_priority
    ON im.event_raw (priority);

CREATE TABLE IF NOT EXISTS im.event_stage (
    event_key text PRIMARY KEY,
    case_id text NOT NULL,
    variant text,
    priority text,
    reporter text,
    ts timestamptz NOT NULL,
    event text NOT NULL,
    issue_type text,
    resolver text,
    report_channel text,
    short_description text,
    customer_satisfaction numeric(5,2)
);

CREATE TABLE IF NOT EXISTS im.sla_policy (
    priority text PRIMARY KEY,
    target_hours integer NOT NULL CHECK (target_hours > 0)
);

CREATE TABLE IF NOT EXISTS im.event_catalog (
    event_name text PRIMARY KEY,
    description text NOT NULL,
    stakeholder_owner text NOT NULL
);

CREATE TABLE IF NOT EXISTS im.event_alias (
    raw_event_name text PRIMARY KEY,
    event_code text NOT NULL REFERENCES im.event_catalog(event_name)
);

CREATE INDEX IF NOT EXISTS idx_event_alias_event_code
    ON im.event_alias (event_code);
