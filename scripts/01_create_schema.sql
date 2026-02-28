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
    customer_satisfaction numeric(5,2)
);

CREATE INDEX IF NOT EXISTS idx_event_raw_case_ts
    ON im.event_raw (case_id, ts);

CREATE INDEX IF NOT EXISTS idx_event_raw_event
    ON im.event_raw (event);

CREATE INDEX IF NOT EXISTS idx_event_raw_variant
    ON im.event_raw (variant);

CREATE INDEX IF NOT EXISTS idx_event_raw_issue_type
    ON im.event_raw (issue_type);

CREATE INDEX IF NOT EXISTS idx_event_raw_priority
    ON im.event_raw (priority);

CREATE TABLE IF NOT EXISTS im.sla_policy (
    priority text PRIMARY KEY,
    target_hours integer NOT NULL CHECK (target_hours > 0)
);

CREATE TABLE IF NOT EXISTS im.event_catalog (
    event_name text PRIMARY KEY,
    description text NOT NULL,
    stakeholder_owner text NOT NULL
);
