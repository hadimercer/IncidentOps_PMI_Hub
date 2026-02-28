-- IncidentOps PMI Hub - analytics views

CREATE SCHEMA IF NOT EXISTS im;

CREATE OR REPLACE VIEW im.v_event_seq AS
SELECT
    s.case_id,
    s.variant,
    s.priority,
    s.reporter,
    s.ts,
    s.event,
    s.issue_type,
    s.resolver,
    s.report_channel,
    s.short_description,
    s.customer_satisfaction,
    s.seq,
    s.next_ts,
    s.next_event,
    EXTRACT(EPOCH FROM (s.next_ts - s.ts)) / 3600.0 AS delta_hours
FROM (
    SELECT
        er.case_id,
        er.variant,
        er.priority,
        er.reporter,
        er.ts,
        er.event,
        er.issue_type,
        er.resolver,
        er.report_channel,
        er.short_description,
        er.customer_satisfaction,
        ROW_NUMBER() OVER (
            PARTITION BY er.case_id
            ORDER BY er.ts, er.event
        ) AS seq,
        LEAD(er.ts) OVER (
            PARTITION BY er.case_id
            ORDER BY er.ts, er.event
        ) AS next_ts,
        LEAD(er.event) OVER (
            PARTITION BY er.case_id
            ORDER BY er.ts, er.event
        ) AS next_event
    FROM im.event_raw er
) s;

-- TODO: add case-level summary view(s), e.g., cycle time and first-time-fix KPIs.
-- TODO: add transition-level aggregation view(s), e.g., edge frequencies and median wait time.
-- TODO: add bottleneck and variant analytics view(s).
