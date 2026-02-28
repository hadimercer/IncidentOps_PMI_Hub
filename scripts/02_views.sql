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

CREATE OR REPLACE VIEW im.v_case AS
WITH case_rollup AS (
    SELECT
        er.case_id,
        MAX(er.variant) AS variant,
        MAX(er.priority) AS priority,
        MAX(er.issue_type) AS issue_type,
        MAX(er.report_channel) AS report_channel,
        MIN(er.ts) FILTER (WHERE er.event = 'Ticket created') AS created_at,
        MAX(er.ts) FILTER (WHERE er.event = 'Ticket closed') AS closed_at,
        COUNT(*) AS event_count,
        COUNT(DISTINCT er.resolver) FILTER (
            WHERE er.resolver IS NOT NULL
              AND er.resolver <> ''
        ) AS distinct_resolvers,
        COUNT(*) FILTER (WHERE er.event ILIKE 'Ticket escalated%') AS escalation_count,
        BOOL_OR(er.event = 'Customer feedback received') AS has_feedback,
        BOOL_OR(er.event ILIKE '%reopen%') AS has_reopen,
        BOOL_OR(er.event ILIKE '%reject%') AS has_reject,
        MAX(er.customer_satisfaction) AS customer_satisfaction
    FROM im.event_raw er
    GROUP BY er.case_id
)
SELECT
    cr.case_id,
    cr.variant,
    cr.priority,
    cr.issue_type,
    cr.report_channel,
    cr.created_at,
    cr.closed_at,
    EXTRACT(EPOCH FROM (cr.closed_at - cr.created_at)) / 3600.0 AS cycle_hours,
    cr.event_count,
    cr.distinct_resolvers,
    GREATEST(cr.distinct_resolvers - 1, 0) AS resolver_changes,
    cr.escalation_count,
    cr.has_feedback,
    cr.has_reopen,
    cr.has_reject,
    cr.customer_satisfaction
FROM case_rollup cr
WHERE cr.created_at IS NOT NULL
  AND cr.closed_at IS NOT NULL
  AND cr.closed_at >= cr.created_at;

CREATE OR REPLACE VIEW im.v_case_sla AS
SELECT
    vc.case_id,
    vc.variant,
    vc.priority,
    vc.issue_type,
    vc.report_channel,
    vc.created_at,
    vc.closed_at,
    vc.cycle_hours,
    vc.event_count,
    vc.distinct_resolvers,
    vc.resolver_changes,
    vc.escalation_count,
    vc.has_feedback,
    vc.has_reopen,
    vc.has_reject,
    vc.customer_satisfaction,
    sp.target_hours,
    CASE
        WHEN sp.target_hours IS NULL THEN NULL
        ELSE vc.cycle_hours <= sp.target_hours::double precision
    END AS met_sla
FROM im.v_case vc
LEFT JOIN im.sla_policy sp
    ON sp.priority = vc.priority;

CREATE OR REPLACE VIEW im.v_variant_summary AS
SELECT
    vcs.variant,
    COUNT(*) AS cases,
    AVG(vcs.cycle_hours) AS avg_cycle_hours,
    percentile_cont(0.90) WITHIN GROUP (ORDER BY vcs.cycle_hours) AS p90_cycle_hours,
    AVG(vcs.customer_satisfaction) AS avg_csat,
    AVG(
        CASE
            WHEN vcs.met_sla IS TRUE THEN 1.0
            WHEN vcs.met_sla IS FALSE THEN 0.0
            ELSE NULL
        END
    ) AS met_sla_rate,
    AVG(CASE WHEN vcs.has_reopen THEN 1.0 ELSE 0.0 END) AS reopen_rate,
    AVG(CASE WHEN vcs.has_reject THEN 1.0 ELSE 0.0 END) AS reject_rate,
    AVG(vcs.escalation_count::double precision) AS avg_escalations,
    AVG(vcs.resolver_changes::double precision) AS avg_resolver_changes
FROM im.v_case_sla vcs
GROUP BY vcs.variant;

CREATE OR REPLACE VIEW im.v_transition_summary AS
SELECT
    ves.event AS from_event,
    ves.next_event AS to_event,
    COUNT(*) AS transition_count,
    AVG(ves.delta_hours) AS avg_delta_hours,
    percentile_cont(0.5) WITHIN GROUP (ORDER BY ves.delta_hours) AS median_delta_hours,
    percentile_cont(0.9) WITHIN GROUP (ORDER BY ves.delta_hours) AS p90_delta_hours
FROM im.v_event_seq ves
WHERE ves.next_ts IS NOT NULL
  AND ves.next_event IS NOT NULL
  AND ves.delta_hours IS NOT NULL
  AND ves.delta_hours >= 0
GROUP BY ves.event, ves.next_event;

CREATE OR REPLACE VIEW im.v_dwell_by_event AS
SELECT
    ves.event,
    COUNT(*) AS occurrences,
    AVG(ves.delta_hours) AS avg_dwell_hours,
    percentile_cont(0.5) WITHIN GROUP (ORDER BY ves.delta_hours) AS median_dwell_hours,
    percentile_cont(0.9) WITHIN GROUP (ORDER BY ves.delta_hours) AS p90_dwell_hours
FROM im.v_event_seq ves
WHERE ves.next_ts IS NOT NULL
  AND ves.delta_hours IS NOT NULL
  AND ves.delta_hours >= 0
GROUP BY ves.event;

CREATE OR REPLACE VIEW im.v_transition_by_variant AS
SELECT
    ves.variant,
    ves.event AS from_event,
    ves.next_event AS to_event,
    COUNT(*) AS transition_count,
    AVG(ves.delta_hours) AS avg_delta_hours
FROM im.v_event_seq ves
WHERE ves.next_ts IS NOT NULL
  AND ves.next_event IS NOT NULL
  AND ves.delta_hours IS NOT NULL
  AND ves.delta_hours >= 0
GROUP BY ves.variant, ves.event, ves.next_event;
