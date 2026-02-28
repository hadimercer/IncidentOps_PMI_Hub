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

CREATE OR REPLACE VIEW im.v_pingpong_cases AS
WITH pingpong AS (
    SELECT
        ves.case_id,
        COUNT(*) AS pingpong_transitions
    FROM im.v_event_seq ves
    WHERE (
        ves.event ILIKE '%level 2%'
        AND ves.next_event ILIKE '%level 3%'
    ) OR (
        ves.event ILIKE '%level 3%'
        AND ves.next_event ILIKE '%level 2%'
    )
    GROUP BY ves.case_id
)
SELECT
    vcs.case_id,
    vcs.variant,
    vcs.priority,
    vcs.issue_type,
    vcs.report_channel,
    vcs.cycle_hours,
    vcs.customer_satisfaction,
    vcs.escalation_count,
    vcs.resolver_changes,
    p.pingpong_transitions,
    vcs.has_reopen,
    vcs.has_reject,
    vcs.met_sla
FROM pingpong p
JOIN im.v_case_sla vcs
    ON vcs.case_id = p.case_id;

CREATE OR REPLACE VIEW im.v_handoff_summary AS
SELECT
    vcs.variant,
    vcs.issue_type,
    COUNT(*) AS cases,
    AVG(vcs.cycle_hours) AS avg_cycle_hours,
    percentile_cont(0.9) WITHIN GROUP (ORDER BY vcs.cycle_hours) AS p90_cycle_hours,
    AVG(vcs.customer_satisfaction) AS avg_csat,
    AVG(
        CASE
            WHEN vcs.met_sla IS TRUE THEN 1.0
            WHEN vcs.met_sla IS FALSE THEN 0.0
            ELSE NULL
        END
    ) AS met_sla_rate,
    AVG(vcs.escalation_count::double precision) AS avg_escalations,
    AVG(vcs.resolver_changes::double precision) AS avg_resolver_changes,
    AVG(CASE WHEN vcs.resolver_changes >= 1 THEN 1.0 ELSE 0.0 END) AS handoff_rate,
    AVG(CASE WHEN vcs.resolver_changes >= 3 THEN 1.0 ELSE 0.0 END) AS high_handoff_rate,
    AVG(CASE WHEN vpc.case_id IS NOT NULL THEN 1.0 ELSE 0.0 END) AS pingpong_rate
FROM im.v_case_sla vcs
LEFT JOIN im.v_pingpong_cases vpc
    ON vpc.case_id = vcs.case_id
GROUP BY vcs.variant, vcs.issue_type;

CREATE OR REPLACE VIEW im.v_worst_handoff_cases AS
SELECT
    vcs.case_id,
    vcs.variant,
    vcs.priority,
    vcs.issue_type,
    vcs.report_channel,
    vcs.cycle_hours,
    vcs.customer_satisfaction,
    vcs.met_sla,
    vcs.escalation_count,
    vcs.resolver_changes,
    vcs.has_reopen,
    vcs.has_reject,
    vcs.has_feedback
FROM im.v_case_sla vcs
ORDER BY
    vcs.resolver_changes DESC,
    vcs.escalation_count DESC,
    vcs.cycle_hours DESC
LIMIT 200;

CREATE OR REPLACE VIEW im.v_closure_compliance AS
WITH last_close AS (
    SELECT
        er.case_id,
        MAX(er.ts) AS last_close_ts
    FROM im.event_raw er
    WHERE er.event = 'Ticket closed'
    GROUP BY er.case_id
),
first_reopen_after_close AS (
    SELECT
        lc.case_id,
        MIN(er.ts) AS first_reopen_ts_after_close
    FROM last_close lc
    JOIN im.event_raw er
        ON er.case_id = lc.case_id
    WHERE er.event ILIKE '%reopen%'
      AND er.ts > lc.last_close_ts
    GROUP BY lc.case_id
)
SELECT
    vcs.case_id,
    vcs.variant,
    vcs.priority,
    vcs.issue_type,
    vcs.report_channel,
    vcs.cycle_hours,
    vcs.customer_satisfaction,
    vcs.met_sla,
    vcs.has_feedback,
    vcs.has_reopen,
    vcs.has_reject,
    NOT COALESCE(vcs.has_feedback, FALSE) AS closed_without_feedback,
    EXTRACT(EPOCH FROM (fr.first_reopen_ts_after_close - lc.last_close_ts)) / 3600.0 AS reopen_after_close_hours,
    (
        EXTRACT(EPOCH FROM (fr.first_reopen_ts_after_close - lc.last_close_ts)) / 3600.0
    ) IS NOT NULL
    AND (
        EXTRACT(EPOCH FROM (fr.first_reopen_ts_after_close - lc.last_close_ts)) / 3600.0
    ) <= 3 AS reopened_within_3h,
    (
        EXTRACT(EPOCH FROM (fr.first_reopen_ts_after_close - lc.last_close_ts)) / 3600.0
    ) IS NOT NULL
    AND (
        EXTRACT(EPOCH FROM (fr.first_reopen_ts_after_close - lc.last_close_ts)) / 3600.0
    ) <= 24 AS reopened_within_24h
FROM im.v_case_sla vcs
LEFT JOIN last_close lc
    ON lc.case_id = vcs.case_id
LEFT JOIN first_reopen_after_close fr
    ON fr.case_id = vcs.case_id;

CREATE OR REPLACE VIEW im.v_cx_summary AS
SELECT
    COUNT(*) AS cases,
    AVG(vcc.customer_satisfaction) AS avg_csat,
    AVG(
        CASE
            WHEN vcc.met_sla IS TRUE THEN 1.0
            WHEN vcc.met_sla IS FALSE THEN 0.0
            ELSE NULL
        END
    ) AS met_sla_rate,
    AVG(CASE WHEN vcc.closed_without_feedback THEN 1.0 ELSE 0.0 END) AS closed_without_feedback_rate,
    AVG(CASE WHEN vcc.reopened_within_3h THEN 1.0 ELSE 0.0 END) AS reopened_within_3h_rate,
    AVG(CASE WHEN vcc.reopened_within_24h THEN 1.0 ELSE 0.0 END) AS reopened_within_24h_rate,
    AVG(CASE WHEN vcc.has_reject THEN 1.0 ELSE 0.0 END) AS reject_rate,
    AVG(vcc.cycle_hours) AS avg_cycle_hours,
    percentile_cont(0.9) WITHIN GROUP (ORDER BY vcc.cycle_hours) AS p90_cycle_hours
FROM im.v_closure_compliance vcc;

CREATE OR REPLACE VIEW im.v_cx_breakdown AS
SELECT
    vcc.issue_type,
    vcc.priority,
    COUNT(*) AS cases,
    AVG(vcc.customer_satisfaction) AS avg_csat,
    AVG(
        CASE
            WHEN vcc.met_sla IS TRUE THEN 1.0
            WHEN vcc.met_sla IS FALSE THEN 0.0
            ELSE NULL
        END
    ) AS met_sla_rate,
    AVG(CASE WHEN vcc.closed_without_feedback THEN 1.0 ELSE 0.0 END) AS closed_without_feedback_rate,
    AVG(CASE WHEN vcc.reopened_within_3h THEN 1.0 ELSE 0.0 END) AS reopened_within_3h_rate,
    AVG(CASE WHEN vcc.has_reject THEN 1.0 ELSE 0.0 END) AS reject_rate,
    AVG(vcc.cycle_hours) AS avg_cycle_hours
FROM im.v_closure_compliance vcc
GROUP BY vcc.issue_type, vcc.priority;

CREATE OR REPLACE VIEW im.v_problem_candidate_cases AS
WITH case_descriptions AS (
    SELECT
        er.case_id,
        MAX(er.short_description) AS short_description_raw,
        NULLIF(
            trim(
                regexp_replace(
                    regexp_replace(
                        lower(COALESCE(MAX(er.short_description), '')),
                        '[^a-z\s]+',
                        ' ',
                        'g'
                    ),
                    '\s+',
                    ' ',
                    'g'
                )
            ),
            ''
        ) AS short_description_norm
    FROM im.event_raw er
    GROUP BY er.case_id
)
SELECT
    vcs.case_id,
    vcs.issue_type,
    vcs.priority,
    vcs.variant,
    vcs.report_channel,
    vcs.cycle_hours,
    vcs.customer_satisfaction,
    vcs.met_sla,
    vcs.has_reopen,
    vcs.has_reject,
    vcs.has_feedback,
    cd.short_description_raw,
    cd.short_description_norm
FROM im.v_case_sla vcs
LEFT JOIN case_descriptions cd
    ON cd.case_id = vcs.case_id;

CREATE OR REPLACE VIEW im.v_problem_candidates AS
WITH candidate_agg AS (
    SELECT
        vpcc.issue_type,
        vpcc.short_description_norm,
        COUNT(*) AS cases,
        AVG(vpcc.cycle_hours) AS avg_cycle_hours,
        percentile_cont(0.9) WITHIN GROUP (ORDER BY vpcc.cycle_hours) AS p90_cycle_hours,
        AVG(vpcc.customer_satisfaction) AS avg_csat,
        AVG(
            CASE
                WHEN vpcc.met_sla IS TRUE THEN 1.0
                WHEN vpcc.met_sla IS FALSE THEN 0.0
                ELSE NULL
            END
        ) AS met_sla_rate,
        AVG(CASE WHEN vpcc.has_reopen THEN 1.0 ELSE 0.0 END) AS reopen_rate,
        AVG(CASE WHEN vpcc.has_reject THEN 1.0 ELSE 0.0 END) AS reject_rate,
        MIN(vpcc.short_description_raw) AS example_description,
        MAX(vc.closed_at) AS last_seen_ts
    FROM im.v_problem_candidate_cases vpcc
    LEFT JOIN im.v_case vc
        ON vc.case_id = vpcc.case_id
    WHERE vpcc.short_description_norm IS NOT NULL
    GROUP BY vpcc.issue_type, vpcc.short_description_norm
)
SELECT
    ca.issue_type,
    ca.short_description_norm,
    ca.cases,
    ca.avg_cycle_hours,
    ca.p90_cycle_hours,
    ca.avg_csat,
    ca.met_sla_rate,
    ca.reopen_rate,
    ca.reject_rate,
    (
        ca.cases * (1.0 - COALESCE(ca.met_sla_rate, 0.0)) * (COALESCE(ca.avg_cycle_hours, 0.0) / 10.0)
        + ca.cases * (COALESCE(ca.reopen_rate, 0.0) + COALESCE(ca.reject_rate, 0.0))
    ) AS impact_score,
    ca.example_description,
    ca.last_seen_ts
FROM candidate_agg ca;

CREATE OR REPLACE VIEW im.v_problem_candidate_top_cases AS
SELECT
    vpcc.case_id,
    vpcc.issue_type,
    vpcc.priority,
    vpcc.variant,
    vpcc.report_channel,
    vpcc.cycle_hours,
    vpcc.customer_satisfaction,
    vpcc.met_sla,
    vpcc.has_reopen,
    vpcc.has_reject,
    vpcc.short_description_raw,
    vpcc.short_description_norm
FROM im.v_problem_candidate_cases vpcc
ORDER BY
    vpcc.cycle_hours DESC,
    vpcc.has_reopen DESC,
    vpcc.has_reject DESC
LIMIT 500;
