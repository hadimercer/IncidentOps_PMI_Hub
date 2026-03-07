-- IncidentOps PMI Hub - analytics views

CREATE SCHEMA IF NOT EXISTS im;

DROP VIEW IF EXISTS im.v_kb_enablement_candidates CASCADE;
DROP VIEW IF EXISTS im.v_fcr_summary CASCADE;
DROP VIEW IF EXISTS im.v_fcr_cases CASCADE;
DROP VIEW IF EXISTS im.v_resolution_level CASCADE;
DROP VIEW IF EXISTS im.v_channel_issue_summary CASCADE;
DROP VIEW IF EXISTS im.v_channel_summary CASCADE;
DROP VIEW IF EXISTS im.v_problem_candidate_top_cases CASCADE;
DROP VIEW IF EXISTS im.v_problem_candidates CASCADE;
DROP VIEW IF EXISTS im.v_problem_candidate_cases CASCADE;
DROP VIEW IF EXISTS im.v_cx_breakdown CASCADE;
DROP VIEW IF EXISTS im.v_cx_summary CASCADE;
DROP VIEW IF EXISTS im.v_closure_compliance CASCADE;
DROP VIEW IF EXISTS im.v_worst_handoff_cases CASCADE;
DROP VIEW IF EXISTS im.v_handoff_summary CASCADE;
DROP VIEW IF EXISTS im.v_pingpong_cases CASCADE;
DROP VIEW IF EXISTS im.v_transition_by_variant CASCADE;
DROP VIEW IF EXISTS im.v_dwell_by_event CASCADE;
DROP VIEW IF EXISTS im.v_transition_summary CASCADE;
DROP VIEW IF EXISTS im.v_variant_summary CASCADE;
DROP VIEW IF EXISTS im.v_case_sla CASCADE;
DROP VIEW IF EXISTS im.v_case CASCADE;
DROP VIEW IF EXISTS im.v_event_seq CASCADE;
DROP VIEW IF EXISTS im.v_event_normalized CASCADE;

CREATE OR REPLACE VIEW im.v_event_normalized AS
SELECT
    er.event_key,
    er.case_id,
    er.variant,
    er.priority,
    er.reporter,
    er.ts,
    er.event AS raw_event_name,
    COALESCE(ea.event_code, er.event) AS event,
    ea.event_code,
    er.issue_type,
    er.resolver,
    er.report_channel,
    er.short_description,
    er.customer_satisfaction,
    CASE
        WHEN ea.event_code IN ('ticket_solved_l1', 'assigned_to_l1', 'work_in_progress_l1') THEN 'L1'
        WHEN ea.event_code IN ('ticket_solved_l2', 'assigned_to_l2', 'escalated_to_l2', 'work_in_progress_l2') THEN 'L2'
        WHEN ea.event_code IN ('ticket_solved_l3', 'assigned_to_l3', 'escalated_to_l3', 'work_in_progress_l3') THEN 'L3'
        ELSE NULL
    END AS event_level
FROM im.event_raw er
LEFT JOIN im.event_alias ea
    ON ea.raw_event_name = er.event;

CREATE OR REPLACE VIEW im.v_event_seq AS
SELECT
    s.case_id,
    s.variant,
    s.priority,
    s.reporter,
    s.ts,
    s.event,
    s.raw_event_name,
    s.event_code,
    s.event_level,
    s.issue_type,
    s.resolver,
    s.report_channel,
    s.short_description,
    s.customer_satisfaction,
    s.seq,
    s.next_ts,
    s.next_event,
    s.next_raw_event_name,
    s.next_event_code,
    s.next_event_level,
    EXTRACT(EPOCH FROM (s.next_ts - s.ts)) / 3600.0 AS delta_hours
FROM (
    SELECT
        ven.case_id,
        ven.variant,
        ven.priority,
        ven.reporter,
        ven.ts,
        ven.event,
        ven.raw_event_name,
        ven.event_code,
        ven.event_level,
        ven.issue_type,
        ven.resolver,
        ven.report_channel,
        ven.short_description,
        ven.customer_satisfaction,
        ROW_NUMBER() OVER (
            PARTITION BY ven.case_id
            ORDER BY ven.ts, ven.event_key
        ) AS seq,
        LEAD(ven.ts) OVER (
            PARTITION BY ven.case_id
            ORDER BY ven.ts, ven.event_key
        ) AS next_ts,
        LEAD(ven.event) OVER (
            PARTITION BY ven.case_id
            ORDER BY ven.ts, ven.event_key
        ) AS next_event,
        LEAD(ven.raw_event_name) OVER (
            PARTITION BY ven.case_id
            ORDER BY ven.ts, ven.event_key
        ) AS next_raw_event_name,
        LEAD(ven.event_code) OVER (
            PARTITION BY ven.case_id
            ORDER BY ven.ts, ven.event_key
        ) AS next_event_code,
        LEAD(ven.event_level) OVER (
            PARTITION BY ven.case_id
            ORDER BY ven.ts, ven.event_key
        ) AS next_event_level
    FROM im.v_event_normalized ven
) s;

CREATE OR REPLACE VIEW im.v_case AS
WITH normalized AS (
    SELECT *
    FROM im.v_event_normalized
),
case_bounds AS (
    SELECT
        n.case_id,
        COALESCE(
            MIN(n.ts) FILTER (WHERE n.event_code = 'ticket_created'),
            MIN(n.ts)
        ) AS created_at,
        MAX(n.ts) FILTER (WHERE n.event_code = 'ticket_closed') AS closed_at,
        COUNT(*) AS event_count,
        COUNT(DISTINCT n.resolver) FILTER (
            WHERE n.resolver IS NOT NULL
              AND n.resolver <> ''
        ) AS distinct_resolvers,
        COUNT(*) FILTER (
            WHERE n.event_code IN (
                'escalated_to_l2',
                'assigned_to_l2',
                'escalated_to_l3',
                'assigned_to_l3'
            )
        ) AS escalation_count,
        BOOL_OR(n.event_code = 'customer_feedback_received') AS has_feedback,
        BOOL_OR(n.event_code = 'ticket_reopened') AS has_reopen,
        BOOL_OR(n.event_code = 'ticket_rejected') AS has_reject,
        MAX(n.customer_satisfaction) AS customer_satisfaction
    FROM normalized n
    GROUP BY n.case_id
),
first_priority AS (
    SELECT DISTINCT ON (n.case_id)
        n.case_id,
        n.priority
    FROM normalized n
    WHERE n.priority IS NOT NULL
      AND n.priority <> ''
    ORDER BY n.case_id, n.ts, n.event_key
),
first_issue_type AS (
    SELECT DISTINCT ON (n.case_id)
        n.case_id,
        n.issue_type
    FROM normalized n
    WHERE n.issue_type IS NOT NULL
      AND n.issue_type <> ''
    ORDER BY n.case_id, n.ts, n.event_key
),
first_report_channel AS (
    SELECT DISTINCT ON (n.case_id)
        n.case_id,
        n.report_channel
    FROM normalized n
    WHERE n.report_channel IS NOT NULL
      AND n.report_channel <> ''
    ORDER BY n.case_id, n.ts, n.event_key
),
variant_rank AS (
    SELECT
        n.case_id,
        n.variant,
        COUNT(*) AS variant_count,
        MAX(n.ts) AS last_seen_ts
    FROM normalized n
    WHERE n.variant IS NOT NULL
      AND n.variant <> ''
    GROUP BY n.case_id, n.variant
),
variant_mode AS (
    SELECT DISTINCT ON (vr.case_id)
        vr.case_id,
        vr.variant
    FROM variant_rank vr
    ORDER BY vr.case_id, vr.variant_count DESC, vr.last_seen_ts DESC, vr.variant
)
SELECT
    cb.case_id,
    vm.variant,
    fp.priority,
    fi.issue_type,
    frc.report_channel,
    cb.created_at,
    cb.closed_at,
    EXTRACT(EPOCH FROM (cb.closed_at - cb.created_at)) / 3600.0 AS cycle_hours,
    cb.event_count,
    cb.distinct_resolvers,
    GREATEST(cb.distinct_resolvers - 1, 0) AS resolver_changes,
    cb.escalation_count,
    cb.has_feedback,
    cb.has_reopen,
    cb.has_reject,
    cb.customer_satisfaction
FROM case_bounds cb
LEFT JOIN variant_mode vm
    ON vm.case_id = cb.case_id
LEFT JOIN first_priority fp
    ON fp.case_id = cb.case_id
LEFT JOIN first_issue_type fi
    ON fi.case_id = cb.case_id
LEFT JOIN first_report_channel frc
    ON frc.case_id = cb.case_id
WHERE cb.created_at IS NOT NULL
  AND cb.closed_at IS NOT NULL
  AND cb.closed_at >= cb.created_at;

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
    WHERE ves.event_level IN ('L2', 'L3')
      AND ves.next_event_level IN ('L2', 'L3')
      AND ves.event_level <> ves.next_event_level
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
        ven.case_id,
        MAX(ven.ts) AS last_close_ts
    FROM im.v_event_normalized ven
    WHERE ven.event_code = 'ticket_closed'
    GROUP BY ven.case_id
),
first_reopen_after_close AS (
    SELECT
        lc.case_id,
        MIN(ven.ts) AS first_reopen_ts_after_close
    FROM last_close lc
    JOIN im.v_event_normalized ven
        ON ven.case_id = lc.case_id
    WHERE ven.event_code = 'ticket_reopened'
      AND ven.ts > lc.last_close_ts
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
        ven.case_id,
        MAX(ven.short_description) AS short_description_raw,
        NULLIF(
            trim(
                regexp_replace(
                    regexp_replace(
                        lower(COALESCE(MAX(ven.short_description), '')),
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
    FROM im.v_event_normalized ven
    GROUP BY ven.case_id
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

CREATE OR REPLACE VIEW im.v_channel_summary AS
SELECT
    vcs.report_channel,
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
    AVG(CASE WHEN vcs.has_reopen THEN 1.0 ELSE 0.0 END) AS reopen_rate,
    AVG(CASE WHEN vcs.has_reject THEN 1.0 ELSE 0.0 END) AS reject_rate,
    AVG(vcs.escalation_count::double precision) AS avg_escalations,
    AVG(vcs.resolver_changes::double precision) AS avg_resolver_changes,
    AVG(CASE WHEN vcs.has_feedback THEN 1.0 ELSE 0.0 END) AS feedback_rate,
    1.0 - AVG(CASE WHEN vcs.has_feedback THEN 1.0 ELSE 0.0 END) AS missing_feedback_rate
FROM im.v_case_sla vcs
GROUP BY vcs.report_channel;

CREATE OR REPLACE VIEW im.v_channel_issue_summary AS
SELECT
    vcs.report_channel,
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
    AVG(CASE WHEN vcs.has_reopen THEN 1.0 ELSE 0.0 END) AS reopen_rate,
    AVG(CASE WHEN vcs.has_reject THEN 1.0 ELSE 0.0 END) AS reject_rate,
    AVG(vcs.escalation_count::double precision) AS avg_escalations,
    AVG(vcs.resolver_changes::double precision) AS avg_resolver_changes,
    AVG(CASE WHEN vcs.has_feedback THEN 1.0 ELSE 0.0 END) AS feedback_rate,
    1.0 - AVG(CASE WHEN vcs.has_feedback THEN 1.0 ELSE 0.0 END) AS missing_feedback_rate
FROM im.v_case_sla vcs
GROUP BY vcs.report_channel, vcs.issue_type;

CREATE OR REPLACE VIEW im.v_resolution_level AS
WITH case_close AS (
    SELECT
        vc.case_id,
        vc.closed_at
    FROM im.v_case vc
),
event_flags AS (
    SELECT
        ven.case_id,
        BOOL_OR(ven.event_code = 'ticket_solved_l1') AS solved_l1,
        BOOL_OR(ven.event_code = 'ticket_solved_l2') AS solved_l2,
        BOOL_OR(ven.event_code = 'ticket_solved_l3') AS solved_l3,
        BOOL_OR(ven.event_code = 'ticket_reopened') AS reopened,
        BOOL_OR(ven.event_code IN ('escalated_to_l2', 'assigned_to_l2')) AS escalated_to_l2,
        BOOL_OR(ven.event_code IN ('escalated_to_l3', 'assigned_to_l3')) AS escalated_to_l3
    FROM im.v_event_normalized ven
    GROUP BY ven.case_id
),
last_solve AS (
    SELECT DISTINCT ON (ven.case_id)
        ven.case_id,
        ven.event_code
    FROM im.v_event_normalized ven
    JOIN case_close cc
        ON cc.case_id = ven.case_id
    WHERE ven.event_code IN ('ticket_solved_l1', 'ticket_solved_l2', 'ticket_solved_l3')
      AND ven.ts <= cc.closed_at
    ORDER BY ven.case_id, ven.ts DESC, ven.event_key DESC
)
SELECT
    vcs.case_id,
    COALESCE(ef.solved_l1, FALSE) AS solved_l1,
    COALESCE(ef.solved_l2, FALSE) AS solved_l2,
    COALESCE(ef.solved_l3, FALSE) AS solved_l3,
    COALESCE(ef.reopened, FALSE) AS reopened,
    COALESCE(ef.escalated_to_l2, FALSE) AS escalated_to_l2,
    COALESCE(ef.escalated_to_l3, FALSE) AS escalated_to_l3,
    CASE ls.event_code
        WHEN 'ticket_solved_l3' THEN 'L3'
        WHEN 'ticket_solved_l2' THEN 'L2'
        WHEN 'ticket_solved_l1' THEN 'L1'
        ELSE 'Unknown'
    END AS resolution_level
FROM im.v_case_sla vcs
LEFT JOIN event_flags ef
    ON ef.case_id = vcs.case_id
LEFT JOIN last_solve ls
    ON ls.case_id = vcs.case_id;

CREATE OR REPLACE VIEW im.v_fcr_cases AS
SELECT
    vcs.case_id,
    vcs.variant,
    vcs.priority,
    vcs.issue_type,
    vcs.report_channel,
    vcs.cycle_hours,
    vcs.customer_satisfaction,
    vcs.met_sla,
    vrl.resolution_level,
    (
        vrl.resolution_level = 'L1'
        AND NOT vrl.reopened
        AND NOT vrl.escalated_to_l2
        AND NOT vrl.escalated_to_l3
    ) AS fcr
FROM im.v_case_sla vcs
JOIN im.v_resolution_level vrl
    ON vrl.case_id = vcs.case_id;

CREATE OR REPLACE VIEW im.v_fcr_summary AS
SELECT
    vfc.issue_type,
    COUNT(*) AS cases,
    AVG(CASE WHEN vfc.fcr THEN 1.0 ELSE 0.0 END) AS fcr_rate,
    AVG(vfc.cycle_hours) AS avg_cycle_hours,
    AVG(vfc.customer_satisfaction) AS avg_csat,
    AVG(
        CASE
            WHEN vfc.met_sla IS TRUE THEN 1.0
            WHEN vfc.met_sla IS FALSE THEN 0.0
            ELSE NULL
        END
    ) AS met_sla_rate,
    AVG(CASE WHEN vrl.reopened THEN 1.0 ELSE 0.0 END) AS reopen_rate,
    AVG(CASE WHEN vrl.escalated_to_l2 THEN 1.0 ELSE 0.0 END) AS l2_escalation_rate,
    AVG(CASE WHEN vrl.escalated_to_l3 THEN 1.0 ELSE 0.0 END) AS l3_escalation_rate
FROM im.v_fcr_cases vfc
JOIN im.v_resolution_level vrl
    ON vrl.case_id = vfc.case_id
GROUP BY vfc.issue_type;

CREATE OR REPLACE VIEW im.v_kb_enablement_candidates AS
SELECT
    vfc.issue_type,
    COUNT(*) AS cases,
    AVG(CASE WHEN vrl.resolution_level = 'L1' THEN 1.0 ELSE 0.0 END) AS l1_solved_rate,
    AVG(CASE WHEN vrl.resolution_level IN ('L2', 'L3') THEN 1.0 ELSE 0.0 END) AS l2_or_l3_rate,
    AVG(CASE WHEN vrl.escalated_to_l2 THEN 1.0 ELSE 0.0 END) AS escalation_to_l2_rate,
    AVG(CASE WHEN vrl.reopened THEN 1.0 ELSE 0.0 END) AS reopen_rate,
    AVG(vfc.cycle_hours) AS avg_cycle_hours,
    AVG(vfc.customer_satisfaction) AS avg_csat,
    (
        COUNT(*)
        * (
            AVG(CASE WHEN vrl.resolution_level IN ('L2', 'L3') THEN 1.0 ELSE 0.0 END)
            + AVG(CASE WHEN vrl.escalated_to_l2 THEN 1.0 ELSE 0.0 END)
        )
        * (AVG(vfc.cycle_hours) / 10.0)
    ) AS enablement_score
FROM im.v_fcr_cases vfc
JOIN im.v_resolution_level vrl
    ON vrl.case_id = vfc.case_id
GROUP BY vfc.issue_type;

