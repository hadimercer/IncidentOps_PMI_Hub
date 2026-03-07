from __future__ import annotations

import io
import os
import time
from typing import Optional
from urllib.parse import parse_qsl, urlencode, urlparse, urlunparse

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psycopg2
import streamlit as st
from dotenv import load_dotenv

from ui.charts import annotate as annotate_chart, style_figure
from ui.primitives import (
    empty_state,
    export_frame,
    global_filter_bar,
    metric_strip,
    narrative_callouts,
    page_masthead,
    section_header,
    sidebar_brand,
    summary_table,
    worklist_table,
)
from ui.theme import TOKENS, apply_theme

st.set_page_config(
    page_title="IncidentOps | PMI Hub",
    layout="wide",
    initial_sidebar_state="expanded",
)

CASE_SCOPE_SQL = """
SELECT
  case_id,
  variant,
  priority,
  issue_type,
  report_channel,
  created_at,
  closed_at,
  cycle_hours,
  event_count,
  escalation_count,
  resolver_changes,
  has_feedback,
  has_reopen,
  has_reject,
  customer_satisfaction,
  met_sla
FROM im.v_case_sla;
"""

EVENT_SCOPE_SQL = """
SELECT
  case_id,
  variant,
  priority,
  issue_type,
  report_channel,
  ts,
  seq,
  event,
  next_event,
  delta_hours
FROM im.v_event_seq;
"""

PINGPONG_CASES_SQL = """
SELECT
  case_id,
  variant,
  priority,
  issue_type,
  report_channel,
  cycle_hours,
  customer_satisfaction,
  escalation_count,
  resolver_changes,
  pingpong_transitions,
  has_reopen,
  has_reject,
  met_sla
FROM im.v_pingpong_cases;
"""

WORST_HANDOFF_SQL = """
SELECT
  case_id,
  variant,
  priority,
  issue_type,
  report_channel,
  cycle_hours,
  customer_satisfaction,
  met_sla,
  escalation_count,
  resolver_changes,
  has_reopen,
  has_reject,
  has_feedback
FROM im.v_worst_handoff_cases;
"""

CLOSURE_COMPLIANCE_SQL = """
SELECT
  case_id,
  variant,
  priority,
  issue_type,
  report_channel,
  cycle_hours,
  customer_satisfaction,
  met_sla,
  has_feedback,
  has_reopen,
  has_reject,
  closed_without_feedback,
  reopen_after_close_hours,
  reopened_within_3h,
  reopened_within_24h
FROM im.v_closure_compliance;
"""

PROBLEM_TOP_CASES_SQL = """
SELECT
  case_id,
  issue_type,
  priority,
  variant,
  report_channel,
  cycle_hours,
  customer_satisfaction,
  met_sla,
  has_reopen,
  has_reject,
  short_description_raw,
  short_description_norm
FROM im.v_problem_candidate_top_cases;
"""

FCR_CASES_ALL_SQL = """
SELECT
  case_id,
  issue_type,
  priority,
  variant,
  report_channel,
  cycle_hours,
  customer_satisfaction,
  met_sla,
  resolution_level,
  fcr
FROM im.v_fcr_cases;
"""


def fmt_int(value: object) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{int(value):,}"


def fmt_float(value: object, decimals: int = 2) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value):.{decimals}f}"


def fmt_pct(value: object, decimals: int = 1) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{float(value) * 100:.{decimals}f}%"


def normalize_neon_url(url: str) -> str:
    parsed = urlparse(url)
    query_params = parse_qsl(parsed.query, keep_blank_values=True)
    if not any(key == "options" for key, _ in query_params):
        return url
    filtered_params = [(key, value) for key, value in query_params if key != "options"]
    updated_query = urlencode(filtered_params, doseq=True)
    return urlunparse(parsed._replace(query=updated_query))


def get_db_url() -> Optional[str]:
    load_dotenv()

    db_url = os.getenv("DATABASE_URL_POOLED") or os.getenv("DATABASE_URL_DIRECT")
    if db_url:
        return normalize_neon_url(db_url)

    try:
        db_url = st.secrets["DATABASE_URL_POOLED"]
    except Exception:
        db_url = None
    if db_url:
        return normalize_neon_url(str(db_url))

    try:
        db_url = st.secrets["DATABASE_URL_DIRECT"]
    except Exception:
        db_url = None
    return normalize_neon_url(str(db_url)) if db_url else None


def connect_db(url: str):
    wait_seconds = [0.5, 1.0, 2.0, 4.0, 4.0]
    last_exception: Optional[Exception] = None
    for attempt, wait_time in enumerate(wait_seconds):
        try:
            return psycopg2.connect(
                url,
                connect_timeout=10,
                application_name="incidentops_pmi_hub",
            )
        except (psycopg2.OperationalError, psycopg2.InterfaceError) as exc:
            last_exception = exc
            if attempt == len(wait_seconds) - 1:
                break
            time.sleep(wait_time)
    if last_exception is not None:
        raise last_exception
    raise RuntimeError("Unable to connect to database.")


@st.cache_resource(show_spinner=False)
def get_connection(db_url: str):
    return connect_db(db_url)


@st.cache_data(ttl=300, show_spinner=False)
def _run_query_cached(
    db_url: str,
    sql: str,
    params: Optional[tuple[object, ...]] = None,
) -> pd.DataFrame:
    conn = get_connection(db_url)
    return pd.read_sql_query(sql, conn, params=params)


def run_query(sql: str, params: Optional[tuple[object, ...]] = None) -> pd.DataFrame:
    db_url = get_db_url()
    if not db_url:
        raise RuntimeError("Database URL is not configured.")
    try:
        return _run_query_cached(db_url, sql, params=params)
    except (psycopg2.OperationalError, psycopg2.InterfaceError):
        get_connection.clear()
        _run_query_cached.clear()
        conn = get_connection(db_url)
        return pd.read_sql_query(sql, conn, params=params)

def _prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    for column in ["created_at", "closed_at", "ts"]:
        if column in prepared.columns:
            prepared[column] = pd.to_datetime(prepared[column], errors="coerce")
    return prepared


def _fill_dimension(df: pd.DataFrame, column: str, fallback: str = "Unspecified") -> pd.DataFrame:
    if column not in df.columns:
        return df
    output = df.copy()
    output[column] = output[column].where(
        output[column].notna() & (output[column].astype(str).str.strip() != ""),
        fallback,
    )
    return output


def _apply_scope_filters(case_df: pd.DataFrame, filters: dict[str, object]) -> pd.DataFrame:
    if case_df.empty:
        return case_df

    scoped = case_df.copy()
    start_date, end_date = filters["date_range"]
    created_dates = pd.to_datetime(scoped["created_at"], errors="coerce").dt.date
    scoped = scoped.loc[(created_dates >= start_date) & (created_dates <= end_date)]

    mapping = {
        "priority": "priority",
        "issue_type": "issue_type",
        "variant": "variant",
        "channel": "report_channel",
    }
    for filter_key, column in mapping.items():
        selected = filters.get(filter_key) or []
        if selected and column in scoped.columns:
            scoped = _fill_dimension(scoped, column)
            scoped = scoped.loc[scoped[column].isin(selected)]

    return scoped.reset_index(drop=True)


def _filter_detail_frame(
    df: pd.DataFrame,
    filters: dict[str, object],
    *,
    case_ids: set[str] | None = None,
    date_column: str | None = None,
) -> pd.DataFrame:
    if df.empty:
        return df

    scoped = df.copy()
    if case_ids is not None and "case_id" in scoped.columns:
        scoped = scoped.loc[scoped["case_id"].isin(case_ids)]

    mapping = {
        "priority": "priority",
        "issue_type": "issue_type",
        "variant": "variant",
        "channel": "report_channel",
    }
    for filter_key, column in mapping.items():
        selected = filters.get(filter_key) or []
        if selected and column in scoped.columns:
            scoped = _fill_dimension(scoped, column)
            scoped = scoped.loc[scoped[column].isin(selected)]

    if date_column and date_column in scoped.columns:
        start_date, end_date = filters["date_range"]
        dates = pd.to_datetime(scoped[date_column], errors="coerce").dt.date
        scoped = scoped.loc[(dates >= start_date) & (dates <= end_date)]

    return scoped.reset_index(drop=True)


def _mean(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce")
    return float(numeric.mean()) if not numeric.dropna().empty else float("nan")


def _share(series: pd.Series) -> float:
    numeric = pd.to_numeric(series, errors="coerce")
    return float(numeric.mean()) if not numeric.dropna().empty else float("nan")


def _case_scope_badge(case_df: pd.DataFrame) -> str:
    return f"{len(case_df):,} cases in scope"


def _build_case_metrics(case_df: pd.DataFrame) -> list[dict[str, str]]:
    feedback_gap = 1.0 - _share(case_df["has_feedback"])
    return [
        {"label": "Cases", "value": fmt_int(len(case_df)), "meta": "Closed cases inside the current scope."},
        {"label": "SLA Met", "value": fmt_pct(_share(case_df["met_sla"])), "meta": "Share of cases closed within target."},
        {"label": "Avg Cycle", "value": fmt_float(_mean(case_df["cycle_hours"])), "meta": "Average closure time in hours."},
        {"label": "P90 Cycle", "value": fmt_float(case_df["cycle_hours"].quantile(0.9)), "meta": "Tail latency for the slowest 10% of cases."},
        {"label": "Reopen Risk", "value": fmt_pct(_share(case_df["has_reopen"])), "meta": "Cases that required another pass after closure."},
        {"label": "Feedback Gap", "value": fmt_pct(feedback_gap), "meta": "Cases closing without customer feedback."},
    ]


def _variant_rollup(case_df: pd.DataFrame) -> pd.DataFrame:
    if case_df.empty:
        return pd.DataFrame()
    scoped = _fill_dimension(case_df, "variant")
    grouped = (
        scoped.groupby("variant", dropna=False)
        .agg(
            cases=("case_id", "count"),
            avg_cycle_hours=("cycle_hours", "mean"),
            avg_csat=("customer_satisfaction", "mean"),
            met_sla_rate=("met_sla", "mean"),
            reopen_rate=("has_reopen", "mean"),
            reject_rate=("has_reject", "mean"),
            avg_escalations=("escalation_count", "mean"),
            avg_resolver_changes=("resolver_changes", "mean"),
        )
        .reset_index()
    )
    grouped["opportunity_score"] = grouped["cases"] * grouped["avg_cycle_hours"].fillna(0) * (
        1 + grouped["reopen_rate"].fillna(0) + grouped["reject_rate"].fillna(0)
    )
    return grouped.sort_values(["opportunity_score", "cases"], ascending=[False, False])


def _group_rollup(case_df: pd.DataFrame, group_column: str) -> pd.DataFrame:
    scoped = _fill_dimension(case_df, group_column)
    grouped = (
        scoped.groupby(group_column, dropna=False)
        .agg(
            cases=("case_id", "count"),
            avg_cycle_hours=("cycle_hours", "mean"),
            avg_csat=("customer_satisfaction", "mean"),
            met_sla_rate=("met_sla", "mean"),
            reopen_rate=("has_reopen", "mean"),
            reject_rate=("has_reject", "mean"),
            feedback_rate=("has_feedback", "mean"),
            avg_escalations=("escalation_count", "mean"),
            avg_resolver_changes=("resolver_changes", "mean"),
        )
        .reset_index()
    )
    return grouped.sort_values("cases", ascending=False)


def _workstream_sankey(transitions: pd.DataFrame, max_rows: int = 18):
    if transitions.empty:
        return None
    top = transitions.sort_values("transition_count", ascending=False).head(max_rows).copy()
    nodes = pd.unique(top[["from_event", "to_event"]].values.ravel("K")).tolist()
    node_ix = {name: idx for idx, name in enumerate(nodes)}

    fig = go.Figure(
        data=[
            go.Sankey(
                node=dict(
                    pad=18,
                    thickness=16,
                    label=nodes,
                    color="rgba(54,194,180,0.72)",
                    line=dict(color="rgba(125,164,185,0.28)", width=0.6),
                ),
                link=dict(
                    source=top["from_event"].map(node_ix).tolist(),
                    target=top["to_event"].map(node_ix).tolist(),
                    value=top["transition_count"].astype(float).tolist(),
                    customdata=top["avg_delta_hours"].round(2),
                    color="rgba(109,133,148,0.28)",
                    hovertemplate="From=%{source.label}<br>To=%{target.label}<br>Transitions=%{value:,.0f}<br>Avg Hours=%{customdata:.2f}<extra></extra>",
                ),
            )
        ]
    )
    return style_figure(fig, title="Dominant event paths", height=460)


def _top_cases(case_df: pd.DataFrame, limit: int = 25) -> pd.DataFrame:
    columns = [
        "case_id",
        "variant",
        "priority",
        "issue_type",
        "report_channel",
        "cycle_hours",
        "customer_satisfaction",
        "met_sla",
        "has_reopen",
        "has_reject",
        "resolver_changes",
        "escalation_count",
    ]
    return case_df.sort_values("cycle_hours", ascending=False)[columns].head(limit)


def _leader_briefing(case_df: pd.DataFrame, variant_df: pd.DataFrame) -> list[str]:
    if case_df.empty:
        return ["The current filter scope returns no closed cases. Widen the date range or remove one of the dimension filters."]

    worst_variant = variant_df.iloc[0]["variant"] if not variant_df.empty else "Unspecified"
    worst_variant_hours = variant_df.iloc[0]["avg_cycle_hours"] if not variant_df.empty else float("nan")
    feedback_gap = 1.0 - _share(case_df["has_feedback"])
    reopen_rate = _share(case_df["has_reopen"])
    return [
        f"The current scope covers {fmt_int(len(case_df))} closed cases with an average cycle time of {fmt_float(_mean(case_df['cycle_hours']))} hours.",
        f"{worst_variant} is the heaviest improvement target, averaging {fmt_float(worst_variant_hours)} hours with the highest combined burden score.",
        f"{fmt_pct(feedback_gap)} of closed cases still have no feedback, while reopen risk sits at {fmt_pct(reopen_rate)} and should be reviewed together with closure quality.",
    ]


def render_command_center(case_df: pd.DataFrame) -> None:
    page_masthead(
        "Command Center",
        "Executive-first operational health, pressure points, and immediate action areas.",
        "Balance service speed, closure quality, and intervention priority across the current scope.",
        badge=_case_scope_badge(case_df),
    )
    if case_df.empty:
        empty_state("No cases match the active filter scope.")
        return

    metric_strip(_build_case_metrics(case_df))
    variant_df = _variant_rollup(case_df)
    issue_df = _group_rollup(case_df, "issue_type")
    channel_df = _group_rollup(case_df, "report_channel")

    primary_left, primary_right = st.columns([1.3, 1.0])
    with primary_left:
        with st.container(border=True):
            section_header("Operational health", "Volume-weighted cycle burden by variant shows where service drag is concentrated.", eyebrow="Primary Story")
            chart_df = variant_df.head(8).sort_values("avg_cycle_hours", ascending=True)
            fig = px.bar(chart_df, x="avg_cycle_hours", y="variant", orientation="h", text="cases", color_discrete_sequence=[TOKENS["chart_primary"]])
            fig.update_traces(hovertemplate="Variant=%{y}<br>Avg Cycle=%{x:.2f} hrs<br>Cases=%{text}<extra></extra>", texttemplate="%{text:.0f}")
            fig.add_vline(x=_mean(case_df["cycle_hours"]), line_dash="dash", line_color=TOKENS["chart_warning"], annotation_text="Scope average", annotation_position="top left")
            if not chart_df.empty:
                annotate_chart(fig, f"Highest pressure: {chart_df.iloc[-1]['variant']}")
            st.plotly_chart(style_figure(fig, title="Cycle burden by variant", xtitle="Average cycle hours", ytitle="Variant", height=430), use_container_width=True)
    with primary_right:
        with st.container(border=True):
            section_header("Where to act", "Rank the top cohorts by combined case volume, delay, and quality risk.", eyebrow="Action Queue")
            top_actions = variant_df.head(10)[["variant", "cases", "avg_cycle_hours", "met_sla_rate", "reopen_rate", "reject_rate", "opportunity_score"]]
            summary_table(top_actions, percent_cols={"met_sla_rate", "reopen_rate", "reject_rate"}, height=430)

    secondary_left, secondary_right = st.columns(2)
    with secondary_left:
        with st.container(border=True):
            section_header("Issue mix", "Compare issue cohorts on cycle time and SLA achievement.", eyebrow="Secondary View")
            chart_df = issue_df.head(10).sort_values("avg_cycle_hours", ascending=True)
            fig = px.bar(chart_df, x="avg_cycle_hours", y="issue_type", orientation="h", color="met_sla_rate", color_continuous_scale=[TOKENS["chart_danger"], TOKENS["chart_primary"]])
            st.plotly_chart(style_figure(fig, title="Issue cohorts by cycle time", xtitle="Average cycle hours", ytitle="Issue type", height=400), use_container_width=True)
    with secondary_right:
        with st.container(border=True):
            section_header("Channel signal", "Cross-check channel performance before routing or staffing changes.", eyebrow="Secondary View")
            chart_df = channel_df.head(8)
            fig = px.scatter(chart_df, x="avg_cycle_hours", y="met_sla_rate", size="cases", color_discrete_sequence=[TOKENS["chart_warning"]], hover_name="report_channel", size_max=54)
            st.plotly_chart(style_figure(fig, title="Channel speed vs SLA delivery", xtitle="Average cycle hours", ytitle="SLA met rate", height=400), use_container_width=True)

    with st.container(border=True):
        section_header("Leader briefing", "Three short reads to focus the review discussion and next actions.", eyebrow="Briefing")
        narrative_callouts(_leader_briefing(case_df, variant_df))

    with st.container(border=True):
        section_header("Worklist", "Start investigation with the longest or most unstable cases inside the current scope.", eyebrow="Drilldown")
        worklist_table(_top_cases(case_df), height=420)

    with st.container(border=True):
        section_header("Evidence export", "Use the current scope to hand off evidence outside the dashboard.", eyebrow="Utility")
        export_frame("Export scoped cases", case_df, key="command_center_cases")

def render_flow(case_df: pd.DataFrame, filters: dict[str, object]) -> None:
    page_masthead(
        "Flow",
        "Process explorer and bottleneck surfacing across the incident lifecycle.",
        "Look for repeated waits, dominant transitions, and the flow variants that generate the most drag.",
        badge=_case_scope_badge(case_df),
    )
    if case_df.empty:
        empty_state("No cases match the active filter scope.")
        return

    event_df = _prepare_frame(run_query(EVENT_SCOPE_SQL))
    scoped_events = _filter_detail_frame(event_df, filters, case_ids=set(case_df["case_id"]), date_column="ts")
    if scoped_events.empty:
        empty_state("No event sequence rows match the active filter scope.")
        return

    transitions = (
        scoped_events.loc[scoped_events["next_event"].notna()]
        .groupby(["event", "next_event"], dropna=False)
        .agg(transition_count=("case_id", "size"), avg_delta_hours=("delta_hours", "mean"))
        .reset_index()
        .rename(columns={"event": "from_event", "next_event": "to_event"})
    )
    dwell = (
        scoped_events.groupby("event", dropna=False)
        .agg(occurrences=("case_id", "size"), avg_dwell_hours=("delta_hours", "mean"))
        .reset_index()
        .sort_values("avg_dwell_hours", ascending=False)
    )
    variant_df = _variant_rollup(case_df)
    avg_steps = scoped_events.groupby("case_id").size().mean()

    metric_strip([
        {"label": "Variants", "value": fmt_int(case_df["variant"].nunique()), "meta": "Distinct flow variants inside scope."},
        {"label": "Avg Steps", "value": fmt_float(avg_steps), "meta": "Average logged events per case."},
        {"label": "Transitions", "value": fmt_int(len(transitions)), "meta": "Observed handoffs between adjacent events."},
        {"label": "Slowest Dwell", "value": fmt_float(dwell["avg_dwell_hours"].max() if not dwell.empty else float("nan")), "meta": "Average hours spent at the slowest event."},
    ])

    hero_left, hero_right = st.columns([1.15, 1.0])
    with hero_left:
        with st.container(border=True):
            section_header("Bottleneck story", "Average dwell time by event shows where cases tend to slow down the most.", eyebrow="Primary Story")
            chart_df = dwell.head(10).sort_values("avg_dwell_hours", ascending=True)
            fig = px.bar(chart_df, x="avg_dwell_hours", y="event", orientation="h", text="occurrences", color_discrete_sequence=[TOKENS["chart_warning"]])
            if not chart_df.empty:
                annotate_chart(fig, f"Slowest stage: {chart_df.iloc[-1]['event']}")
            st.plotly_chart(style_figure(fig, title="Average dwell by event", xtitle="Average dwell hours", ytitle="Event", height=430), use_container_width=True)
    with hero_right:
        with st.container(border=True):
            section_header("Variant evidence", "The most common variants should be reviewed together with their cycle burden.", eyebrow="Primary Story")
            summary_table(variant_df.head(12)[["variant", "cases", "avg_cycle_hours", "met_sla_rate", "reopen_rate", "avg_resolver_changes"]], percent_cols={"met_sla_rate", "reopen_rate"}, height=430)

    lower_left, lower_right = st.columns(2)
    with lower_left:
        with st.container(border=True):
            section_header("Transition map", "Use the network view for context, but prioritize the ranked bottleneck table for action.", eyebrow="Secondary View")
            sankey = _workstream_sankey(transitions)
            if sankey is None:
                empty_state("No transition network is available for the current scope.")
            else:
                st.plotly_chart(sankey, use_container_width=True)
    with lower_right:
        with st.container(border=True):
            section_header("Transition hotspots", "Large and slow transitions deserve redesign before lower-volume flow noise.", eyebrow="Secondary View")
            transition_table = transitions.sort_values(["avg_delta_hours", "transition_count"], ascending=[False, False]).head(15)
            summary_table(transition_table, height=430)

    with st.container(border=True):
        section_header("Worklist", "Start with the cases that combine the most steps and the longest cycle time.", eyebrow="Drilldown")
        step_counts = scoped_events.groupby("case_id").size().rename("event_steps").reset_index()
        flow_cases = case_df.merge(step_counts, on="case_id", how="left").sort_values(["event_steps", "cycle_hours"], ascending=[False, False])
        worklist_table(flow_cases[["case_id", "variant", "priority", "issue_type", "report_channel", "event_steps", "cycle_hours", "met_sla", "has_reopen"]], height=420)

    with st.container(border=True):
        section_header("Evidence export", "Export the filtered transition set for deeper process analysis.", eyebrow="Utility")
        export_frame("Export scoped transitions", transitions, key="flow_transitions")


def render_handoffs(case_df: pd.DataFrame, filters: dict[str, object]) -> None:
    page_masthead(
        "Handoffs",
        "Escalation and ping-pong review with clear cohort comparison and case evidence.",
        "Focus on resolver churn, escalations, and the issue cohorts where handoffs become operationally expensive.",
        badge=_case_scope_badge(case_df),
    )
    if case_df.empty:
        empty_state("No cases match the active filter scope.")
        return

    pingpong_df = _prepare_frame(run_query(PINGPONG_CASES_SQL))
    worst_df = _prepare_frame(run_query(WORST_HANDOFF_SQL))
    scoped_pingpong = _filter_detail_frame(pingpong_df, filters, case_ids=set(case_df["case_id"]))
    scoped_worst = _filter_detail_frame(worst_df, filters, case_ids=set(case_df["case_id"]))
    pingpong_ids = set(scoped_pingpong["case_id"]) if "case_id" in scoped_pingpong.columns else set()

    handoff_df = (
        _fill_dimension(case_df, "issue_type")
        .groupby("issue_type", dropna=False)
        .agg(
            cases=("case_id", "count"),
            avg_cycle_hours=("cycle_hours", "mean"),
            avg_csat=("customer_satisfaction", "mean"),
            met_sla_rate=("met_sla", "mean"),
            avg_escalations=("escalation_count", "mean"),
            avg_resolver_changes=("resolver_changes", "mean"),
            handoff_rate=("resolver_changes", lambda s: (pd.to_numeric(s, errors="coerce") > 0).mean()),
            high_handoff_rate=("resolver_changes", lambda s: (pd.to_numeric(s, errors="coerce") >= 2).mean()),
        )
        .reset_index()
    )
    pingpong_map = (
        _fill_dimension(case_df.assign(_pingpong=case_df["case_id"].isin(pingpong_ids)), "issue_type")
        .groupby("issue_type")
        ["_pingpong"]
        .mean()
    )
    handoff_df["pingpong_rate"] = handoff_df["issue_type"].map(pingpong_map)
    handoff_df = handoff_df.sort_values("pingpong_rate", ascending=False)

    metric_strip([
        {"label": "Avg Resolver Changes", "value": fmt_float(_mean(case_df["resolver_changes"])), "meta": "Average ownership changes per case."},
        {"label": "Avg Escalations", "value": fmt_float(_mean(case_df["escalation_count"])), "meta": "Average escalations recorded per case."},
        {"label": "Ping-Pong Rate", "value": fmt_pct(len(scoped_pingpong) / len(case_df) if len(case_df) else float("nan")), "meta": "Cases flagged by repeated back-and-forth movement."},
        {"label": "Ping-Pong Cycle", "value": fmt_float(_mean(scoped_pingpong["cycle_hours"]) if not scoped_pingpong.empty else float("nan")), "meta": "Average cycle time for ping-pong cases."},
    ])

    hero_left, hero_right = st.columns([1.15, 1.0])
    with hero_left:
        with st.container(border=True):
            section_header("Cohort comparison", "Issue cohorts with high ping-pong and handoff rates deserve review before staffing changes.", eyebrow="Primary Story")
            chart_df = handoff_df.head(10).sort_values("pingpong_rate", ascending=True)
            fig = px.bar(chart_df, x="pingpong_rate", y="issue_type", orientation="h", text="cases", color_discrete_sequence=[TOKENS["chart_danger"]])
            st.plotly_chart(style_figure(fig, title="Ping-pong risk by issue type", xtitle="Ping-pong rate", ytitle="Issue type", height=430), use_container_width=True)
    with hero_right:
        with st.container(border=True):
            section_header("Handoff evidence", "Read handoff burden next to SLA and cycle performance before deciding on routing changes.", eyebrow="Primary Story")
            summary_table(handoff_df[["issue_type", "cases", "avg_cycle_hours", "met_sla_rate", "handoff_rate", "high_handoff_rate", "pingpong_rate"]].head(12), percent_cols={"met_sla_rate", "handoff_rate", "high_handoff_rate", "pingpong_rate"}, height=430)

    lower_left, lower_right = st.columns(2)
    with lower_left:
        with st.container(border=True):
            section_header("Escalation relationship", "Resolver churn and cycle time tend to rise together in the worst cohorts.", eyebrow="Secondary View")
            fig = px.scatter(handoff_df, x="avg_resolver_changes", y="avg_cycle_hours", size="cases", hover_name="issue_type", color_discrete_sequence=[TOKENS["chart_warning"]], size_max=52)
            st.plotly_chart(style_figure(fig, title="Resolver churn vs cycle time", xtitle="Average resolver changes", ytitle="Average cycle hours", height=400), use_container_width=True)
    with lower_right:
        with st.container(border=True):
            section_header("Action framing", "Start with ping-pong cases, then review the worst handoff cases for policy or routing fixes.", eyebrow="Secondary View")
            narrative_callouts([
                f"{fmt_pct(len(scoped_pingpong) / len(case_df) if len(case_df) else float('nan'))} of scoped cases are classified as ping-pong and should be reviewed as avoidable operational churn.",
                f"Average resolver changes across the scope sit at {fmt_float(_mean(case_df['resolver_changes']))}, with escalation volume averaging {fmt_float(_mean(case_df['escalation_count']))} per case.",
                "Use the worklist below to separate pure routing issues from true specialist escalation demand.",
            ])

    with st.container(border=True):
        section_header("Worklist", "Ping-pong cases and worst handoff cases are separated so follow-up can be action-oriented.", eyebrow="Drilldown")
        tab1, tab2 = st.tabs(["Ping-pong cases", "Worst handoff cases"])
        with tab1:
            worklist_table(scoped_pingpong.head(200), height=380)
        with tab2:
            worklist_table(scoped_worst.head(200), height=380)

    with st.container(border=True):
        section_header("Evidence export", "Keep the export surface separate from the review flow.", eyebrow="Utility")
        export_frame("Export ping-pong cases", scoped_pingpong, key="handoff_pingpong_cases")

def render_quality(case_df: pd.DataFrame, filters: dict[str, object]) -> None:
    page_masthead(
        "Quality",
        "Closure quality, customer feedback leakage, and reopen risk in one review surface.",
        "Treat feedback loss and fast reopens as quality outcomes, not just reporting gaps.",
        badge=_case_scope_badge(case_df),
    )
    if case_df.empty:
        empty_state("No cases match the active filter scope.")
        return

    closure_df = _prepare_frame(run_query(CLOSURE_COMPLIANCE_SQL))
    scoped_closure = _filter_detail_frame(closure_df, filters, case_ids=set(case_df["case_id"]))
    if scoped_closure.empty:
        empty_state("No closure-compliance rows match the active filter scope.")
        return

    issue_quality = (
        _fill_dimension(scoped_closure, "issue_type")
        .groupby("issue_type", dropna=False)
        .agg(
            cases=("case_id", "count"),
            avg_cycle_hours=("cycle_hours", "mean"),
            avg_csat=("customer_satisfaction", "mean"),
            met_sla_rate=("met_sla", "mean"),
            closed_without_feedback_rate=("closed_without_feedback", "mean"),
            reopened_within_24h_rate=("reopened_within_24h", "mean"),
            reject_rate=("has_reject", "mean"),
        )
        .reset_index()
        .sort_values("closed_without_feedback_rate", ascending=False)
    )
    priority_quality = (
        _fill_dimension(scoped_closure, "priority")
        .groupby("priority", dropna=False)
        .agg(
            cases=("case_id", "count"),
            reopened_within_24h_rate=("reopened_within_24h", "mean"),
            reject_rate=("has_reject", "mean"),
            avg_cycle_hours=("cycle_hours", "mean"),
        )
        .reset_index()
        .sort_values("reopened_within_24h_rate", ascending=False)
    )

    metric_strip([
        {"label": "Avg CSAT", "value": fmt_float(_mean(case_df["customer_satisfaction"])), "meta": "Average satisfaction across closed cases."},
        {"label": "SLA Met", "value": fmt_pct(_share(case_df["met_sla"])), "meta": "Cases that still meet target despite quality issues."},
        {"label": "No Feedback", "value": fmt_pct(_share(scoped_closure["closed_without_feedback"])), "meta": "Cases closed without customer feedback."},
        {"label": "Reopen <24h", "value": fmt_pct(_share(scoped_closure["reopened_within_24h"])), "meta": "Fast reopens after closure."},
        {"label": "Reject Rate", "value": fmt_pct(_share(scoped_closure["has_reject"])), "meta": "Cases with a reject event before closure."},
    ])

    hero_left, hero_right = st.columns([1.15, 1.0])
    with hero_left:
        with st.container(border=True):
            section_header("Feedback leakage", "Issue cohorts with weak feedback capture can hide closure quality problems.", eyebrow="Primary Story")
            chart_df = issue_quality.head(10).sort_values("closed_without_feedback_rate", ascending=True)
            fig = px.bar(chart_df, x="closed_without_feedback_rate", y="issue_type", orientation="h", text="cases", color_discrete_sequence=[TOKENS["chart_warning"]])
            st.plotly_chart(style_figure(fig, title="Closed without feedback by issue type", xtitle="Closed-without-feedback rate", ytitle="Issue type", height=430), use_container_width=True)
    with hero_right:
        with st.container(border=True):
            section_header("Quality evidence", "Keep feedback leakage, reopen risk, and cycle burden in the same view.", eyebrow="Primary Story")
            summary_table(issue_quality[["issue_type", "cases", "avg_cycle_hours", "avg_csat", "met_sla_rate", "closed_without_feedback_rate", "reopened_within_24h_rate", "reject_rate"]].head(12), percent_cols={"met_sla_rate", "closed_without_feedback_rate", "reopened_within_24h_rate", "reject_rate"}, height=430)

    lower_left, lower_right = st.columns(2)
    with lower_left:
        with st.container(border=True):
            section_header("Priority risk", "Fast reopens should be reviewed by priority to validate closure policy and follow-up speed.", eyebrow="Secondary View")
            fig = px.bar(priority_quality.sort_values("reopened_within_24h_rate", ascending=True), x="reopened_within_24h_rate", y="priority", orientation="h", color_discrete_sequence=[TOKENS["chart_danger"]])
            st.plotly_chart(style_figure(fig, title="Fast reopens by priority", xtitle="Reopened within 24 hours", ytitle="Priority", height=400), use_container_width=True)
    with lower_right:
        with st.container(border=True):
            section_header("Outcome relationship", "Cycle time and CSAT should be read together to avoid optimizing only for speed.", eyebrow="Secondary View")
            scatter_df = issue_quality.dropna(subset=["avg_cycle_hours", "avg_csat"])
            fig = px.scatter(scatter_df, x="avg_cycle_hours", y="avg_csat", size="cases", hover_name="issue_type", color_discrete_sequence=[TOKENS["chart_primary"]], size_max=52)
            st.plotly_chart(style_figure(fig, title="Cycle time vs CSAT by issue type", xtitle="Average cycle hours", ytitle="Average CSAT", height=400), use_container_width=True)

    with st.container(border=True):
        section_header("Worklist", "Review the worst closure outcomes before moving to broad policy changes.", eyebrow="Drilldown")
        worklist = scoped_closure.sort_values(["reopened_within_24h", "closed_without_feedback", "cycle_hours"], ascending=[False, False, False])
        worklist_table(worklist.head(200), height=420)

    with st.container(border=True):
        section_header("Evidence export", "Export the scoped closure-compliance cases for follow-up.", eyebrow="Utility")
        export_frame("Export quality worklist", scoped_closure, key="quality_cases")


def render_intake(case_df: pd.DataFrame) -> None:
    page_masthead(
        "Intake",
        "Channel performance framed as routing quality and avoidable intake friction.",
        "Compare intake channels on speed, SLA delivery, and the quality issues they introduce downstream.",
        badge=_case_scope_badge(case_df),
    )
    if case_df.empty:
        empty_state("No cases match the active filter scope.")
        return

    channel_df = _group_rollup(case_df, "report_channel")
    channel_df["missing_feedback_rate"] = 1.0 - channel_df["feedback_rate"]
    channel_issue_df = (
        _fill_dimension(_fill_dimension(case_df, "report_channel"), "issue_type")
        .groupby(["report_channel", "issue_type"], dropna=False)
        .agg(
            cases=("case_id", "count"),
            avg_cycle_hours=("cycle_hours", "mean"),
            met_sla_rate=("met_sla", "mean"),
            reopen_rate=("has_reopen", "mean"),
        )
        .reset_index()
        .sort_values(["report_channel", "cases"], ascending=[True, False])
    )

    metric_strip([
        {"label": "Channels", "value": fmt_int(case_df["report_channel"].nunique()), "meta": "Distinct intake channels in the current scope."},
        {"label": "Avg Cycle", "value": fmt_float(_mean(case_df["cycle_hours"])), "meta": "Overall cycle benchmark for intake comparison."},
        {"label": "SLA Met", "value": fmt_pct(_share(case_df["met_sla"])), "meta": "Scope-level baseline for channel comparisons."},
        {"label": "Missing Feedback", "value": fmt_pct(1.0 - _share(case_df["has_feedback"])), "meta": "Potential feedback leakage by intake path."},
    ])

    hero_left, hero_right = st.columns([1.1, 1.0])
    with hero_left:
        with st.container(border=True):
            section_header("Channel performance", "Review routing quality before optimizing for channel volume alone.", eyebrow="Primary Story")
            chart_df = channel_df.sort_values("avg_cycle_hours", ascending=True)
            fig = px.bar(chart_df, x="avg_cycle_hours", y="report_channel", orientation="h", text="cases", color="missing_feedback_rate", color_continuous_scale=[TOKENS["chart_primary"], TOKENS["chart_warning"], TOKENS["chart_danger"]])
            st.plotly_chart(style_figure(fig, title="Average cycle time by channel", xtitle="Average cycle hours", ytitle="Report channel", height=430), use_container_width=True)
    with hero_right:
        with st.container(border=True):
            section_header("Routing evidence", "Keep speed, SLA, and feedback loss together in the ranked summary.", eyebrow="Primary Story")
            summary_table(channel_df[["report_channel", "cases", "avg_cycle_hours", "met_sla_rate", "reopen_rate", "missing_feedback_rate", "avg_escalations", "avg_resolver_changes"]].head(12), percent_cols={"met_sla_rate", "reopen_rate", "missing_feedback_rate"}, height=430)

    lower_left, lower_right = st.columns(2)
    with lower_left:
        with st.container(border=True):
            section_header("Issue mix by channel", "Look for channels that repeatedly attract the same expensive or unstable issues.", eyebrow="Secondary View")
            top_issue_mix = channel_issue_df.sort_values("cases", ascending=False).head(20)
            fig = px.bar(top_issue_mix, x="cases", y="issue_type", color="report_channel", orientation="h")
            st.plotly_chart(style_figure(fig, title="Top issue/channel combinations", xtitle="Cases", ytitle="Issue type", height=400), use_container_width=True)
    with lower_right:
        with st.container(border=True):
            section_header("Intake actions", "Use channel evidence to fix routing, form quality, or workforce alignment.", eyebrow="Secondary View")
            narrative_callouts([
                f"The slowest channel in scope is {channel_df.iloc[0]['report_channel'] if not channel_df.empty else 'n/a'}, averaging {fmt_float(channel_df.iloc[0]['avg_cycle_hours'] if not channel_df.empty else float('nan'))} hours.",
                "Review issue/channel combinations with both high volume and weak SLA delivery before changing channel mix goals.",
                "Pair this page with Handoffs to see whether intake friction is turning into resolver churn downstream.",
            ])

    with st.container(border=True):
        section_header("Worklist", "Investigate intake cases that combine long cycle time with visible downstream instability.", eyebrow="Drilldown")
        worklist_table(_top_cases(case_df), height=420)

    with st.container(border=True):
        section_header("Evidence export", "Export scoped intake cases for routing and intake-quality analysis.", eyebrow="Utility")
        export_frame("Export intake cases", case_df, key="intake_cases")


def render_enablement(case_df: pd.DataFrame, filters: dict[str, object]) -> None:
    page_masthead(
        "Enablement",
        "FCR, knowledge opportunities, and prevention backlog in one improvement-program surface.",
        "Separate top opportunities, supporting evidence, and next actions so enablement work can be prioritized like a real program.",
        badge=_case_scope_badge(case_df),
    )
    if case_df.empty:
        empty_state("No cases match the active filter scope.")
        return

    fcr_df = _prepare_frame(run_query(FCR_CASES_ALL_SQL))
    problem_df = _prepare_frame(run_query(PROBLEM_TOP_CASES_SQL))
    scoped_fcr = _filter_detail_frame(fcr_df, filters, case_ids=set(case_df["case_id"]))
    scoped_problem = _filter_detail_frame(problem_df, filters, case_ids=set(case_df["case_id"]))
    if scoped_fcr.empty:
        empty_state("No FCR cases match the active filter scope.")
        return

    issue_fcr = (
        _fill_dimension(scoped_fcr, "issue_type")
        .groupby("issue_type", dropna=False)
        .agg(
            cases=("case_id", "count"),
            fcr_rate=("fcr", "mean"),
            avg_cycle_hours=("cycle_hours", "mean"),
            avg_csat=("customer_satisfaction", "mean"),
            met_sla_rate=("met_sla", "mean"),
            l1_rate=("resolution_level", lambda s: (s == "L1").mean()),
            l2_rate=("resolution_level", lambda s: (s == "L2").mean()),
            l3_rate=("resolution_level", lambda s: (s == "L3").mean()),
        )
        .reset_index()
    )
    issue_fcr["enablement_score"] = issue_fcr["cases"] * issue_fcr["avg_cycle_hours"].fillna(0) * (1 - issue_fcr["fcr_rate"].fillna(0))
    issue_fcr = issue_fcr.sort_values("enablement_score", ascending=False)

    backlog_df = pd.DataFrame()
    if not scoped_problem.empty:
        backlog_df = (
            _fill_dimension(scoped_problem, "issue_type")
            .groupby(["issue_type", "short_description_norm"], dropna=False)
            .agg(cases=("case_id", "count"), avg_cycle_hours=("cycle_hours", "mean"))
            .reset_index()
            .sort_values(["cases", "avg_cycle_hours"], ascending=[False, False])
        )

    metric_strip([
        {"label": "FCR Rate", "value": fmt_pct(_share(scoped_fcr["fcr"])), "meta": "Share of cases resolved in first contact."},
        {"label": "L1 Resolved", "value": fmt_pct((scoped_fcr["resolution_level"] == "L1").mean()), "meta": "Cases resolved at L1 before escalation."},
        {"label": "Avg Cycle", "value": fmt_float(_mean(scoped_fcr["cycle_hours"])), "meta": "Average cycle time for FCR-scoped cases."},
        {"label": "Avg CSAT", "value": fmt_float(_mean(scoped_fcr["customer_satisfaction"])), "meta": "Satisfaction readout for enablement review."},
    ])

    hero_left, hero_right = st.columns([1.1, 1.0])
    with hero_left:
        with st.container(border=True):
            section_header("Top opportunities", "Issue types with low FCR and high cycle burden should drive knowledge or training investment.", eyebrow="Primary Story")
            chart_df = issue_fcr.head(10).sort_values("enablement_score", ascending=True)
            fig = px.bar(chart_df, x="enablement_score", y="issue_type", orientation="h", text="cases", color_discrete_sequence=[TOKENS["chart_primary"]])
            st.plotly_chart(style_figure(fig, title="Knowledge and training opportunity score", xtitle="Enablement score", ytitle="Issue type", height=430), use_container_width=True)
    with hero_right:
        with st.container(border=True):
            section_header("Evidence table", "Read FCR, cycle time, and resolution-level distribution together.", eyebrow="Primary Story")
            summary_table(issue_fcr[["issue_type", "cases", "fcr_rate", "avg_cycle_hours", "avg_csat", "met_sla_rate", "l1_rate", "l2_rate", "l3_rate"]].head(12), percent_cols={"fcr_rate", "met_sla_rate", "l1_rate", "l2_rate", "l3_rate"}, height=430)

    lower_left, lower_right = st.columns(2)
    with lower_left:
        with st.container(border=True):
            section_header("Prevention backlog", "Problem patterns help separate knowledge enablement from deeper defect or process work.", eyebrow="Secondary View")
            if backlog_df.empty:
                empty_state("No scoped prevention backlog rows are available.")
            else:
                summary_table(backlog_df.head(15), height=400)
    with lower_right:
        with st.container(border=True):
            section_header("Recommended next actions", "Keep actions explicit so this page can drive a real improvement program.", eyebrow="Secondary View")
            top_issue = issue_fcr.iloc[0]["issue_type"] if not issue_fcr.empty else "n/a"
            narrative_callouts([
                f"Start with {top_issue}: it carries the highest combined enablement burden in the current scope.",
                "Use low-FCR, high-L2/L3 cohorts to prioritize KB articles, playbooks, and L1 training updates.",
                "Review the prevention backlog separately from enablement items so product or problem management work does not get lost inside support operations.",
            ])

    with st.container(border=True):
        section_header("Worklist", "Keep analyst drilldown separate from the ranked opportunity view.", eyebrow="Drilldown")
        tab1, tab2 = st.tabs(["Low-FCR cases", "Problem backlog cases"])
        with tab1:
            worklist_table(scoped_fcr.sort_values(["fcr", "cycle_hours"], ascending=[True, False]).head(200), height=380)
        with tab2:
            if scoped_problem.empty:
                empty_state("No scoped problem candidate rows are available.")
            else:
                worklist_table(scoped_problem.sort_values("cycle_hours", ascending=False).head(200), height=380)

    with st.container(border=True):
        section_header("Evidence export", "Exports are available without interrupting the review sequence.", eyebrow="Utility")
        export_frame("Export FCR cases", scoped_fcr, key="enablement_fcr_cases")


def _navigation_choice() -> str:
    with st.sidebar:
        sidebar_brand()
        st.markdown("##### Workstreams")
        return st.radio(
            "Navigation",
            ["Command Center", "Flow", "Handoffs", "Quality", "Intake", "Enablement"],
            label_visibility="collapsed",
        )


def main() -> None:
    apply_theme()

    db_url = get_db_url()
    if not db_url:
        st.warning("Database config missing. Set DATABASE_URL_POOLED (preferred) or DATABASE_URL_DIRECT in .env or Streamlit secrets.")
        st.stop()

    try:
        case_scope = _prepare_frame(run_query(CASE_SCOPE_SQL))
    except Exception as exc:
        st.error(f"Failed to query case scope: {exc}")
        st.stop()

    if case_scope.empty:
        st.warning("No data returned from im.v_case_sla.")
        st.stop()

    min_date = pd.to_datetime(case_scope["created_at"]).dt.date.min()
    max_date = pd.to_datetime(case_scope["created_at"]).dt.date.max()
    priorities = sorted(_fill_dimension(case_scope, "priority")["priority"].dropna().unique().tolist())
    issue_types = sorted(_fill_dimension(case_scope, "issue_type")["issue_type"].dropna().unique().tolist())
    variants = sorted(_fill_dimension(case_scope, "variant")["variant"].dropna().unique().tolist())
    channels = sorted(_fill_dimension(case_scope, "report_channel")["report_channel"].dropna().unique().tolist())

    page = _navigation_choice()
    filters = global_filter_bar(
        min_date=min_date,
        max_date=max_date,
        priorities=priorities,
        issue_types=issue_types,
        variants=variants,
        channels=channels,
    )
    scoped_cases = _apply_scope_filters(case_scope, filters)

    if page == "Command Center":
        render_command_center(scoped_cases)
    elif page == "Flow":
        render_flow(scoped_cases, filters)
    elif page == "Handoffs":
        render_handoffs(scoped_cases, filters)
    elif page == "Quality":
        render_quality(scoped_cases, filters)
    elif page == "Intake":
        render_intake(scoped_cases)
    elif page == "Enablement":
        render_enablement(scoped_cases, filters)


if __name__ == "__main__":
    main()
