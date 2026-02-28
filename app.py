from __future__ import annotations

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

P_DARK_BLUE = "#1E5C7A"
P_MID_BLUE = "#2E7AA3"
P_TEXT_DARK = "#FAFAFA"
P_TEXT_MED = "rgba(255,255,255,0.78)"
COLOR_TEAL = "#4DB6AC"
COLOR_ORANGE = "#FF7043"

ST_BG = "#0E1117"
ST_TEXT = P_TEXT_DARK
ACCENT = COLOR_TEAL
CARD_BG = "#262730"


st.set_page_config(
    page_title="IncidentOps | PMI Hub",
    layout="wide",
    initial_sidebar_state="expanded",
)

KPI_SQL = """
SELECT
  count(*) AS cases,
  avg(cycle_hours) AS avg_cycle_hours,
  percentile_cont(0.9) WITHIN GROUP (ORDER BY cycle_hours) AS p90_cycle_hours,
  avg(customer_satisfaction) AS avg_csat,
  avg(CASE WHEN met_sla THEN 1 ELSE 0 END) AS met_sla_rate,
  avg(CASE WHEN has_reopen THEN 1 ELSE 0 END) AS reopen_rate,
  avg(CASE WHEN has_reject THEN 1 ELSE 0 END) AS reject_rate,
  avg(escalation_count) AS avg_escalations,
  avg(resolver_changes) AS avg_resolver_changes,
  avg(CASE WHEN has_feedback THEN 1 ELSE 0 END) AS feedback_rate
FROM im.v_case_sla;
"""

VARIANT_SQL = """
SELECT
  variant,
  cases,
  avg_cycle_hours,
  p90_cycle_hours,
  avg_csat,
  met_sla_rate,
  reopen_rate,
  reject_rate,
  avg_escalations,
  avg_resolver_changes
FROM im.v_variant_summary
ORDER BY cases DESC;
"""

RANGE_SQL = """
SELECT min(created_at) AS min_created, max(closed_at) AS max_closed
FROM im.v_case;
"""

TRANSITION_SUMMARY_SQL = """
SELECT
  from_event,
  to_event,
  transition_count,
  avg_delta_hours,
  median_delta_hours,
  p90_delta_hours
FROM im.v_transition_summary
ORDER BY transition_count DESC;
"""

TRANSITION_BY_VARIANT_SQL = """
SELECT
  variant,
  from_event,
  to_event,
  transition_count,
  avg_delta_hours
FROM im.v_transition_by_variant;
"""

DWELL_SQL = """
SELECT
  event,
  occurrences,
  avg_dwell_hours,
  median_dwell_hours,
  p90_dwell_hours
FROM im.v_dwell_by_event
ORDER BY avg_dwell_hours DESC;
"""

HANDOFF_SUMMARY_SQL = """
SELECT
  variant,
  issue_type,
  cases,
  avg_cycle_hours,
  p90_cycle_hours,
  avg_csat,
  met_sla_rate,
  avg_escalations,
  avg_resolver_changes,
  handoff_rate,
  high_handoff_rate,
  pingpong_rate
FROM im.v_handoff_summary;
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

PINGPONG_KPI_SQL = """
SELECT
  count(*) AS pingpong_cases,
  avg(cycle_hours) AS avg_cycle_hours_pingpong,
  avg(customer_satisfaction) AS avg_csat_pingpong,
  avg(CASE WHEN met_sla THEN 1 ELSE 0 END) AS met_sla_rate_pingpong,
  avg(resolver_changes) AS avg_resolver_changes_pingpong,
  avg(escalation_count) AS avg_escalations_pingpong
FROM im.v_pingpong_cases;
"""

OVERALL_HANDOFF_BASELINE_SQL = """
SELECT
  count(*) AS overall_cases,
  avg(cycle_hours) AS avg_cycle_hours_overall,
  avg(customer_satisfaction) AS avg_csat_overall,
  avg(CASE WHEN met_sla THEN 1 ELSE 0 END) AS met_sla_rate_overall,
  avg(resolver_changes) AS avg_resolver_changes_overall,
  avg(escalation_count) AS avg_escalations_overall
FROM im.v_case_sla;
"""

CX_SUMMARY_SQL = """
SELECT
  cases,
  avg_csat,
  met_sla_rate,
  closed_without_feedback_rate,
  reopened_within_3h_rate,
  reopened_within_24h_rate,
  reject_rate,
  avg_cycle_hours,
  p90_cycle_hours
FROM im.v_cx_summary;
"""

CX_BREAKDOWN_SQL = """
SELECT
  issue_type,
  priority,
  cases,
  avg_csat,
  met_sla_rate,
  closed_without_feedback_rate,
  reopened_within_3h_rate,
  reject_rate,
  avg_cycle_hours
FROM im.v_cx_breakdown;
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

PROBLEM_CANDIDATES_SQL = """
SELECT
  issue_type,
  short_description_norm,
  cases,
  avg_cycle_hours,
  p90_cycle_hours,
  avg_csat,
  met_sla_rate,
  reopen_rate,
  reject_rate,
  impact_score,
  example_description,
  last_seen_ts
FROM im.v_problem_candidates;
"""

PROBLEM_CASES_BY_CANDIDATE_SQL = """
SELECT
  case_id,
  priority,
  variant,
  report_channel,
  cycle_hours,
  customer_satisfaction,
  met_sla,
  has_reopen,
  has_reject,
  has_feedback,
  short_description_raw
FROM im.v_problem_candidate_cases
WHERE issue_type = %s
  AND short_description_norm = %s
ORDER BY cycle_hours DESC
LIMIT 200;
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

CHANNEL_SUMMARY_SQL = """
SELECT
  report_channel,
  cases,
  avg_cycle_hours,
  p90_cycle_hours,
  avg_csat,
  met_sla_rate,
  reopen_rate,
  reject_rate,
  avg_escalations,
  avg_resolver_changes,
  feedback_rate,
  missing_feedback_rate
FROM im.v_channel_summary;
"""

CHANNEL_ISSUE_SUMMARY_SQL = """
SELECT
  report_channel,
  issue_type,
  cases,
  avg_cycle_hours,
  p90_cycle_hours,
  avg_csat,
  met_sla_rate,
  reopen_rate,
  reject_rate,
  avg_escalations,
  avg_resolver_changes,
  feedback_rate,
  missing_feedback_rate
FROM im.v_channel_issue_summary;
"""

FCR_OVERVIEW_SQL = """
SELECT
  COUNT(*) AS cases,
  AVG(CASE WHEN fcr THEN 1 ELSE 0 END) AS fcr_rate,
  AVG(cycle_hours) AS avg_cycle_hours,
  AVG(
    CASE
      WHEN met_sla IS TRUE THEN 1.0
      WHEN met_sla IS FALSE THEN 0.0
      ELSE NULL
    END
  ) AS met_sla_rate,
  AVG(customer_satisfaction) AS avg_csat,
  AVG(CASE WHEN resolution_level = 'L1' THEN 1.0 ELSE 0.0 END) AS l1_rate,
  AVG(CASE WHEN resolution_level = 'L2' THEN 1.0 ELSE 0.0 END) AS l2_rate,
  AVG(CASE WHEN resolution_level = 'L3' THEN 1.0 ELSE 0.0 END) AS l3_rate
FROM im.v_fcr_cases;
"""

FCR_LEVEL_DIST_SQL = """
SELECT
  resolution_level,
  COUNT(*) AS cases
FROM im.v_fcr_cases
GROUP BY resolution_level;
"""

FCR_SUMMARY_SQL = """
SELECT
  issue_type,
  cases,
  fcr_rate,
  avg_cycle_hours,
  avg_csat,
  met_sla_rate,
  reopen_rate,
  l2_escalation_rate,
  l3_escalation_rate
FROM im.v_fcr_summary;
"""

KB_ENABLEMENT_SQL = """
SELECT
  issue_type,
  cases,
  l1_solved_rate,
  l2_or_l3_rate,
  escalation_to_l2_rate,
  reopen_rate,
  avg_cycle_hours,
  avg_csat,
  enablement_score
FROM im.v_kb_enablement_candidates;
"""

FCR_CASES_BY_ISSUE_SQL = """
SELECT
  case_id,
  priority,
  variant,
  report_channel,
  cycle_hours,
  customer_satisfaction,
  met_sla,
  resolution_level,
  fcr
FROM im.v_fcr_cases
WHERE issue_type = %s;
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


def header(title: str, subtitle: str) -> None:
    st.markdown(
        f"""
        <div class="page-header">
          <h1>{title}</h1>
          <p>{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_tile(
    label: str,
    value: str,
    sublabel: str = "",
    bg: str = CARD_BG,
    accent: str = ACCENT,
) -> None:
    st.markdown(
        f"""
        <div class="io-card" style="background:{bg}; border-left-color:{accent};">
          <div class="io-card-label">{label}</div>
          <div class="io-card-value">{value}</div>
          <div class="io-card-sub">{sublabel}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def ui_explainer(title: str, what: list[str], why: list[str], how: list[str]) -> None:
    with st.expander(title, expanded=False):
        if what:
            st.markdown("**What**")
            st.markdown("\n".join([f"- {item}" for item in what]))
        if why:
            st.markdown("**Why**")
            st.markdown("\n".join([f"- {item}" for item in why]))
        if how:
            st.markdown("**How**")
            st.markdown("\n".join([f"- {item}" for item in how]))


def style_table(
    df: pd.DataFrame,
    pct_cols: set[str] | None = None,
    hour_cols: set[str] | None = None,
    good_high_cols: set[str] | None = None,
    good_low_cols: set[str] | None = None,
    bad_high_cols: set[str] | None = None,
    bad_low_cols: set[str] | None = None,
):
    if df.empty:
        return df

    pct_cols = pct_cols or set()
    hour_cols = hour_cols or set()
    good_high_cols = good_high_cols or set()
    good_low_cols = good_low_cols or set()
    bad_high_cols = bad_high_cols or set()
    bad_low_cols = bad_low_cols or set()

    formatters: dict[str, object] = {}
    existing_pct_cols = [col for col in pct_cols if col in df.columns]
    existing_hour_cols = [col for col in hour_cols if col in df.columns]

    for col in existing_pct_cols:
        formatters[col] = lambda v: "n/a" if pd.isna(v) else f"{float(v) * 100:.1f}%"
    for col in existing_hour_cols:
        formatters[col] = lambda v: "n/a" if pd.isna(v) else f"{float(v):.2f}"

    styled = df.style.format(formatters, na_rep="n/a")

    high_subset = [
        col for col in good_high_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
    ]
    if high_subset:
        styled = styled.background_gradient(cmap="Greens", subset=high_subset)

    low_subset = [
        col for col in good_low_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
    ]
    if low_subset:
        styled = styled.background_gradient(cmap="Greens_r", subset=low_subset)

    bad_high_subset = [
        col for col in bad_high_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
    ]
    if bad_high_subset:
        styled = styled.background_gradient(cmap="Reds", subset=bad_high_subset)

    bad_low_subset = [
        col for col in bad_low_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])
    ]
    if bad_low_subset:
        styled = styled.background_gradient(cmap="Reds_r", subset=bad_low_subset)

    return styled


def clayout(
    fig,
    title: str = "",
    xtitle: str = "",
    ytitle: str = "",
    h: int = 440,
):
    fig.update_layout(
        title=title,
        height=h,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=ST_TEXT, size=13),
        legend=dict(
            bgcolor="rgba(0,0,0,0)",
            bordercolor="rgba(0,0,0,0)",
            title=dict(font=dict(color=ST_TEXT)),
            font=dict(color=ST_TEXT),
        ),
        margin=dict(l=20, r=20, t=60, b=20),
    )
    fig.update_xaxes(
        title=xtitle,
        gridcolor="rgba(250,250,250,0.08)",
        zeroline=False,
        linecolor="rgba(250,250,250,0.25)",
        tickfont=dict(color=ST_TEXT),
        title_font=dict(color=ST_TEXT),
    )
    fig.update_yaxes(
        title=ytitle,
        gridcolor="rgba(250,250,250,0.10)",
        zeroline=False,
        linecolor="rgba(250,250,250,0.25)",
        tickfont=dict(color=ST_TEXT),
        title_font=dict(color=ST_TEXT),
    )
    return fig


def build_sankey(transitions: pd.DataFrame, title: str, max_rows: int, height: int = 520):
    if transitions.empty:
        return None

    use_cols = ["from_event", "to_event", "transition_count", "avg_delta_hours"]
    missing_cols = [c for c in use_cols if c not in transitions.columns]
    if missing_cols:
        return None

    sankey_df = transitions[use_cols].copy()
    sankey_df = sankey_df.dropna(subset=["from_event", "to_event"])
    sankey_df["transition_count"] = pd.to_numeric(sankey_df["transition_count"], errors="coerce")
    sankey_df["avg_delta_hours"] = pd.to_numeric(sankey_df["avg_delta_hours"], errors="coerce")
    sankey_df = sankey_df[sankey_df["transition_count"] > 0]
    sankey_df = sankey_df.sort_values("transition_count", ascending=False).head(max_rows)
    if sankey_df.empty:
        return None

    nodes = pd.unique(sankey_df[["from_event", "to_event"]].values.ravel("K")).tolist()
    node_ix = {name: idx for idx, name in enumerate(nodes)}

    sources = sankey_df["from_event"].map(node_ix).tolist()
    targets = sankey_df["to_event"].map(node_ix).tolist()
    values = sankey_df["transition_count"].astype(float).tolist()
    hover_hours = sankey_df["avg_delta_hours"].tolist()

    fig = go.Figure(
        data=[
            go.Sankey(
                arrangement="snap",
                node=dict(
                    pad=18,
                    thickness=16,
                    line=dict(color="rgba(250,250,250,0.25)", width=0.8),
                    label=nodes,
                    color="rgba(77,182,172,0.75)",
                    hovertemplate="%{label}<extra></extra>",
                ),
                link=dict(
                    source=sources,
                    target=targets,
                    value=values,
                    customdata=hover_hours,
                    color="rgba(127,224,214,0.38)",
                    hovertemplate=(
                        "From=%{source.label}<br>To=%{target.label}<br>"
                        "Transitions=%{value:,.0f}<br>Avg Hours=%{customdata:.2f}<extra></extra>"
                    ),
                ),
            )
        ]
    )
    fig.update_layout(
        title=title,
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=ST_TEXT, size=13),
        margin=dict(l=10, r=10, t=55, b=10),
    )
    return fig


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

    db_url = os.getenv("DATABASE_URL_POOLED")
    if not db_url:
        db_url = os.getenv("DATABASE_URL_DIRECT")
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
    db_url: str, sql: str, params: Optional[tuple[object, ...]] = None
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


def render_coming_soon(page_name: str) -> None:
    header(page_name, "This module is planned next.")
    st.info("Coming soon. Executive Overview is currently active.")


def render_executive_overview(db_url: str) -> None:
    header("Executive Overview", "Cross-case operational KPIs and variant performance.")

    try:
        kpi_df = run_query(KPI_SQL)
        variant_df = run_query(VARIANT_SQL)
        range_df = run_query(RANGE_SQL)
    except Exception as exc:
        st.error(f"Failed to query dashboard views: {exc}")
        st.stop()

    if kpi_df.empty:
        st.warning("No data returned from im.v_case_sla.")
        st.stop()

    k = kpi_df.iloc[0]
    missing_feedback_rate = (
        1.0 - float(k["feedback_rate"]) if pd.notna(k["feedback_rate"]) else float("nan")
    )

    row1 = st.columns(4)
    row2 = st.columns(4)

    with row1[0]:
        metric_tile("Cases", fmt_int(k["cases"]), "Closed cases in scope")
    with row1[1]:
        metric_tile("Avg Cycle (hrs)", fmt_float(k["avg_cycle_hours"]), "Mean time to close")
    with row1[2]:
        metric_tile("P90 Cycle (hrs)", fmt_float(k["p90_cycle_hours"]), "90th percentile cycle time")
    with row1[3]:
        metric_tile("Avg CSAT", fmt_float(k["avg_csat"]), "Customer satisfaction score")

    with row2[0]:
        metric_tile("SLA Met %", fmt_pct(k["met_sla_rate"]), "Cases meeting SLA target")
    with row2[1]:
        metric_tile("Reopen %", fmt_pct(k["reopen_rate"]), "Cases with reopen events")
    with row2[2]:
        metric_tile("Reject %", fmt_pct(k["reject_rate"]), "Cases with reject events")
    with row2[3]:
        metric_tile("Missing Feedback %", fmt_pct(missing_feedback_rate), "1 - feedback rate")

    st.markdown("#### Variant Leaderboard")
    if variant_df.empty:
        st.info("No rows in im.v_variant_summary yet.")
    else:
        table_df = variant_df.sort_values("cases", ascending=False).copy()
        table_df["cases"] = table_df["cases"].astype("Int64")

        for col in ["avg_cycle_hours", "p90_cycle_hours", "avg_csat", "avg_escalations", "avg_resolver_changes"]:
            table_df[col] = table_df[col].map(lambda v: fmt_float(v, 2))

        for col in ["met_sla_rate", "reopen_rate", "reject_rate"]:
            table_df[col] = table_df[col].map(lambda v: fmt_pct(v, 1))

        st.dataframe(table_df, use_container_width=True, hide_index=True)

    st.markdown("#### Variant Charts")
    if not variant_df.empty:
        plot_df = variant_df.copy()

        fig_cases = px.bar(
            plot_df,
            x="variant",
            y="cases",
            color="variant",
            text_auto=True,
        )
        fig_cases.update_traces(hovertemplate="Variant=%{x}<br>Cases=%{y}<extra></extra>")
        st.plotly_chart(
            clayout(fig_cases, title="Cases by Variant", xtitle="Variant", ytitle="Cases"),
            use_container_width=True,
        )

        fig_cycle = px.bar(
            plot_df,
            x="variant",
            y="avg_cycle_hours",
            color="variant",
            text_auto=".2f",
        )
        fig_cycle.update_traces(
            hovertemplate="Variant=%{x}<br>Avg Cycle=%{y:.2f} hrs<extra></extra>",
        )
        st.plotly_chart(
            clayout(fig_cycle, title="Average Cycle Hours by Variant", xtitle="Variant", ytitle="Avg Cycle (hrs)"),
            use_container_width=True,
        )

        fig_scatter = px.scatter(
            plot_df,
            x="avg_cycle_hours",
            y="avg_csat",
            size="cases",
            color="variant",
            hover_name="variant",
            size_max=56,
        )
        fig_scatter.update_traces(
            marker=dict(line=dict(width=1, color="rgba(250,250,250,0.35)"), opacity=0.84),
            hovertemplate="Variant=%{hovertext}<br>Avg Cycle=%{x:.2f} hrs<br>Avg CSAT=%{y:.2f}<extra></extra>",
        )
        st.plotly_chart(
            clayout(
                fig_scatter,
                title="Cycle Time vs CSAT by Variant",
                xtitle="Avg Cycle (hrs)",
                ytitle="Avg CSAT",
            ),
            use_container_width=True,
        )

    if not range_df.empty:
        min_created = range_df.iloc[0]["min_created"]
        max_closed = range_df.iloc[0]["max_closed"]
        if pd.notna(min_created) and pd.notna(max_closed):
            st.caption(
                f"Data range: created {pd.to_datetime(min_created):%Y-%m-%d} to closed {pd.to_datetime(max_closed):%Y-%m-%d}"
            )

    with st.expander("Data Notes"):
        st.write("For process improvement, not individual performance evaluation.")


def render_process_explorer(db_url: str) -> None:
    header("Process Explorer", "Variant and transition flow analytics across the incident lifecycle.")
    ui_explainer(
        "How to read this page",
        what=[
            "Variants summarize common case paths from the event log.",
            "Transitions and dwell metrics show where work flows smoothly or stalls.",
        ],
        why=[
            "Identify dominant paths and deviations that create extra cycle time.",
            "Prioritize improvement efforts where delay is both frequent and large.",
        ],
        how=[
            "Start with variants to find dominant flow patterns.",
            "Inspect top transitions and overall Sankey to see network pressure points.",
            "Drill into one variant, then review dwell hotspots by event.",
        ],
    )

    try:
        variant_df = run_query(VARIANT_SQL)
        transition_df = run_query(TRANSITION_SUMMARY_SQL)
        variant_transition_df = run_query(TRANSITION_BY_VARIANT_SQL)
        dwell_df = run_query(DWELL_SQL)
    except Exception as exc:
        st.error(f"Failed to query process explorer views: {exc}")
        st.stop()

    st.markdown("#### A) Variant Leaderboard")
    variant_options: list[str] = []
    variants_sorted = pd.DataFrame()
    if variant_df.empty:
        st.info("No rows in im.v_variant_summary.")
    else:
        variants_sorted = variant_df.copy()
        variants_sorted["cases"] = pd.to_numeric(variants_sorted["cases"], errors="coerce")
        for col in ["avg_cycle_hours", "p90_cycle_hours", "avg_csat", "avg_escalations", "avg_resolver_changes"]:
            variants_sorted[col] = pd.to_numeric(variants_sorted[col], errors="coerce")
        for col in ["met_sla_rate", "reopen_rate", "reject_rate"]:
            variants_sorted[col] = pd.to_numeric(variants_sorted[col], errors="coerce")
        variants_sorted = variants_sorted.sort_values("cases", ascending=False)

        variant_options = [
            str(v) for v in variants_sorted["variant"].tolist() if pd.notna(v) and str(v).strip() != ""
        ]

        table_cols = [
            "variant",
            "cases",
            "avg_cycle_hours",
            "p90_cycle_hours",
            "avg_csat",
            "met_sla_rate",
            "reopen_rate",
            "reject_rate",
            "avg_escalations",
            "avg_resolver_changes",
        ]
        top_variants = variants_sorted[table_cols].head(20).copy()
        top_variants["cases"] = top_variants["cases"].astype("Int64")
        st.dataframe(
            style_table(
                top_variants,
                pct_cols={"met_sla_rate", "reopen_rate", "reject_rate"},
                hour_cols={"avg_cycle_hours", "p90_cycle_hours"},
                good_high_cols={"cases", "met_sla_rate"},
                good_low_cols={"avg_cycle_hours", "p90_cycle_hours", "reopen_rate", "reject_rate"},
            ),
            use_container_width=True,
            hide_index=True,
        )

        chart_df = variants_sorted.dropna(subset=["variant", "cases"]).head(15).copy()
        if chart_df.empty:
            st.info("Not enough variant data for charting.")
        else:
            fig_cases = px.bar(
                chart_df,
                x="variant",
                y="cases",
                color="variant",
                text_auto=True,
            )
            fig_cases.update_traces(hovertemplate="Variant=%{x}<br>Cases=%{y}<extra></extra>")
            st.plotly_chart(
                clayout(fig_cases, title="Cases by Variant", xtitle="Variant", ytitle="Cases", h=500),
                use_container_width=True,
            )

    st.divider()
    st.markdown("#### B) Top Transitions (Overall)")
    transition_sorted = pd.DataFrame()
    if transition_df.empty:
        st.info("No rows in im.v_transition_summary.")
    else:
        transition_sorted = transition_df.copy()
        transition_sorted["transition_count"] = pd.to_numeric(
            transition_sorted["transition_count"], errors="coerce"
        )
        for col in ["avg_delta_hours", "median_delta_hours", "p90_delta_hours"]:
            transition_sorted[col] = pd.to_numeric(transition_sorted[col], errors="coerce")
        transition_sorted = transition_sorted.dropna(subset=["transition_count"])
        transition_sorted = transition_sorted.sort_values("transition_count", ascending=False)

        top20 = transition_sorted[
            ["from_event", "to_event", "transition_count", "avg_delta_hours", "median_delta_hours", "p90_delta_hours"]
        ].head(20).copy()
        top20["transition_count"] = top20["transition_count"].astype("Int64")
        st.dataframe(
            style_table(
                top20,
                hour_cols={"avg_delta_hours", "median_delta_hours", "p90_delta_hours"},
                good_high_cols={"transition_count"},
                good_low_cols={"avg_delta_hours", "median_delta_hours", "p90_delta_hours"},
            ),
            use_container_width=True,
            hide_index=True,
        )

        fig_overall = build_sankey(
            transitions=transition_sorted,
            title="Overall Flow Sankey (Top 25 Transitions)",
            max_rows=25,
            height=540,
        )
        if fig_overall is None:
            st.info("Not enough transition data to draw overall Sankey.")
        else:
            st.plotly_chart(fig_overall, use_container_width=True)

    st.divider()
    st.markdown("#### C) Variant Drilldown")
    if not variant_options:
        st.info("No variant values available for drilldown.")
    elif variant_transition_df.empty:
        st.info("No rows in im.v_transition_by_variant.")
    else:
        selected_variant = st.selectbox(
            "Variant drilldown",
            options=variant_options,
            index=0,
            key="process_variant_select",
        )
        vdf = variant_transition_df.copy()
        vdf["variant"] = vdf["variant"].astype("string")
        vdf["transition_count"] = pd.to_numeric(vdf["transition_count"], errors="coerce")
        vdf["avg_delta_hours"] = pd.to_numeric(vdf["avg_delta_hours"], errors="coerce")
        vdf = vdf.dropna(subset=["transition_count"])
        selected_df = vdf[vdf["variant"] == selected_variant].copy()
        selected_df = selected_df.sort_values("transition_count", ascending=False)

        if selected_df.empty:
            st.info(f"No transition rows for variant '{selected_variant}'.")
        else:
            top20_variant = selected_df[["from_event", "to_event", "transition_count", "avg_delta_hours"]].head(20).copy()
            top20_variant["transition_count"] = top20_variant["transition_count"].astype("Int64")
            st.dataframe(
                style_table(
                    top20_variant,
                    hour_cols={"avg_delta_hours"},
                    good_high_cols={"transition_count"},
                    good_low_cols={"avg_delta_hours"},
                ),
                use_container_width=True,
                hide_index=True,
            )

            fig_variant = build_sankey(
                transitions=selected_df,
                title=f"Variant Flow Sankey: {selected_variant} (Top 20)",
                max_rows=20,
                height=500,
            )
            if fig_variant is None:
                st.info("Not enough transition data to draw variant Sankey.")
            else:
                st.plotly_chart(fig_variant, use_container_width=True)

    st.divider()
    st.markdown("#### D) Dwell Times")
    dwell_sorted = pd.DataFrame()
    if dwell_df.empty:
        st.info("No rows in im.v_dwell_by_event.")
    else:
        dwell_sorted = dwell_df.copy()
        dwell_sorted["occurrences"] = pd.to_numeric(dwell_sorted["occurrences"], errors="coerce")
        for col in ["avg_dwell_hours", "median_dwell_hours", "p90_dwell_hours"]:
            dwell_sorted[col] = pd.to_numeric(dwell_sorted[col], errors="coerce")
        dwell_sorted = dwell_sorted.sort_values("avg_dwell_hours", ascending=False)

        dwell_display = dwell_sorted[
            ["event", "occurrences", "avg_dwell_hours", "median_dwell_hours", "p90_dwell_hours"]
        ].head(20).copy()
        dwell_display["occurrences"] = dwell_display["occurrences"].astype("Int64")
        st.dataframe(
            style_table(
                dwell_display,
                hour_cols={"avg_dwell_hours", "median_dwell_hours", "p90_dwell_hours"},
                good_high_cols={"occurrences"},
                good_low_cols={"avg_dwell_hours", "median_dwell_hours", "p90_dwell_hours"},
            ),
            use_container_width=True,
            hide_index=True,
        )

        top10_dwell = dwell_sorted.head(10).sort_values("avg_dwell_hours", ascending=True)
        fig_dwell = px.bar(
            top10_dwell,
            x="avg_dwell_hours",
            y="event",
            orientation="h",
            text_auto=".2f",
        )
        fig_dwell.update_traces(
            hovertemplate="Event=%{y}<br>Avg Dwell=%{x:.2f} hrs<extra></extra>",
        )
        st.plotly_chart(
            clayout(
                fig_dwell,
                title="Top 10 Events by Avg Dwell Hours",
                xtitle="Avg Dwell (hrs)",
                ytitle="Event",
                h=500,
            ),
            use_container_width=True,
        )

    insight_callouts: list[str] = []
    if not transition_sorted.empty:
        common = transition_sorted.sort_values("transition_count", ascending=False).iloc[0]
        insight_callouts.append(
            f"Most common transition: {common['from_event']} -> {common['to_event']} ({int(common['transition_count']):,} transitions)."
        )

        slow = transition_sorted.dropna(subset=["p90_delta_hours"]).sort_values("p90_delta_hours", ascending=False)
        if not slow.empty:
            slowest = slow.iloc[0]
            insight_callouts.append(
                f"Slowest p90 transition: {slowest['from_event']} -> {slowest['to_event']} (~{float(slowest['p90_delta_hours']):.2f} hrs)."
            )

    if not dwell_sorted.empty:
        dwell_non_null = dwell_sorted.dropna(subset=["avg_dwell_hours"])
        if not dwell_non_null.empty:
            longest = dwell_non_null.iloc[0]
            insight_callouts.append(
                f"Longest avg dwell step: {longest['event']} (~{float(longest['avg_dwell_hours']):.2f} hrs)."
            )

    if insight_callouts:
        st.info(insight_callouts[0])
        if len(insight_callouts) > 1:
            st.info(insight_callouts[1])


def render_bottlenecks(db_url: str) -> None:
    header("Bottlenecks", "Identify where time is lost across steps and handoffs.")
    ui_explainer(
        "How to read this page",
        what=[
            "Dwell time shows how long work sits in a step.",
            "Transition delay shows handoff lag between two events.",
        ],
        why=[
            "Bottlenecks usually appear where delay is high and repeats often.",
            "These views highlight where waiting is likely driving cycle-time inflation.",
        ],
        how=[
            "Start with top dwell steps.",
            "Review slowest p90 transitions.",
            "Prioritize fixes on offenders with both high delay and high frequency.",
        ],
    )

    try:
        dwell_df = run_query(DWELL_SQL)
        transition_df = run_query(TRANSITION_SUMMARY_SQL)
    except Exception as exc:
        st.error(f"Failed to query bottleneck views: {exc}")
        st.stop()

    dwell_num = pd.DataFrame()
    if not dwell_df.empty:
        dwell_num = dwell_df.copy()
        dwell_num["occurrences"] = pd.to_numeric(dwell_num["occurrences"], errors="coerce")
        for col in ["avg_dwell_hours", "median_dwell_hours", "p90_dwell_hours"]:
            dwell_num[col] = pd.to_numeric(dwell_num[col], errors="coerce")
        dwell_num = dwell_num.sort_values("avg_dwell_hours", ascending=False)

    transition_num = pd.DataFrame()
    if not transition_df.empty:
        transition_num = transition_df.copy()
        transition_num["transition_count"] = pd.to_numeric(transition_num["transition_count"], errors="coerce")
        for col in ["avg_delta_hours", "median_delta_hours", "p90_delta_hours"]:
            transition_num[col] = pd.to_numeric(transition_num[col], errors="coerce")
        transition_num = transition_num.dropna(subset=["transition_count"])

    st.divider()
    st.markdown("#### Row 1: Top Dwell Steps")
    if dwell_num.empty:
        st.info("No rows in im.v_dwell_by_event.")
    else:
        top_dwell = dwell_num[
            ["event", "occurrences", "avg_dwell_hours", "median_dwell_hours", "p90_dwell_hours"]
        ].head(15).copy()
        top_dwell["occurrences"] = top_dwell["occurrences"].astype("Int64")
        st.dataframe(
            style_table(
                top_dwell,
                hour_cols={"avg_dwell_hours", "median_dwell_hours", "p90_dwell_hours"},
                good_high_cols={"occurrences"},
                good_low_cols={"avg_dwell_hours", "median_dwell_hours", "p90_dwell_hours"},
            ),
            use_container_width=True,
            hide_index=True,
        )

    if dwell_num.empty:
        st.info("Not enough dwell data for charting.")
    else:
        top10_dwell = dwell_num.head(10).copy().sort_values("avg_dwell_hours", ascending=True)
        fig_dwell = px.bar(
            top10_dwell,
            x="avg_dwell_hours",
            y="event",
            orientation="h",
            text_auto=".2f",
        )
        fig_dwell.update_traces(hovertemplate="Event=%{y}<br>Avg Dwell=%{x:.2f} hrs<extra></extra>")
        st.plotly_chart(
            clayout(
                fig_dwell,
                title="Top 10 Dwell Steps by Avg Hours",
                xtitle="Avg Dwell (hrs)",
                ytitle="Event",
                h=500,
            ),
            use_container_width=True,
        )

    st.divider()
    st.markdown("#### Row 2: Slowest Transitions by P90")
    if transition_num.empty:
        st.info("No rows in im.v_transition_summary.")
    else:
        slowest15 = transition_num.dropna(subset=["p90_delta_hours"]).sort_values(
            "p90_delta_hours", ascending=False
        ).head(15)
        slowest15 = slowest15[
            [
                "from_event",
                "to_event",
                "transition_count",
                "avg_delta_hours",
                "median_delta_hours",
                "p90_delta_hours",
            ]
        ].copy()
        slowest15["transition_count"] = slowest15["transition_count"].astype("Int64")
        st.dataframe(
            style_table(
                slowest15,
                hour_cols={"avg_delta_hours", "median_delta_hours", "p90_delta_hours"},
                good_high_cols={"transition_count"},
                good_low_cols={"avg_delta_hours", "median_delta_hours", "p90_delta_hours"},
            ),
            use_container_width=True,
            hide_index=True,
        )

    if transition_num.empty:
        st.info("Not enough transition data for slowest-transition chart.")
    else:
        top10_slowest = transition_num.dropna(subset=["p90_delta_hours"]).sort_values(
            "p90_delta_hours", ascending=False
        ).head(10)
        if top10_slowest.empty:
            st.info("Not enough transition data for slowest-transition chart.")
        else:
            top10_slowest = top10_slowest.copy()
            top10_slowest["transition_label"] = (
                top10_slowest["from_event"].astype(str) + " -> " + top10_slowest["to_event"].astype(str)
            )
            top10_slowest = top10_slowest.sort_values("p90_delta_hours", ascending=True)
            fig_slow = px.bar(
                top10_slowest,
                x="p90_delta_hours",
                y="transition_label",
                orientation="h",
                text_auto=".2f",
            )
            fig_slow.update_traces(hovertemplate="Transition=%{y}<br>P90 Delay=%{x:.2f} hrs<extra></extra>")
            st.plotly_chart(
                clayout(
                    fig_slow,
                    title="Top 10 Slowest Transitions (P90)",
                    xtitle="P90 Delay (hrs)",
                    ytitle="Transition",
                    h=500,
                ),
                use_container_width=True,
            )

    st.divider()
    st.markdown("#### Row 3: Most Common Transitions")
    if transition_num.empty:
        st.info("No transition frequency rows available.")
    else:
        common15 = transition_num.sort_values("transition_count", ascending=False).head(15).copy()
        common15 = common15[
            ["from_event", "to_event", "transition_count", "avg_delta_hours", "median_delta_hours", "p90_delta_hours"]
        ]
        common15["transition_count"] = common15["transition_count"].astype("Int64")
        st.dataframe(
            style_table(
                common15,
                hour_cols={"avg_delta_hours", "median_delta_hours", "p90_delta_hours"},
                good_high_cols={"transition_count"},
                good_low_cols={"p90_delta_hours"},
            ),
            use_container_width=True,
            hide_index=True,
        )

    action_items: list[str] = []
    if not dwell_num.empty:
        top_dwell_step = dwell_num.dropna(subset=["avg_dwell_hours"]).head(1)
        if not top_dwell_step.empty:
            row = top_dwell_step.iloc[0]
            action_items.append(
                f"Run a capacity review for `{row['event']}` where avg dwell is ~{float(row['avg_dwell_hours']):.2f} hrs."
            )
    if not transition_num.empty:
        top_slow_transition = transition_num.dropna(subset=["p90_delta_hours"]).sort_values(
            "p90_delta_hours", ascending=False
        ).head(1)
        if not top_slow_transition.empty:
            row = top_slow_transition.iloc[0]
            action_items.append(
                "Audit routing/assignment rules for "
                f"`{row['from_event']} -> {row['to_event']}` (p90 ~{float(row['p90_delta_hours']):.2f} hrs)."
            )
    action_items.append("Revalidate SLA timers and handoff alerts for delayed stages and transitions.")
    st.info("Recommended next actions:\n" + "\n".join([f"- {item}" for item in action_items[:3]]))

    with st.expander("Data Notes"):
        st.write(
            "These metrics are computed from event timestamps and should be used to identify process friction, not individual performance."
        )


def render_escalations_handoffs(db_url: str) -> None:
    header("Escalations & Handoffs", "Analyze handoff tax, escalation intensity, and ping-pong behavior.")
    ui_explainer(
        "How to read this page",
        what=[
            "Handoffs are approximated by resolver changes.",
            "Escalations are tracked via escalation_count.",
            "Ping-pong indicates repeated back-and-forth transitions between resolver levels.",
        ],
        why=[
            "Excess handoffs often increase cycle time and degrade SLA and CSAT outcomes.",
            "Ping-pong patterns highlight routing or ownership ambiguity.",
        ],
        how=[
            "Use Hotspots to find problematic variant/issue combos.",
            "Use Ping-Pong Cases to isolate case-level patterns.",
            "Use Worklist to export highest-priority remediation cases.",
        ],
    )

    try:
        handoff_df = run_query(HANDOFF_SUMMARY_SQL)
        pingpong_df = run_query(PINGPONG_CASES_SQL)
        worst_df = run_query(WORST_HANDOFF_SQL)
        pingpong_kpi_df = run_query(PINGPONG_KPI_SQL)
        overall_kpi_df = run_query(OVERALL_HANDOFF_BASELINE_SQL)
    except Exception as exc:
        st.error(f"Failed to query escalation/handoff views: {exc}")
        st.stop()

    handoff_num = pd.DataFrame()
    if not handoff_df.empty:
        handoff_num = handoff_df.copy()
        for col in [
            "cases",
            "avg_cycle_hours",
            "p90_cycle_hours",
            "avg_csat",
            "met_sla_rate",
            "avg_escalations",
            "avg_resolver_changes",
            "handoff_rate",
            "high_handoff_rate",
            "pingpong_rate",
            "reopen_rate",
            "reject_rate",
        ]:
            if col in handoff_num.columns:
                handoff_num[col] = pd.to_numeric(handoff_num[col], errors="coerce")

    pingpong_cases_num = pd.DataFrame()
    if not pingpong_df.empty:
        pingpong_cases_num = pingpong_df.copy()
        for col in [
            "cycle_hours",
            "customer_satisfaction",
            "escalation_count",
            "resolver_changes",
            "pingpong_transitions",
        ]:
            pingpong_cases_num[col] = pd.to_numeric(pingpong_cases_num[col], errors="coerce")
        for col in ["met_sla", "has_reopen", "has_reject"]:
            if col in pingpong_cases_num.columns:
                pingpong_cases_num[col] = pingpong_cases_num[col].fillna(False).astype(bool)

    insight_callouts: list[str] = []
    if not handoff_num.empty:
        hotspot_pool = handoff_num.copy()
        if "cases" in hotspot_pool.columns:
            hotspot_pool = hotspot_pool[hotspot_pool["cases"].fillna(0) >= 10]
        if not hotspot_pool.empty:
            worst_hotspot = hotspot_pool.sort_values(
                ["high_handoff_rate", "cases"], ascending=[False, False]
            ).iloc[0]
            insight_callouts.append(
                "Worst hotspot combo: "
                f"{worst_hotspot['variant']} | {worst_hotspot['issue_type']} "
                f"(high handoff={fmt_pct(worst_hotspot['high_handoff_rate'])}, "
                f"cases={fmt_int(worst_hotspot['cases'])})."
            )

    if not pingpong_kpi_df.empty and not overall_kpi_df.empty:
        ping_sla = pd.to_numeric(pingpong_kpi_df.iloc[0]["met_sla_rate_pingpong"], errors="coerce")
        overall_sla = pd.to_numeric(overall_kpi_df.iloc[0]["met_sla_rate_overall"], errors="coerce")
        if pd.notna(ping_sla) and pd.notna(overall_sla):
            delta = ping_sla - overall_sla
            insight_callouts.append(
                f"Ping-pong cohort SLA vs overall: {fmt_pct(ping_sla)} vs {fmt_pct(overall_sla)} ({delta * 100:+.1f}pp)."
            )

    if not pingpong_cases_num.empty and "issue_type" in pingpong_cases_num.columns:
        issue_counts = pingpong_cases_num["issue_type"].dropna().astype(str).value_counts().reset_index()
        issue_counts.columns = ["issue_type", "cases"]
        if not issue_counts.empty:
            top_issue = issue_counts.iloc[0]
            insight_callouts.append(
                f"Top issue_type driving ping-pong cases: {top_issue['issue_type']} ({fmt_int(top_issue['cases'])} cases)."
            )

    for message in insight_callouts[:3]:
        st.info(message)

    tab_overview, tab_hotspots, tab_pingpong, tab_worklist = st.tabs(
        ["Overview", "Hotspots", "Ping-Pong Cases", "Worklist"]
    )

    with tab_overview:
        st.markdown("#### Cohort Comparison: Ping-Pong vs Overall")
        if pingpong_kpi_df.empty or overall_kpi_df.empty:
            st.info("Not enough KPI data to compare ping-pong cohort versus overall baseline.")
        else:
            p = pingpong_kpi_df.iloc[0]
            o = overall_kpi_df.iloc[0]
            overview_left, overview_right = st.columns(2)

            with overview_left:
                st.markdown("##### Ping-Pong Cohort")
                ping_cols_1 = st.columns(3)
                with ping_cols_1[0]:
                    metric_tile("Cases", fmt_int(p["pingpong_cases"]))
                with ping_cols_1[1]:
                    metric_tile("Avg Cycle (hrs)", fmt_float(p["avg_cycle_hours_pingpong"]))
                with ping_cols_1[2]:
                    metric_tile("Avg CSAT", fmt_float(p["avg_csat_pingpong"]))

                ping_cols_2 = st.columns(3)
                with ping_cols_2[0]:
                    metric_tile("SLA Met %", fmt_pct(p["met_sla_rate_pingpong"]))
                with ping_cols_2[1]:
                    metric_tile("Avg Resolver Changes", fmt_float(p["avg_resolver_changes_pingpong"]))
                with ping_cols_2[2]:
                    metric_tile("Avg Escalations", fmt_float(p["avg_escalations_pingpong"]))

            with overview_right:
                st.markdown("##### Overall Baseline")
                base_cols_1 = st.columns(3)
                with base_cols_1[0]:
                    metric_tile("Cases", fmt_int(o["overall_cases"]))
                with base_cols_1[1]:
                    metric_tile("Avg Cycle (hrs)", fmt_float(o["avg_cycle_hours_overall"]))
                with base_cols_1[2]:
                    metric_tile("Avg CSAT", fmt_float(o["avg_csat_overall"]))

                base_cols_2 = st.columns(3)
                with base_cols_2[0]:
                    metric_tile("SLA Met %", fmt_pct(o["met_sla_rate_overall"]))
                with base_cols_2[1]:
                    metric_tile("Avg Resolver Changes", fmt_float(o["avg_resolver_changes_overall"]))
                with base_cols_2[2]:
                    metric_tile("Avg Escalations", fmt_float(o["avg_escalations_overall"]))

            compare_df = pd.DataFrame(
                [
                    {"metric": "Avg Cycle (hrs)", "cohort": "Ping-Pong", "value": p["avg_cycle_hours_pingpong"]},
                    {"metric": "Avg Cycle (hrs)", "cohort": "Overall", "value": o["avg_cycle_hours_overall"]},
                    {
                        "metric": "Avg Resolver Changes",
                        "cohort": "Ping-Pong",
                        "value": p["avg_resolver_changes_pingpong"],
                    },
                    {"metric": "Avg Resolver Changes", "cohort": "Overall", "value": o["avg_resolver_changes_overall"]},
                    {"metric": "Avg Escalations", "cohort": "Ping-Pong", "value": p["avg_escalations_pingpong"]},
                    {"metric": "Avg Escalations", "cohort": "Overall", "value": o["avg_escalations_overall"]},
                    {"metric": "SLA Met Rate", "cohort": "Ping-Pong", "value": p["met_sla_rate_pingpong"]},
                    {"metric": "SLA Met Rate", "cohort": "Overall", "value": o["met_sla_rate_overall"]},
                ]
            )
            compare_df["value"] = pd.to_numeric(compare_df["value"], errors="coerce")
            fig_compare = px.bar(compare_df, x="metric", y="value", color="cohort", barmode="group")
            fig_compare.update_traces(
                hovertemplate="Metric=%{x}<br>Cohort=%{fullData.name}<br>Value=%{y:.3f}<extra></extra>"
            )
            st.plotly_chart(
                clayout(
                    fig_compare,
                    title="Ping-Pong Cohort vs Overall Baseline",
                    xtitle="Metric",
                    ytitle="Value",
                    h=430,
                ),
                use_container_width=True,
            )

    with tab_hotspots:
        st.markdown("#### Handoff Hotspots")
        if handoff_num.empty:
            st.info("No rows in im.v_handoff_summary.")
        else:
            sort_metric = st.selectbox(
                "Rank by",
                options=["high_handoff_rate", "pingpong_rate"],
                format_func=lambda v: "High Handoff Rate" if v == "high_handoff_rate" else "Ping-Pong Rate",
                key="eh_hotspots_rank_metric",
            )
            ranked = handoff_num.sort_values([sort_metric, "cases"], ascending=[False, False])
            top25 = ranked.head(25).copy()

            base_cols = [
                "variant",
                "issue_type",
                "cases",
                "avg_cycle_hours",
                "p90_cycle_hours",
                "avg_csat",
                "met_sla_rate",
                "avg_resolver_changes",
                "avg_escalations",
                "handoff_rate",
                "high_handoff_rate",
                "reopen_rate",
                "reject_rate",
                "pingpong_rate",
            ]
            display_cols = [col for col in base_cols if col in top25.columns]
            top25_display = top25[display_cols].copy()
            if "cases" in top25_display.columns:
                top25_display["cases"] = top25_display["cases"].astype("Int64")

            st.dataframe(
                style_table(
                    top25_display,
                    pct_cols={
                        "met_sla_rate",
                        "handoff_rate",
                        "high_handoff_rate",
                        "pingpong_rate",
                        "reopen_rate",
                        "reject_rate",
                    },
                    hour_cols={"avg_cycle_hours", "p90_cycle_hours"},
                    good_high_cols={"met_sla_rate", "avg_csat", "cases"},
                    bad_high_cols={"high_handoff_rate", "pingpong_rate", "reject_rate", "reopen_rate"},
                ),
                use_container_width=True,
                hide_index=True,
            )

            top10_hot = ranked.head(10).copy()
            top10_hot["label"] = top10_hot["variant"].astype(str) + " | " + top10_hot["issue_type"].astype(str)
            top10_hot = top10_hot.sort_values(sort_metric, ascending=True)
            fig_hot = px.bar(
                top10_hot,
                x=sort_metric,
                y="label",
                orientation="h",
                text_auto=".1%",
            )
            fig_hot.update_traces(hovertemplate="Group=%{y}<br>Rate=%{x:.1%}<extra></extra>")
            st.plotly_chart(
                clayout(
                    fig_hot,
                    title="Top 10 Hotspot Groups",
                    xtitle="Rate",
                    ytitle="Variant | Issue Type",
                    h=560,
                ),
                use_container_width=True,
            )

    with tab_pingpong:
        st.markdown("#### Ping-Pong Explorer")
        if pingpong_cases_num.empty:
            st.info("No rows in im.v_pingpong_cases.")
        else:
            cases_df = pingpong_cases_num.copy()

            filter_col1, filter_col2, filter_col3 = st.columns([1.0, 1.2, 0.9])
            with filter_col1:
                priority_opts = sorted([str(v) for v in cases_df["priority"].dropna().unique().tolist()])
                selected_priorities = st.multiselect(
                    "Priority", options=priority_opts, default=priority_opts, key="eh_pingpong_priority"
                )
            with filter_col2:
                issue_opts = sorted([str(v) for v in cases_df["issue_type"].dropna().unique().tolist()])
                selected_issues = st.multiselect(
                    "Issue Type", options=issue_opts, default=issue_opts, key="eh_pingpong_issue_type"
                )
            with filter_col3:
                sla_filter = st.selectbox(
                    "SLA", options=["All", "Met", "Missed"], index=0, key="eh_pingpong_sla_filter"
                )

            filtered = cases_df.copy()
            if selected_priorities:
                filtered = filtered[filtered["priority"].astype(str).isin(selected_priorities)]
            else:
                filtered = filtered.iloc[0:0]
            if selected_issues:
                filtered = filtered[filtered["issue_type"].astype(str).isin(selected_issues)]
            else:
                filtered = filtered.iloc[0:0]

            if sla_filter == "Met":
                filtered = filtered[filtered["met_sla"] == True]
            elif sla_filter == "Missed":
                filtered = filtered[filtered["met_sla"] == False]

            filtered = filtered.sort_values(
                ["resolver_changes", "escalation_count", "cycle_hours"],
                ascending=[False, False, False],
            )

            if filtered.empty:
                st.info("No rows match current filter selection.")
            else:
                display_cols = [
                    "case_id",
                    "priority",
                    "issue_type",
                    "variant",
                    "cycle_hours",
                    "customer_satisfaction",
                    "met_sla",
                    "resolver_changes",
                    "escalation_count",
                    "pingpong_transitions",
                    "has_reopen",
                    "has_reject",
                ]
                top100 = filtered[display_cols].head(100).copy()
                st.dataframe(
                    style_table(
                        top100,
                        hour_cols={"cycle_hours"},
                        good_high_cols={"met_sla"},
                        bad_high_cols={"cycle_hours"},
                        bad_low_cols={"customer_satisfaction", "met_sla"},
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

            st.download_button(
                "Download filtered ping-pong CSV",
                data=filtered.to_csv(index=False).encode("utf-8"),
                file_name="pingpong_cases_filtered.csv",
                mime="text/csv",
                key="dl_eh_pingpong_filtered",
            )

            if filtered.empty:
                st.info("No rows match current filter selection.")
            else:
                plot_df = filtered.dropna(
                    subset=["resolver_changes", "cycle_hours", "pingpong_transitions"]
                ).copy()
                if plot_df.empty:
                    st.info("Not enough numeric data for ping-pong scatter chart.")
                else:
                    fig_scatter = px.scatter(
                        plot_df,
                        x="resolver_changes",
                        y="cycle_hours",
                        size="pingpong_transitions",
                        color="priority",
                        hover_name="case_id",
                        size_max=56,
                    )
                    fig_scatter.update_traces(
                        marker=dict(line=dict(width=1, color="rgba(250,250,250,0.35)"), opacity=0.82),
                        hovertemplate=(
                            "Case=%{hovertext}<br>Resolver Changes=%{x:.0f}<br>Cycle=%{y:.2f} hrs"
                            "<br>Ping-Pong Transitions=%{marker.size:.0f}<extra></extra>"
                        ),
                    )
                    st.plotly_chart(
                        clayout(
                            fig_scatter,
                            title="Resolver Changes vs Cycle Time (Ping-Pong Cases)",
                            xtitle="Resolver Changes",
                            ytitle="Cycle Hours",
                            h=520,
                        ),
                        use_container_width=True,
                    )

    with tab_worklist:
        st.markdown("#### Ops Worklist: Worst Handoff Cases")
        if worst_df.empty:
            st.info("No rows in im.v_worst_handoff_cases.")
        else:
            worklist = worst_df.copy()
            for col in ["cycle_hours", "customer_satisfaction", "escalation_count", "resolver_changes"]:
                worklist[col] = pd.to_numeric(worklist[col], errors="coerce")
            if "met_sla" in worklist.columns:
                worklist["met_sla"] = worklist["met_sla"].fillna(False).astype(bool)
            worklist = worklist.sort_values(
                ["resolver_changes", "escalation_count", "cycle_hours"],
                ascending=[False, False, False],
            )

            display_cols = [
                "case_id",
                "variant",
                "priority",
                "issue_type",
                "report_channel",
                "cycle_hours",
                "customer_satisfaction",
                "met_sla",
                "escalation_count",
                "resolver_changes",
                "has_reopen",
                "has_reject",
            ]
            display_cols = [col for col in display_cols if col in worklist.columns]
            worklist_display = worklist[display_cols].head(200).copy()
            st.dataframe(
                style_table(
                    worklist_display,
                    hour_cols={"cycle_hours"},
                    good_high_cols={"met_sla"},
                    bad_high_cols={"cycle_hours"},
                    bad_low_cols={"customer_satisfaction", "met_sla"},
                ),
                use_container_width=True,
                hide_index=True,
            )

            st.download_button(
                "Download worklist CSV",
                data=worklist.to_csv(index=False).encode("utf-8"),
                file_name="worst_handoff_cases.csv",
                mime="text/csv",
                key="dl_eh_worklist",
            )


def render_quality_cx(db_url: str) -> None:
    header("Quality & CX", "Track closure quality, feedback coverage, and reopen-after-close outcomes.")
    ui_explainer(
        "How to read this page",
        what=[
            "Tracks closure compliance, feedback gaps, reopen-after-close, and rejects.",
            "Compares quality outcomes across issue and priority cohorts.",
        ],
        why=[
            "These factors are strong leading indicators of customer experience outcomes.",
            "Reducing reopen/reject/feedback gaps generally improves cycle time and SLA reliability.",
        ],
        how=[
            "Start in Overview for global signal health.",
            "Use Cohorts to isolate problematic case subsets.",
            "Use Breakdown to prioritize issue_type | priority fixes and Export for offline analysis.",
        ],
    )

    try:
        cx_summary_df = run_query(CX_SUMMARY_SQL)
        cx_breakdown_df = run_query(CX_BREAKDOWN_SQL)
        closure_df = run_query(CLOSURE_COMPLIANCE_SQL)
    except Exception as exc:
        st.error(f"Failed to query quality/CX views: {exc}")
        st.stop()

    closure_num = pd.DataFrame()
    if not closure_df.empty:
        closure_num = closure_df.copy()
        for col in ["cycle_hours", "customer_satisfaction", "reopen_after_close_hours"]:
            closure_num[col] = pd.to_numeric(closure_num[col], errors="coerce")
        for col in [
            "met_sla",
            "has_reject",
            "closed_without_feedback",
            "reopened_within_3h",
            "reopened_within_24h",
        ]:
            if col in closure_num.columns:
                closure_num[col] = closure_num[col].fillna(False).astype(bool)

    tab_overview, tab_cohorts, tab_breakdown, tab_export = st.tabs(
        ["Overview", "Cohorts", "Breakdown", "Export"]
    )

    with tab_overview:
        st.markdown("#### CX Summary")
        if cx_summary_df.empty:
            st.info("No rows in im.v_cx_summary.")
        else:
            s = cx_summary_df.iloc[0]
            row_1 = st.columns(5)
            row_2 = st.columns(4)

            with row_1[0]:
                metric_tile("Cases", fmt_int(s["cases"]))
            with row_1[1]:
                metric_tile("Avg CSAT", fmt_float(s["avg_csat"]))
            with row_1[2]:
                metric_tile("SLA Met %", fmt_pct(s["met_sla_rate"]))
            with row_1[3]:
                metric_tile("No Feedback %", fmt_pct(s["closed_without_feedback_rate"]))
            with row_1[4]:
                metric_tile("Reject %", fmt_pct(s["reject_rate"]))

            with row_2[0]:
                metric_tile("Reopen <=3h %", fmt_pct(s["reopened_within_3h_rate"]))
            with row_2[1]:
                metric_tile("Reopen <=24h %", fmt_pct(s["reopened_within_24h_rate"]))
            with row_2[2]:
                metric_tile("Avg Cycle (hrs)", fmt_float(s["avg_cycle_hours"]))
            with row_2[3]:
                metric_tile("P90 Cycle (hrs)", fmt_float(s["p90_cycle_hours"]))

            col_rates, col_cycle = st.columns([1.3, 1.0])
            with col_rates:
                rates_df = pd.DataFrame(
                    {
                        "metric": [
                            "No Feedback",
                            "Reopen <=3h",
                            "Reopen <=24h",
                            "Reject",
                            "SLA Met",
                        ],
                        "rate": [
                            s["closed_without_feedback_rate"],
                            s["reopened_within_3h_rate"],
                            s["reopened_within_24h_rate"],
                            s["reject_rate"],
                            s["met_sla_rate"],
                        ],
                    }
                )
                rates_df["rate"] = pd.to_numeric(rates_df["rate"], errors="coerce").fillna(0.0)
                fig_rates = px.bar(rates_df, x="metric", y="rate", text_auto=".1%")
                fig_rates.update_traces(
                    hovertemplate="Metric=%{x}<br>Rate=%{y:.1%}<extra></extra>",
                )
                st.plotly_chart(
                    clayout(
                        fig_rates,
                        title="Quality/CX Rates",
                        xtitle="Metric",
                        ytitle="Rate",
                        h=420,
                    ),
                    use_container_width=True,
                )

            with col_cycle:
                cycle_df = pd.DataFrame(
                    {
                        "metric": ["Avg Cycle", "P90 Cycle"],
                        "hours": [s["avg_cycle_hours"], s["p90_cycle_hours"]],
                    }
                )
                cycle_df["hours"] = pd.to_numeric(cycle_df["hours"], errors="coerce")
                fig_cycle = px.bar(cycle_df, x="metric", y="hours", text_auto=".2f")
                fig_cycle.update_traces(
                    hovertemplate="Metric=%{x}<br>Hours=%{y:.2f}<extra></extra>",
                )
                st.plotly_chart(
                    clayout(
                        fig_cycle,
                        title="Cycle Time Benchmarks",
                        xtitle="Metric",
                        ytitle="Hours",
                        h=420,
                    ),
                    use_container_width=True,
                )

            st.info(
                "CX signal check: "
                f"No feedback={fmt_pct(s['closed_without_feedback_rate'])}, "
                f"Reopen<=24h={fmt_pct(s['reopened_within_24h_rate'])}, "
                f"Reject={fmt_pct(s['reject_rate'])}."
            )

    with tab_cohorts:
        st.markdown("#### Cohort Explorer")
        if closure_num.empty:
            st.info("No rows in im.v_closure_compliance.")
        else:
            f1, f2, f3 = st.columns([1.2, 1.2, 1.0])
            with f1:
                only_no_feedback = st.checkbox(
                    "Closed without feedback only", value=False, key="qcx_cohort_only_no_feedback"
                )
                only_reopen_3h = st.checkbox(
                    "Reopened within 3 hours only", value=False, key="qcx_cohort_only_reopen_3h"
                )
                only_reopen_24h = st.checkbox(
                    "Reopened within 24 hours only", value=False, key="qcx_cohort_only_reopen_24h"
                )
            with f2:
                only_rejected = st.checkbox("Rejected only", value=False, key="qcx_cohort_only_rejected")
                only_missed_sla = st.checkbox("Missed SLA only", value=False, key="qcx_cohort_only_missed_sla")
                priority_opts = sorted([str(v) for v in closure_num["priority"].dropna().unique().tolist()])
                selected_priorities = st.multiselect(
                    "Priority",
                    options=priority_opts,
                    default=priority_opts,
                    key="qcx_cohort_priority",
                )
            with f3:
                issue_opts = sorted([str(v) for v in closure_num["issue_type"].dropna().unique().tolist()])
                selected_issue_types = st.multiselect(
                    "Issue Type",
                    options=issue_opts,
                    default=issue_opts,
                    key="qcx_cohort_issue_type",
                )

            filtered = closure_num.copy()
            if selected_priorities:
                filtered = filtered[filtered["priority"].astype(str).isin(selected_priorities)]
            else:
                filtered = filtered.iloc[0:0]
            if selected_issue_types:
                filtered = filtered[filtered["issue_type"].astype(str).isin(selected_issue_types)]
            else:
                filtered = filtered.iloc[0:0]

            if only_no_feedback:
                filtered = filtered[filtered["closed_without_feedback"]]
            if only_reopen_3h:
                filtered = filtered[filtered["reopened_within_3h"]]
            if only_reopen_24h:
                filtered = filtered[filtered["reopened_within_24h"]]
            if only_rejected:
                filtered = filtered[filtered["has_reject"]]
            if only_missed_sla:
                filtered = filtered[filtered["met_sla"] == False]

            filtered = filtered.sort_values("cycle_hours", ascending=False)

            display_cols = [
                "case_id",
                "priority",
                "issue_type",
                "variant",
                "report_channel",
                "cycle_hours",
                "customer_satisfaction",
                "met_sla",
                "closed_without_feedback",
                "reopened_within_3h",
                "reopened_within_24h",
                "has_reject",
                "reopen_after_close_hours",
            ]
            top120 = filtered[display_cols].head(120).copy()
            st.dataframe(
                style_table(
                    top120,
                    hour_cols={"cycle_hours", "reopen_after_close_hours"},
                    good_high_cols={"met_sla"},
                    bad_high_cols={
                        "closed_without_feedback",
                        "reopened_within_3h",
                        "reopened_within_24h",
                        "has_reject",
                    },
                    bad_low_cols={"customer_satisfaction", "met_sla"},
                ),
                use_container_width=True,
                hide_index=True,
            )
            st.download_button(
                "Download filtered cohort CSV",
                data=filtered.to_csv(index=False).encode("utf-8"),
                file_name="quality_cx_cohort_filtered.csv",
                mime="text/csv",
                key="dl_qcx_cohort_filtered",
            )

            plot_df = filtered.dropna(subset=["cycle_hours", "customer_satisfaction"]).copy()
            if plot_df.empty:
                st.info("Not enough numeric rows for cohort scatter chart.")
            else:
                fig_scatter = px.scatter(
                    plot_df,
                    x="cycle_hours",
                    y="customer_satisfaction",
                    color="met_sla",
                    hover_name="case_id",
                )
                fig_scatter.update_traces(
                    marker=dict(size=10, line=dict(width=1, color="rgba(250,250,250,0.30)"), opacity=0.80),
                    hovertemplate="Case=%{hovertext}<br>Cycle=%{x:.2f} hrs<br>CSAT=%{y:.2f}<extra></extra>",
                )
                st.plotly_chart(
                    clayout(
                        fig_scatter,
                        title="Cycle Time vs CSAT (Filtered Cohort)",
                        xtitle="Cycle Hours",
                        ytitle="Customer Satisfaction",
                        h=520,
                    ),
                    use_container_width=True,
                )

    with tab_breakdown:
        st.markdown("#### CX Breakdown by Issue Type and Priority")
        if cx_breakdown_df.empty:
            st.info("No rows in im.v_cx_breakdown.")
        else:
            bdf = cx_breakdown_df.copy()
            for col in [
                "cases",
                "avg_csat",
                "met_sla_rate",
                "closed_without_feedback_rate",
                "reopened_within_3h_rate",
                "reject_rate",
                "avg_cycle_hours",
            ]:
                bdf[col] = pd.to_numeric(bdf[col], errors="coerce")
            reopen24_by_combo = pd.DataFrame(columns=["issue_type", "priority", "reopened_within_24h_rate"])
            if not closure_num.empty:
                reopen24_by_combo = (
                    closure_num[["issue_type", "priority", "reopened_within_24h"]]
                    .dropna(subset=["issue_type", "priority"])
                    .assign(reopened_within_24h=lambda d: d["reopened_within_24h"].fillna(False).astype(float))
                    .groupby(["issue_type", "priority"], as_index=False)
                    .agg(reopened_within_24h_rate=("reopened_within_24h", "mean"))
                )

            bdf = bdf.merge(reopen24_by_combo, on=["issue_type", "priority"], how="left")
            bdf["reopened_within_24h_rate"] = pd.to_numeric(bdf["reopened_within_24h_rate"], errors="coerce")
            bdf["combo"] = bdf["issue_type"].astype(str) + " | " + bdf["priority"].astype(str)
            bdf = bdf.sort_values("cases", ascending=False)

            table_cols = [
                "issue_type",
                "priority",
                "cases",
                "avg_csat",
                "met_sla_rate",
                "closed_without_feedback_rate",
                "reopened_within_3h_rate",
                "reopened_within_24h_rate",
                "reject_rate",
                "avg_cycle_hours",
            ]
            top40 = bdf[table_cols].head(40).copy()
            top40["cases"] = top40["cases"].astype("Int64")
            st.dataframe(
                style_table(
                    top40,
                    pct_cols={
                        "met_sla_rate",
                        "closed_without_feedback_rate",
                        "reopened_within_3h_rate",
                        "reopened_within_24h_rate",
                        "reject_rate",
                    },
                    hour_cols={"avg_cycle_hours"},
                    good_high_cols={"met_sla_rate", "avg_csat", "cases"},
                    bad_high_cols={"closed_without_feedback_rate", "reopened_within_24h_rate", "reject_rate"},
                ),
                use_container_width=True,
                hide_index=True,
            )

            chart_cols = st.columns(2)
            with chart_cols[0]:
                top_feedback_gap = bdf.dropna(subset=["closed_without_feedback_rate"]).sort_values(
                    "closed_without_feedback_rate", ascending=False
                ).head(10)
                fig_gap = px.bar(
                    top_feedback_gap,
                    x="combo",
                    y="closed_without_feedback_rate",
                    text_auto=".1%",
                )
                fig_gap.update_traces(hovertemplate="Combo=%{x}<br>No Feedback=%{y:.1%}<extra></extra>")
                st.plotly_chart(
                    clayout(
                        fig_gap,
                        title="Top No-Feedback Rates by Issue|Priority",
                        xtitle="Issue Type | Priority",
                        ytitle="Rate",
                        h=460,
                    ),
                    use_container_width=True,
                )

            with chart_cols[1]:
                top_reopen24 = bdf.dropna(subset=["reopened_within_24h_rate"]).sort_values(
                    "reopened_within_24h_rate", ascending=False
                ).head(10)
                fig_reopen = px.bar(
                    top_reopen24,
                    x="combo",
                    y="reopened_within_24h_rate",
                    text_auto=".1%",
                )
                fig_reopen.update_traces(
                    hovertemplate="Combo=%{x}<br>Reopened <=24h=%{y:.1%}<extra></extra>",
                )
                st.plotly_chart(
                    clayout(
                        fig_reopen,
                        title="Top Reopen<=24h Rates by Issue|Priority",
                        xtitle="Issue Type | Priority",
                        ytitle="Rate",
                        h=460,
                    ),
                    use_container_width=True,
                )

            risk_pool = bdf.dropna(
                subset=["closed_without_feedback_rate", "reopened_within_24h_rate", "reject_rate"]
            ).copy()
            if not risk_pool.empty:
                risk_pool["quality_risk_score"] = (
                    risk_pool["closed_without_feedback_rate"] * 0.4
                    + risk_pool["reopened_within_24h_rate"] * 0.35
                    + risk_pool["reject_rate"] * 0.25
                )
                worst_combo = risk_pool.sort_values("quality_risk_score", ascending=False).iloc[0]
                st.info(
                    "Highest-risk cohort: "
                    f"{worst_combo['issue_type']} | {worst_combo['priority']} "
                    f"(No feedback={fmt_pct(worst_combo['closed_without_feedback_rate'])}, "
                    f"Reopen<=24h={fmt_pct(worst_combo['reopened_within_24h_rate'])}, "
                    f"Reject={fmt_pct(worst_combo['reject_rate'])})."
                )

    with tab_export:
        st.markdown("#### Export Datasets")
        if cx_breakdown_df.empty:
            st.info("No rows in im.v_cx_breakdown to export.")
        else:
            st.download_button(
                "Download v_cx_breakdown CSV",
                data=cx_breakdown_df.to_csv(index=False).encode("utf-8"),
                file_name="v_cx_breakdown.csv",
                mime="text/csv",
                key="dl_qcx_breakdown",
            )

        if closure_df.empty:
            st.info("No rows in im.v_closure_compliance to export.")
        else:
            st.download_button(
                "Download v_closure_compliance CSV",
                data=closure_df.to_csv(index=False).encode("utf-8"),
                file_name="v_closure_compliance.csv",
                mime="text/csv",
                key="dl_qcx_closure",
            )
            st.caption("v_closure_compliance can be large depending on case volume.")

        st.write(
            "Use cohort exports for targeted remediation and the full exports for offline trend analysis in BI or notebooks."
        )

    with st.expander("Data Notes"):
        st.write(
            "These metrics are for process improvement and customer outcomes, not individual performance evaluation."
        )


def render_problem_candidates(db_url: str) -> None:
    header("Problem Candidates", "Rank recurring clusters for prevention backlog planning and evidence-based drilldown.")
    ui_explainer(
        "How to read this page",
        what=[
            "This is a ranked prevention backlog inferred from incident outcomes and repeat patterns.",
            "Each row combines operational pain signals into a candidate priority.",
        ],
        why=[
            "Use it to prioritize problem records and prevention work with measurable impact.",
            "Focuses effort on candidates with high cycle-time, quality, and SLA risk.",
        ],
        how=[
            "Start with Backlog to identify top opportunities.",
            "Use Drilldown to validate evidence at case level.",
            "Export candidate and case evidence for problem management workflows.",
        ],
    )

    try:
        backlog_df = run_query(PROBLEM_CANDIDATES_SQL)
        top_cases_df = run_query(PROBLEM_TOP_CASES_SQL)
    except Exception as exc:
        st.error(f"Failed to query problem-candidate views: {exc}")
        st.stop()

    tabs = st.tabs(["Backlog", "Drilldown", "Top Cases", "Export"])

    selected_issue_type: Optional[str] = None
    selected_norm: Optional[str] = None

    with tabs[0]:
        st.markdown("#### Ranked Prevention Backlog")
        if backlog_df.empty:
            st.info("No rows in im.v_problem_candidates.")
        else:
            df = backlog_df.copy()
            for col in [
                "cases",
                "impact_score",
                "avg_cycle_hours",
                "p90_cycle_hours",
                "avg_csat",
                "met_sla_rate",
                "reopen_rate",
                "reject_rate",
            ]:
                df[col] = pd.to_numeric(df[col], errors="coerce")

            filter_cols = st.columns([1.5, 1.0])
            with filter_cols[0]:
                issue_options = sorted([str(v) for v in df["issue_type"].dropna().unique().tolist()])
                selected_issue_types = st.multiselect(
                    "Issue Type",
                    options=issue_options,
                    default=issue_options,
                    key="pc_backlog_issue_filter",
                )
            with filter_cols[1]:
                max_cases = int(df["cases"].dropna().max()) if not df["cases"].dropna().empty else 10
                min_cases = st.slider(
                    "Minimum Cases",
                    min_value=1,
                    max_value=max(max_cases, 1),
                    value=min(10, max_cases),
                    key="pc_backlog_min_cases",
                )

            filtered = df.copy()
            if selected_issue_types:
                filtered = filtered[filtered["issue_type"].astype(str).isin(selected_issue_types)]
            else:
                filtered = filtered.iloc[0:0]
            filtered = filtered[filtered["cases"] >= min_cases]
            filtered = filtered.sort_values(["impact_score", "cases"], ascending=[False, False])

            table_cols = [
                "issue_type",
                "short_description_norm",
                "cases",
                "impact_score",
                "avg_cycle_hours",
                "p90_cycle_hours",
                "avg_csat",
                "met_sla_rate",
                "reopen_rate",
                "reject_rate",
                "last_seen_ts",
            ]
            top40 = filtered[table_cols].head(40).copy()
            if not top40.empty:
                top40["cases"] = top40["cases"].astype("Int64")
            st.dataframe(
                style_table(
                    top40,
                    pct_cols={"met_sla_rate", "reopen_rate", "reject_rate"},
                    hour_cols={"avg_cycle_hours", "p90_cycle_hours"},
                    good_high_cols={"met_sla_rate", "avg_csat", "cases"},
                    bad_high_cols={"impact_score", "avg_cycle_hours", "p90_cycle_hours", "reopen_rate", "reject_rate"},
                ),
                use_container_width=True,
                hide_index=True,
            )

            chart_df = filtered.head(15).copy()
            impact_col, scatter_col = st.columns(2)
            with impact_col:
                if chart_df.empty:
                    st.info("No candidates match current filters.")
                else:
                    chart_df["candidate_label"] = chart_df.apply(
                        lambda r: f"{r['issue_type']} | {str(r['short_description_norm'])[:34]}"
                        + ("..." if len(str(r["short_description_norm"])) > 34 else ""),
                        axis=1,
                    )
                    fig_impact = px.bar(
                        chart_df.sort_values("impact_score", ascending=True),
                        x="impact_score",
                        y="candidate_label",
                        orientation="h",
                        text_auto=".2f",
                    )
                    fig_impact.update_traces(hovertemplate="Candidate=%{y}<br>Impact=%{x:.2f}<extra></extra>")
                    st.plotly_chart(
                        clayout(
                            fig_impact,
                            title="Impact Score by Candidate (Top 15)",
                            xtitle="Impact Score",
                            ytitle="Issue | Description",
                            h=340,
                        ),
                        use_container_width=True,
                    )

            with scatter_col:
                if filtered.empty:
                    st.info("No candidates match current filters.")
                else:
                    bubble_df = filtered.head(60).copy()
                    fig_scatter = px.scatter(
                        bubble_df,
                        x="cases",
                        y="avg_cycle_hours",
                        size="impact_score",
                        color="issue_type",
                        hover_name="short_description_norm",
                        size_max=52,
                    )
                    fig_scatter.update_traces(
                        marker=dict(line=dict(width=1, color="rgba(250,250,250,0.35)"), opacity=0.82),
                        hovertemplate=(
                            "Candidate=%{hovertext}<br>Cases=%{x:.0f}<br>Avg Cycle=%{y:.2f} hrs<extra></extra>"
                        ),
                    )
                    st.plotly_chart(
                        clayout(
                            fig_scatter,
                            title="Cases vs Avg Cycle (Bubble Size = Impact)",
                            xtitle="Cases",
                            ytitle="Avg Cycle Hours",
                            h=340,
                        ),
                        use_container_width=True,
                    )

            if filtered.empty:
                st.info("No candidates match current filters.")
            else:
                top_candidate = filtered.iloc[0]
                st.info(
                    "Top impact candidate: "
                    f"{top_candidate['issue_type']} | {top_candidate['short_description_norm']} "
                    f"(cases={fmt_int(top_candidate['cases'])}, impact={fmt_float(top_candidate['impact_score'])}, "
                    f"avg_cycle={fmt_float(top_candidate['avg_cycle_hours'])}h, sla_met={fmt_pct(top_candidate['met_sla_rate'])})."
                )

                worst_sla_pool = filtered[filtered["cases"] >= min_cases].dropna(subset=["met_sla_rate"])
                if not worst_sla_pool.empty:
                    worst_sla = worst_sla_pool.sort_values(["met_sla_rate", "cases"], ascending=[True, False]).iloc[0]
                    st.info(
                        "Worst SLA candidate: "
                        f"{worst_sla['issue_type']} | {worst_sla['short_description_norm']} "
                        f"(sla_met={fmt_pct(worst_sla['met_sla_rate'])}, cases={fmt_int(worst_sla['cases'])}, "
                        f"reject={fmt_pct(worst_sla['reject_rate'])}, reopen={fmt_pct(worst_sla['reopen_rate'])})."
                    )

                candidate_options = filtered.apply(
                    lambda r: f"{r['issue_type']} || {r['short_description_norm']}",
                    axis=1,
                ).tolist()
                selected_key = st.selectbox(
                    "Candidate",
                    options=candidate_options,
                    index=0,
                    key="pc_candidate_key",
                )
                selected_issue_type, selected_norm = selected_key.split(" || ", 1)

                st.download_button(
                    "Download filtered backlog CSV",
                    data=filtered.to_csv(index=False).encode("utf-8"),
                    file_name="problem_candidates_filtered.csv",
                    mime="text/csv",
                    key="dl_pc_backlog_filtered",
                )

    with tabs[1]:
        st.markdown("#### Candidate Drilldown")
        if backlog_df.empty:
            st.info("No candidate selected because backlog is empty.")
        else:
            if not selected_issue_type or not selected_norm:
                st.info("Select a candidate in Backlog to load drilldown cases.")
            else:
                try:
                    drill_df = run_query(
                        PROBLEM_CASES_BY_CANDIDATE_SQL,
                        params=(selected_issue_type, selected_norm),
                    )
                except Exception as exc:
                    st.error(f"Failed to query candidate drilldown: {exc}")
                    st.stop()

                st.caption(f"Selected candidate: {selected_issue_type} | {selected_norm}")
                st.info(
                    "How to interpret: use this table to validate recurrence and impact drivers; use the distribution to assess tail-risk cycle times."
                )
                if drill_df.empty:
                    st.info("No cases found for selected candidate.")
                else:
                    for col in ["cycle_hours", "customer_satisfaction"]:
                        drill_df[col] = pd.to_numeric(drill_df[col], errors="coerce")

                    drill_left, drill_right = st.columns([1.25, 0.95])
                    with drill_left:
                        display_cols = [
                            "case_id",
                            "priority",
                            "variant",
                            "report_channel",
                            "cycle_hours",
                            "customer_satisfaction",
                            "met_sla",
                            "has_reopen",
                            "has_reject",
                            "has_feedback",
                            "short_description_raw",
                        ]
                        table_df = drill_df[display_cols].head(200).copy()
                        st.dataframe(
                            style_table(
                                table_df,
                                hour_cols={"cycle_hours"},
                                good_high_cols={"met_sla", "has_feedback"},
                                bad_high_cols={"cycle_hours", "has_reopen", "has_reject"},
                                bad_low_cols={"customer_satisfaction", "met_sla"},
                            ),
                            use_container_width=True,
                            hide_index=True,
                        )

                    with drill_right:
                        hist_df = drill_df.dropna(subset=["cycle_hours"]).copy()
                        if hist_df.empty:
                            st.info("Not enough cycle-hour values for histogram.")
                        else:
                            fig_hist = px.histogram(hist_df, x="cycle_hours", nbins=20)
                            fig_hist.update_traces(
                                hovertemplate="Cycle Hours=%{x:.2f}<br>Cases=%{y}<extra></extra>",
                            )
                            st.plotly_chart(
                                clayout(
                                    fig_hist,
                                    title="Cycle Hours Distribution (Selected Candidate)",
                                    xtitle="Cycle Hours",
                                    ytitle="Cases",
                                    h=460,
                                ),
                                use_container_width=True,
                            )

                    st.download_button(
                        "Download candidate cases CSV",
                        data=drill_df.to_csv(index=False).encode("utf-8"),
                        file_name="problem_candidate_cases.csv",
                        mime="text/csv",
                        key="dl_pc_drill_cases",
                    )

    with tabs[2]:
        st.markdown("#### Top Candidate Cases")
        if top_cases_df.empty:
            st.info("No rows in im.v_problem_candidate_top_cases.")
        else:
            tc = top_cases_df.copy()
            tc["cycle_hours"] = pd.to_numeric(tc["cycle_hours"], errors="coerce")
            tc["customer_satisfaction"] = pd.to_numeric(tc["customer_satisfaction"], errors="coerce")

            filter_cols = st.columns(2)
            with filter_cols[0]:
                issue_options = sorted([str(v) for v in tc["issue_type"].dropna().unique().tolist()])
                issue_filter = st.multiselect(
                    "Issue Type", options=issue_options, default=issue_options, key="pc_top_issue"
                )
            with filter_cols[1]:
                priority_options = sorted([str(v) for v in tc["priority"].dropna().unique().tolist()])
                priority_filter = st.multiselect(
                    "Priority", options=priority_options, default=priority_options, key="pc_top_priority"
                )

            if issue_filter:
                tc = tc[tc["issue_type"].astype(str).isin(issue_filter)]
            else:
                tc = tc.iloc[0:0]
            if priority_filter:
                tc = tc[tc["priority"].astype(str).isin(priority_filter)]
            else:
                tc = tc.iloc[0:0]

            st.download_button(
                "Download filtered top cases CSV",
                data=tc.to_csv(index=False).encode("utf-8"),
                file_name="problem_candidate_top_cases_filtered.csv",
                mime="text/csv",
                key="dl_pc_top_cases_filtered",
            )

            table_cols = [
                "case_id",
                "issue_type",
                "priority",
                "variant",
                "report_channel",
                "cycle_hours",
                "customer_satisfaction",
                "met_sla",
                "has_reopen",
                "has_reject",
                "short_description_norm",
            ]
            tc_display = tc[table_cols].head(300).copy()
            st.dataframe(
                style_table(
                    tc_display,
                    hour_cols={"cycle_hours"},
                    good_high_cols={"met_sla"},
                    bad_high_cols={"cycle_hours", "has_reopen", "has_reject"},
                    bad_low_cols={"customer_satisfaction", "met_sla"},
                ),
                use_container_width=True,
                hide_index=True,
            )

    with tabs[3]:
        st.markdown("#### Export")
        if backlog_df.empty:
            st.info("No backlog rows to export.")
        else:
            st.download_button(
                "Download full backlog CSV",
                data=backlog_df.to_csv(index=False).encode("utf-8"),
                file_name="v_problem_candidates.csv",
                mime="text/csv",
                key="dl_pc_backlog_export",
            )

        if top_cases_df.empty:
            st.info("No top-case rows to export.")
        else:
            st.download_button(
                "Download top cases CSV",
                data=top_cases_df.to_csv(index=False).encode("utf-8"),
                file_name="v_problem_candidate_top_cases.csv",
                mime="text/csv",
                key="dl_pc_top_cases_export",
            )

        st.write(
            "Use backlog to pick prevention epics; use drilldown to create a problem record with evidence."
        )
def render_channel_intake(db_url: str) -> None:
    header("Channel & Intake", "Compare intake channel effectiveness and identify problematic channel+issue combinations.")
    ui_explainer(
        "How to read this page",
        what=[
            "Compares intake channels on SLA, cycle time, CSAT, reject/reopen, and missing feedback.",
            "Highlights where channel + issue combinations create avoidable friction.",
        ],
        why=[
            "Improves intake forms, routing, and channel guidance to users.",
            "Helps reduce preventable delays, rejects, and quality gaps.",
        ],
        how=[
            "Start in Overview for baseline channel performance.",
            "Use Channel Comparison to benchmark tradeoffs.",
            "Use Bad Combos to isolate high-pain channel|issue combinations.",
        ],
    )

    try:
        channel_df = run_query(CHANNEL_SUMMARY_SQL)
        channel_issue_df = run_query(CHANNEL_ISSUE_SUMMARY_SQL)
    except Exception as exc:
        st.error(f"Failed to query channel/intake views: {exc}")
        st.stop()

    tabs = st.tabs(["Overview", "Channel Comparison", "Bad Combos", "Export"])
    filtered_bad_combos_for_export = pd.DataFrame()

    with tabs[0]:
        st.markdown("#### Channel Performance Overview")
        if channel_df.empty:
            st.info("No rows in im.v_channel_summary.")
        else:
            cdf = channel_df.copy()
            for col in [
                "cases",
                "avg_cycle_hours",
                "p90_cycle_hours",
                "avg_csat",
                "met_sla_rate",
                "reopen_rate",
                "reject_rate",
                "avg_escalations",
                "avg_resolver_changes",
                "feedback_rate",
                "missing_feedback_rate",
            ]:
                cdf[col] = pd.to_numeric(cdf[col], errors="coerce")

            cycle_best = cdf.sort_values("avg_cycle_hours", ascending=True).iloc[0]
            cycle_worst = cdf.sort_values("avg_cycle_hours", ascending=False).iloc[0]
            row_metrics = st.columns(2)
            with row_metrics[0]:
                metric_tile(
                    "Best Cycle Channel",
                    str(cycle_best["report_channel"]),
                    f"{fmt_float(cycle_best['avg_cycle_hours'])} hrs avg cycle",
                )
            with row_metrics[1]:
                metric_tile(
                    "Worst Cycle Channel",
                    str(cycle_worst["report_channel"]),
                    f"{fmt_float(cycle_worst['avg_cycle_hours'])} hrs avg cycle",
                )

            top_channels = cdf.sort_values("cases", ascending=False).head(20).copy()
            top_channels["cases"] = top_channels["cases"].astype("Int64")
            table_cols = [
                "report_channel",
                "cases",
                "avg_cycle_hours",
                "p90_cycle_hours",
                "avg_csat",
                "met_sla_rate",
                "reopen_rate",
                "reject_rate",
                "feedback_rate",
                "missing_feedback_rate",
            ]
            st.dataframe(
                style_table(
                    top_channels[table_cols],
                    pct_cols={"met_sla_rate", "reopen_rate", "reject_rate", "feedback_rate", "missing_feedback_rate"},
                    hour_cols={"avg_cycle_hours", "p90_cycle_hours"},
                    good_high_cols={"met_sla_rate", "avg_csat", "feedback_rate", "cases"},
                    bad_high_cols={"avg_cycle_hours", "p90_cycle_hours", "reject_rate", "reopen_rate", "missing_feedback_rate"},
                ),
                use_container_width=True,
                hide_index=True,
            )

            cycle_col, sla_col = st.columns(2)
            with cycle_col:
                fig_cycle = px.bar(cdf.sort_values("avg_cycle_hours", ascending=False), x="report_channel", y="avg_cycle_hours", text_auto=".2f")
                fig_cycle.update_traces(hovertemplate="Channel=%{x}<br>Avg Cycle=%{y:.2f} hrs<extra></extra>")
                st.plotly_chart(
                    clayout(
                        fig_cycle,
                        title="Avg Cycle Hours by Channel",
                        xtitle="Report Channel",
                        ytitle="Avg Cycle (hrs)",
                        h=420,
                    ),
                    use_container_width=True,
                    key="ch_overview_avg_cycle",
                )

            with sla_col:
                fig_sla = px.bar(cdf.sort_values("met_sla_rate", ascending=False), x="report_channel", y="met_sla_rate", text_auto=".1%")
                fig_sla.update_traces(hovertemplate="Channel=%{x}<br>SLA Met=%{y:.1%}<extra></extra>")
                st.plotly_chart(
                    clayout(
                        fig_sla,
                        title="SLA Met Rate by Channel",
                        xtitle="Report Channel",
                        ytitle="Rate",
                        h=420,
                    ),
                    use_container_width=True,
                    key="ch_overview_sla_met",
                )

    with tabs[1]:
        st.markdown("#### Channel Comparison")
        if channel_df.empty:
            st.info("No rows in im.v_channel_summary.")
        else:
            cdf = channel_df.copy()
            for col in [
                "cases",
                "avg_cycle_hours",
                "p90_cycle_hours",
                "avg_csat",
                "met_sla_rate",
                "reopen_rate",
                "reject_rate",
                "avg_escalations",
                "avg_resolver_changes",
                "feedback_rate",
                "missing_feedback_rate",
            ]:
                cdf[col] = pd.to_numeric(cdf[col], errors="coerce")

            rank_metric = st.selectbox(
                "Rank channels by",
                options=["cases", "avg_cycle_hours", "met_sla_rate", "avg_csat", "missing_feedback_rate"],
                index=0,
                key="ci_rank_metric",
            )
            ascending = rank_metric in {"avg_cycle_hours", "missing_feedback_rate"}
            ranked = cdf.sort_values(rank_metric, ascending=ascending)

            table_cols = [
                "report_channel",
                "cases",
                "avg_cycle_hours",
                "p90_cycle_hours",
                "avg_csat",
                "met_sla_rate",
                "reopen_rate",
                "reject_rate",
                "feedback_rate",
                "missing_feedback_rate",
                "avg_escalations",
                "avg_resolver_changes",
            ]
            top20 = ranked[table_cols].head(20).copy()
            top20["cases"] = top20["cases"].astype("Int64")
            st.dataframe(
                style_table(
                    top20,
                    pct_cols={"met_sla_rate", "reopen_rate", "reject_rate", "feedback_rate", "missing_feedback_rate"},
                    hour_cols={"avg_cycle_hours", "p90_cycle_hours"},
                    good_high_cols={"met_sla_rate", "avg_csat", "feedback_rate", "cases"},
                    bad_high_cols={"avg_cycle_hours", "p90_cycle_hours", "reject_rate", "reopen_rate", "missing_feedback_rate"},
                ),
                use_container_width=True,
                hide_index=True,
            )

            compare_left, compare_right = st.columns(2)
            with compare_left:
                fig_cycle = px.bar(
                    cdf.sort_values("avg_cycle_hours", ascending=False),
                    x="report_channel",
                    y="avg_cycle_hours",
                    text_auto=".2f",
                )
                fig_cycle.update_traces(
                    hovertemplate="Channel=%{x}<br>Avg Cycle=%{y:.2f} hrs<extra></extra>",
                )
                st.plotly_chart(
                    clayout(
                        fig_cycle,
                        title="Avg Cycle Hours by Channel",
                        xtitle="Report Channel",
                        ytitle="Avg Cycle (hrs)",
                        h=420,
                    ),
                    use_container_width=True,
                    key="ch_compare_avg_cycle",
                )

            with compare_right:
                fig_sla = px.bar(
                    cdf.sort_values("met_sla_rate", ascending=False),
                    x="report_channel",
                    y="met_sla_rate",
                    text_auto=".1%",
                )
                fig_sla.update_traces(
                    hovertemplate="Channel=%{x}<br>SLA Met=%{y:.1%}<extra></extra>",
                )
                st.plotly_chart(
                    clayout(
                        fig_sla,
                        title="SLA Met Rate by Channel",
                        xtitle="Report Channel",
                        ytitle="Rate",
                        h=420,
                    ),
                    use_container_width=True,
                    key="ch_compare_sla_met",
                )

            extra_left, extra_right = st.columns(2)
            with extra_left:
                fig_feedback = px.bar(
                    cdf.sort_values("missing_feedback_rate", ascending=False),
                    x="report_channel",
                    y="missing_feedback_rate",
                    text_auto=".1%",
                )
                fig_feedback.update_traces(
                    hovertemplate="Channel=%{x}<br>Missing Feedback=%{y:.1%}<extra></extra>",
                )
                st.plotly_chart(
                    clayout(
                        fig_feedback,
                        title="Missing Feedback Rate by Channel",
                        xtitle="Report Channel",
                        ytitle="Rate",
                        h=420,
                    ),
                    use_container_width=True,
                    key="ch_compare_missing_feedback",
                )

            with extra_right:
                fig_scatter = px.scatter(
                    cdf,
                    x="avg_cycle_hours",
                    y="avg_csat",
                    color="report_channel",
                    hover_name="report_channel",
                )
                fig_scatter.update_traces(
                    marker=dict(size=12, line=dict(width=1, color="rgba(250,250,250,0.3)"), opacity=0.85),
                    hovertemplate="Channel=%{hovertext}<br>Avg Cycle=%{x:.2f} hrs<br>Avg CSAT=%{y:.2f}<extra></extra>",
                )
                st.plotly_chart(
                    clayout(
                        fig_scatter,
                        title="Avg Cycle vs Avg CSAT by Channel",
                        xtitle="Avg Cycle (hrs)",
                        ytitle="Avg CSAT",
                        h=420,
                    ),
                    use_container_width=True,
                    key="ch_compare_cycle_vs_csat",
                )

    with tabs[2]:
        st.markdown("#### Bad Channel + Issue Combos")
        if channel_issue_df.empty:
            st.info("No rows in im.v_channel_issue_summary.")
            filtered_bad_combos_for_export = pd.DataFrame()
        else:
            cidf = channel_issue_df.copy()
            for col in [
                "cases",
                "avg_cycle_hours",
                "p90_cycle_hours",
                "avg_csat",
                "met_sla_rate",
                "reopen_rate",
                "reject_rate",
                "avg_escalations",
                "avg_resolver_changes",
                "feedback_rate",
                "missing_feedback_rate",
            ]:
                cidf[col] = pd.to_numeric(cidf[col], errors="coerce")

            fcols = st.columns([1.1, 1.2, 1.0])
            with fcols[0]:
                channel_opts = sorted([str(v) for v in cidf["report_channel"].dropna().unique().tolist()])
                selected_channels = st.multiselect(
                    "Channel", options=channel_opts, default=channel_opts, key="ci_bad_channel"
                )
            with fcols[1]:
                issue_opts = sorted([str(v) for v in cidf["issue_type"].dropna().unique().tolist()])
                selected_issues = st.multiselect(
                    "Issue Type", options=issue_opts, default=issue_opts, key="ci_bad_issue"
                )
            with fcols[2]:
                max_cases = int(cidf["cases"].dropna().max()) if not cidf["cases"].dropna().empty else 25
                min_cases = st.slider(
                    "Min Cases",
                    min_value=1,
                    max_value=max(max_cases, 1),
                    value=min(25, max_cases),
                    key="ci_bad_min_cases",
                )

            filtered = cidf.copy()
            if selected_channels:
                filtered = filtered[filtered["report_channel"].astype(str).isin(selected_channels)]
            else:
                filtered = filtered.iloc[0:0]
            if selected_issues:
                filtered = filtered[filtered["issue_type"].astype(str).isin(selected_issues)]
            else:
                filtered = filtered.iloc[0:0]
            filtered = filtered[filtered["cases"] >= min_cases]

            filtered["pain_score"] = (
                filtered["cases"]
                * (
                    (1.0 - filtered["met_sla_rate"].fillna(0.0))
                    + filtered["reject_rate"].fillna(0.0)
                    + filtered["reopen_rate"].fillna(0.0)
                    + filtered["missing_feedback_rate"].fillna(0.0)
                )
                + filtered["cases"] * (filtered["avg_cycle_hours"].fillna(0.0) / 10.0)
            )

            top25 = filtered.sort_values("pain_score", ascending=False).head(25).copy()
            filtered_bad_combos_for_export = top25.copy()

            bad_left, bad_right = st.columns([1.25, 1.0])
            with bad_left:
                table_cols = [
                    "report_channel",
                    "issue_type",
                    "cases",
                    "pain_score",
                    "avg_cycle_hours",
                    "p90_cycle_hours",
                    "avg_csat",
                    "met_sla_rate",
                    "reopen_rate",
                    "reject_rate",
                    "missing_feedback_rate",
                ]
                table_df = top25[table_cols].copy()
                if not table_df.empty:
                    table_df["cases"] = table_df["cases"].astype("Int64")
                st.dataframe(
                    style_table(
                        table_df,
                        pct_cols={"met_sla_rate", "reopen_rate", "reject_rate", "missing_feedback_rate"},
                        hour_cols={"avg_cycle_hours", "p90_cycle_hours"},
                        good_high_cols={"met_sla_rate", "avg_csat", "cases"},
                        bad_high_cols={"pain_score", "avg_cycle_hours", "p90_cycle_hours", "reopen_rate", "reject_rate", "missing_feedback_rate"},
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

                st.download_button(
                    "Download filtered bad combos CSV",
                    data=top25.to_csv(index=False).encode("utf-8"),
                    file_name="channel_bad_combos_filtered.csv",
                    mime="text/csv",
                    key="dl_ci_bad_combos_filtered",
                )

            with bad_right:
                if top25.empty:
                    st.info("No rows match current bad-combo filters.")
                else:
                    chart_df = top25.head(10).copy()
                    chart_df["combo_label"] = chart_df["report_channel"].astype(str) + " | " + chart_df["issue_type"].astype(str)
                    chart_df = chart_df.sort_values("pain_score", ascending=True)
                    fig_pain = px.bar(
                        chart_df,
                        x="pain_score",
                        y="combo_label",
                        orientation="h",
                        text_auto=".2f",
                    )
                    fig_pain.update_traces(
                        hovertemplate="Combo=%{y}<br>Pain Score=%{x:.2f}<extra></extra>",
                    )
                    st.plotly_chart(
                        clayout(
                            fig_pain,
                            title="Top 10 Pain Scores (Channel | Issue)",
                            xtitle="Pain Score",
                            ytitle="Channel | Issue",
                            h=520,
                        ),
                        use_container_width=True,
                        key="ch_bad_combos_pain",
                    )

            if not top25.empty:
                worst_combo = top25.iloc[0]
                st.info(
                    "Worst combo by pain_score: "
                    f"{worst_combo['report_channel']} | {worst_combo['issue_type']} "
                    f"(pain={fmt_float(worst_combo['pain_score'])}, cases={fmt_int(worst_combo['cases'])})."
                )
                missing_pool = top25[top25["cases"] >= min_cases].dropna(subset=["missing_feedback_rate"])
                if not missing_pool.empty:
                    worst_missing = missing_pool.sort_values("missing_feedback_rate", ascending=False).iloc[0]
                    st.info(
                        "Worst missing-feedback combo: "
                        f"{worst_missing['report_channel']} | {worst_missing['issue_type']} "
                        f"(missing_feedback={fmt_pct(worst_missing['missing_feedback_rate'])}, cases={fmt_int(worst_missing['cases'])})."
                    )

    with tabs[3]:
        st.markdown("#### Export")
        if channel_df.empty:
            st.info("No channel summary rows to export.")
        else:
            st.download_button(
                "Download v_channel_summary CSV",
                data=channel_df.to_csv(index=False).encode("utf-8"),
                file_name="v_channel_summary.csv",
                mime="text/csv",
                key="dl_ci_channel_summary",
            )

        if filtered_bad_combos_for_export.empty:
            st.info("No filtered bad combos available yet. Set filters in Bad Combos tab first.")
        else:
            st.download_button(
                "Download filtered bad combos CSV",
                data=filtered_bad_combos_for_export.to_csv(index=False).encode("utf-8"),
                file_name="channel_bad_combos_filtered.csv",
                mime="text/csv",
                key="dl_ci_bad_combos_export",
            )

        st.write(
            "Use these outputs to prioritize intake form improvements, routing-rule tuning, support-team training, "
            "and channel nudges that reduce avoidable friction."
        )

    with st.expander("Data Notes"):
        st.write(
            "Channel comparisons reflect both process performance and data quality, not individual performance."
        )
def render_knowledge_fcr(db_url: str) -> None:
    header("Knowledge & FCR", "Analyze FCR proxy performance and prioritize knowledge/training enablement opportunities.")
    ui_explainer(
        "How to read this page",
        what=[
            "Tracks FCR proxy, resolution-level mix, and enablement backlog by issue type.",
            "Highlights where knowledge/training could increase L1 resolution.",
        ],
        why=[
            "Higher L1 resolution typically reduces escalations, reopens, and overall cycle time.",
            "Improves customer outcomes while reducing operational load.",
        ],
        how=[
            "Start with Enablement Backlog to prioritize issue types.",
            "Use Drilldown evidence to plan training/KB updates.",
            "Export filtered case evidence for implementation tracking.",
        ],
    )

    try:
        overview_df = run_query(FCR_OVERVIEW_SQL)
        level_dist_df = run_query(FCR_LEVEL_DIST_SQL)
        fcr_summary_df = run_query(FCR_SUMMARY_SQL)
        kb_df = run_query(KB_ENABLEMENT_SQL)
    except Exception as exc:
        st.error(f"Failed to query knowledge/FCR views: {exc}")
        st.stop()

    default_issue_type: Optional[str] = None
    if not kb_df.empty:
        kb_sorted_for_default = kb_df.copy()
        kb_sorted_for_default["enablement_score"] = pd.to_numeric(
            kb_sorted_for_default["enablement_score"], errors="coerce"
        )
        kb_sorted_for_default = kb_sorted_for_default.sort_values("enablement_score", ascending=False)
        if not kb_sorted_for_default.empty:
            default_issue_type = str(kb_sorted_for_default.iloc[0]["issue_type"])
    elif not fcr_summary_df.empty:
        fcr_sorted_for_default = fcr_summary_df.copy()
        fcr_sorted_for_default["cases"] = pd.to_numeric(fcr_sorted_for_default["cases"], errors="coerce")
        fcr_sorted_for_default = fcr_sorted_for_default.sort_values("cases", ascending=False)
        if not fcr_sorted_for_default.empty:
            default_issue_type = str(fcr_sorted_for_default.iloc[0]["issue_type"])

    if default_issue_type and "kfcr_issue_type" not in st.session_state:
        st.session_state["kfcr_issue_type"] = default_issue_type

    tabs = st.tabs(["Overview", "Enablement Backlog", "Drilldown", "Export"])

    with tabs[0]:
        st.markdown("#### FCR Proxy Overview")
        if overview_df.empty:
            st.info("No rows in im.v_fcr_cases.")
        else:
            k = overview_df.iloc[0]

            row1 = st.columns(4)
            row2 = st.columns(4)
            with row1[0]:
                metric_tile("FCR %", fmt_pct(k["fcr_rate"]))
            with row1[1]:
                metric_tile("L1 Resolution %", fmt_pct(k["l1_rate"]))
            with row1[2]:
                metric_tile("L2 Resolution %", fmt_pct(k["l2_rate"]))
            with row1[3]:
                metric_tile("L3 Resolution %", fmt_pct(k["l3_rate"]))
            with row2[0]:
                metric_tile("Avg Cycle (hrs)", fmt_float(k["avg_cycle_hours"]))
            with row2[1]:
                metric_tile("SLA Met %", fmt_pct(k["met_sla_rate"]))
            with row2[2]:
                metric_tile("Avg CSAT", fmt_float(k["avg_csat"]))
            with row2[3]:
                metric_tile("Cases", fmt_int(k["cases"]))

            st.divider()
            chart_cols = st.columns(2)
            with chart_cols[0]:
                if level_dist_df.empty:
                    st.info("No resolution-level distribution rows.")
                else:
                    dist = level_dist_df.copy()
                    dist["cases"] = pd.to_numeric(dist["cases"], errors="coerce")
                    fig_dist = px.pie(dist, values="cases", names="resolution_level", hole=0.35)
                    fig_dist.update_traces(
                        textinfo="percent+label",
                        hovertemplate="Level=%{label}<br>Cases=%{value}<br>Share=%{percent}<extra></extra>",
                    )
                    st.plotly_chart(
                        clayout(
                            fig_dist,
                            title="Resolution Level Distribution",
                            h=430,
                        ),
                        use_container_width=True,
                    )

            with chart_cols[1]:
                if fcr_summary_df.empty:
                    st.info("No rows in im.v_fcr_summary.")
                else:
                    fs = fcr_summary_df.copy()
                    fs["fcr_rate"] = pd.to_numeric(fs["fcr_rate"], errors="coerce")
                    fs["cases"] = pd.to_numeric(fs["cases"], errors="coerce")
                    fs = fs.sort_values("fcr_rate", ascending=False)
                    fig_fcr = px.bar(fs, x="issue_type", y="fcr_rate", text_auto=".1%")
                    fig_fcr.update_traces(
                        hovertemplate="Issue Type=%{x}<br>FCR Rate=%{y:.1%}<extra></extra>",
                    )
                    st.plotly_chart(
                        clayout(
                            fig_fcr,
                            title="FCR Rate by Issue Type",
                            xtitle="Issue Type",
                            ytitle="FCR Rate",
                            h=430,
                        ),
                        use_container_width=True,
                    )

            top_opportunity = None
            if not kb_df.empty:
                kb_check = kb_df.copy()
                kb_check["enablement_score"] = pd.to_numeric(kb_check["enablement_score"], errors="coerce")
                kb_check = kb_check.sort_values("enablement_score", ascending=False)
                if not kb_check.empty:
                    top_opportunity = kb_check.iloc[0]
            if top_opportunity is not None:
                st.info(
                    f"Overall FCR is {fmt_pct(k['fcr_rate'])}. Top issue_type opportunity: "
                    f"{top_opportunity['issue_type']} (enablement_score={fmt_float(top_opportunity['enablement_score'])})."
                )

    with tabs[1]:
        st.markdown("#### Knowledge Enablement Backlog")
        if kb_df.empty:
            st.info("No rows in im.v_kb_enablement_candidates.")
        else:
            kb = kb_df.copy()
            for col in [
                "cases",
                "l1_solved_rate",
                "l2_or_l3_rate",
                "escalation_to_l2_rate",
                "reopen_rate",
                "avg_cycle_hours",
                "avg_csat",
                "enablement_score",
            ]:
                kb[col] = pd.to_numeric(kb[col], errors="coerce")
            kb = kb.sort_values("enablement_score", ascending=False)

            table_cols = [
                "issue_type",
                "cases",
                "enablement_score",
                "l1_solved_rate",
                "l2_or_l3_rate",
                "escalation_to_l2_rate",
                "reopen_rate",
                "avg_cycle_hours",
                "avg_csat",
            ]
            table_df = kb[table_cols].head(30).copy()
            table_df["cases"] = table_df["cases"].astype("Int64")
            st.dataframe(
                style_table(
                    table_df,
                    pct_cols={"l1_solved_rate", "l2_or_l3_rate", "escalation_to_l2_rate", "reopen_rate"},
                    hour_cols={"avg_cycle_hours"},
                    good_high_cols={"l1_solved_rate", "avg_csat", "cases"},
                    bad_high_cols={"enablement_score", "l2_or_l3_rate", "escalation_to_l2_rate", "reopen_rate"},
                ),
                use_container_width=True,
                hide_index=True,
            )

            top10 = kb.head(10).copy().sort_values("enablement_score", ascending=True)
            fig_enable = px.bar(
                top10,
                x="enablement_score",
                y="issue_type",
                orientation="h",
                text_auto=".2f",
            )
            fig_enable.update_traces(
                hovertemplate="Issue Type=%{y}<br>Enablement Score=%{x:.2f}<extra></extra>",
            )
            st.plotly_chart(
                clayout(
                    fig_enable,
                    title="Top Enablement Scores by Issue Type",
                    xtitle="Enablement Score",
                    ytitle="Issue Type",
                    h=500,
                ),
                use_container_width=True,
            )

    with tabs[2]:
        st.markdown("#### Issue-Type Drilldown")
        issue_options: list[str] = []
        if not kb_df.empty:
            issue_options = sorted([str(v) for v in kb_df["issue_type"].dropna().unique().tolist()])
        elif not fcr_summary_df.empty:
            issue_options = sorted([str(v) for v in fcr_summary_df["issue_type"].dropna().unique().tolist()])

        if not issue_options:
            st.info("No issue_type values available for drilldown.")
        else:
            default_index = 0
            current_issue = st.session_state.get("kfcr_issue_type")
            if current_issue in issue_options:
                default_index = issue_options.index(current_issue)

            selected_issue = st.selectbox(
                "Issue Type",
                options=issue_options,
                index=default_index,
                key="kfcr_issue_type",
            )

            try:
                drill_df = run_query(
                    FCR_CASES_BY_ISSUE_SQL,
                    params=(selected_issue,),
                )
            except Exception as exc:
                st.error(f"Failed to query FCR drilldown cases: {exc}")
                st.stop()

            if drill_df.empty:
                st.info("No FCR case rows for selected issue type.")
            else:
                drill_df["cycle_hours"] = pd.to_numeric(drill_df["cycle_hours"], errors="coerce")
                drill_df["customer_satisfaction"] = pd.to_numeric(drill_df["customer_satisfaction"], errors="coerce")
                drill_df["fcr"] = drill_df["fcr"].fillna(False).astype(bool)

                escalated_cases = drill_df[
                    (drill_df["resolution_level"] != "L1") | (drill_df["fcr"] == False)
                ].sort_values("cycle_hours", ascending=False).head(200)
                fcr_wins = drill_df[drill_df["fcr"] == True].sort_values("cycle_hours", ascending=False).head(200)

                split_left, split_right = st.columns(2)
                with split_left:
                    st.markdown("##### Escalated / Not FCR (Top 200)")
                    esc_cols = [
                        "case_id",
                        "priority",
                        "variant",
                        "report_channel",
                        "cycle_hours",
                        "customer_satisfaction",
                        "met_sla",
                        "resolution_level",
                        "fcr",
                    ]
                    st.dataframe(
                        style_table(
                            escalated_cases[esc_cols],
                            hour_cols={"cycle_hours"},
                            good_high_cols={"met_sla"},
                            bad_high_cols={"cycle_hours"},
                            bad_low_cols={"customer_satisfaction", "met_sla"},
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

                with split_right:
                    st.markdown("##### FCR Wins (Top 200)")
                    win_cols = [
                        "case_id",
                        "priority",
                        "variant",
                        "report_channel",
                        "cycle_hours",
                        "customer_satisfaction",
                        "met_sla",
                        "resolution_level",
                        "fcr",
                    ]
                    st.dataframe(
                        style_table(
                            fcr_wins[win_cols],
                            hour_cols={"cycle_hours"},
                            good_high_cols={"met_sla", "fcr"},
                            bad_high_cols={"cycle_hours"},
                            bad_low_cols={"customer_satisfaction", "met_sla"},
                        ),
                        use_container_width=True,
                        hide_index=True,
                    )

                st.download_button(
                    "Download escalated cases CSV",
                    data=escalated_cases.to_csv(index=False).encode("utf-8"),
                    file_name=f"fcr_escalated_cases_{selected_issue}.csv",
                    mime="text/csv",
                    key="dl_kfcr_escalated_cases",
                )
                st.download_button(
                    "Download FCR wins CSV",
                    data=fcr_wins.to_csv(index=False).encode("utf-8"),
                    file_name=f"fcr_wins_{selected_issue}.csv",
                    mime="text/csv",
                    key="dl_kfcr_fcr_wins",
                )

                hist_df = drill_df.dropna(subset=["cycle_hours"]).copy()
                if hist_df.empty:
                    st.info("Not enough cycle-hour values for issue-type histogram.")
                else:
                    fig_hist = px.histogram(hist_df, x="cycle_hours", nbins=20)
                    fig_hist.update_traces(
                        hovertemplate="Cycle Hours=%{x:.2f}<br>Cases=%{y}<extra></extra>",
                    )
                    st.plotly_chart(
                        clayout(
                            fig_hist,
                            title=f"Cycle Hours Distribution: {selected_issue}",
                            xtitle="Cycle Hours",
                            ytitle="Cases",
                            h=430,
                        ),
                        use_container_width=True,
                    )

    with tabs[3]:
        st.markdown("#### Export")
        if kb_df.empty:
            st.info("No rows in im.v_kb_enablement_candidates to export.")
        else:
            st.download_button(
                "Download v_kb_enablement_candidates CSV",
                data=kb_df.to_csv(index=False).encode("utf-8"),
                file_name="v_kb_enablement_candidates.csv",
                mime="text/csv",
                key="dl_kfcr_kb_enablement",
            )

        if fcr_summary_df.empty:
            st.info("No rows in im.v_fcr_summary to export.")
        else:
            st.download_button(
                "Download v_fcr_summary CSV",
                data=fcr_summary_df.to_csv(index=False).encode("utf-8"),
                file_name="v_fcr_summary.csv",
                mime="text/csv",
                key="dl_kfcr_fcr_summary",
            )

        selected_issue_for_export = st.session_state.get("kfcr_issue_type", default_issue_type)
        if selected_issue_for_export:
            try:
                export_cases_df = run_query(
                    FCR_CASES_BY_ISSUE_SQL,
                    params=(selected_issue_for_export,),
                )
            except Exception as exc:
                st.error(f"Failed to query filtered FCR cases for export: {exc}")
                st.stop()

            st.download_button(
                f"Download v_fcr_cases CSV ({selected_issue_for_export})",
                data=export_cases_df.to_csv(index=False).encode("utf-8"),
                file_name=f"v_fcr_cases_{selected_issue_for_export}.csv",
                mime="text/csv",
                key="dl_kfcr_filtered_cases",
            )
            st.caption("Filtered v_fcr_cases export can be large depending on issue-type volume.")
        else:
            st.info("No issue type selected for filtered v_fcr_cases export.")

    with st.expander("Data Notes"):
        st.write(
            "FCR proxy is defined as solved by L1 with no escalation to L2/L3 and no reopen. "
            "Use these metrics to prioritize knowledge and training investments."
        )
def main() -> None:
    st.markdown(
        f"""
        <style>
          .page-header {{
            background: linear-gradient(90deg, {P_DARK_BLUE} 0%, {P_MID_BLUE} 100%);
            border-radius: 14px;
            padding: 18px 20px;
            margin: 0 0 16px 0;
          }}
          .page-header h1 {{
            color: {P_TEXT_DARK};
            font-size: 1.9rem;
            font-weight: 700;
            margin: 0;
            line-height: 1.2;
          }}
          .page-header p {{
            color: {P_TEXT_MED};
            font-size: 0.96rem;
            margin: 6px 0 0 0;
            line-height: 1.35;
          }}
          .sidebar-logo {{
            background: linear-gradient(135deg, {P_DARK_BLUE} 0%, {P_MID_BLUE} 100%);
            border-radius: 14px;
            padding: 14px 14px 12px 14px;
            margin: 4px 0 12px 0;
          }}
          .sidebar-logo h2 {{
            color: {P_TEXT_DARK};
            font-size: 1.25rem;
            font-weight: 700;
            margin: 0;
            line-height: 1.2;
          }}
          .sidebar-logo p  {{
            color: {P_TEXT_MED};
            font-size: 0.84rem;
            margin: 5px 0 0 0;
            line-height: 1.25;
          }}
          .info-pill {{
            display: inline-block;
            padding: 0.2rem 0.55rem;
            border-radius: 999px;
            font-size: 0.78rem;
            font-weight: 600;
            color: {P_TEXT_DARK};
            background: rgba(77,182,172,0.26);
            border: 1px solid rgba(77,182,172,0.55);
          }}
          .divider {{
            height: 1px;
            width: 100%;
            margin: 0.6rem 0 1rem 0;
            background: linear-gradient(90deg, rgba(255,255,255,0.32), rgba(255,255,255,0.08));
          }}
          [data-testid="stMetricLabel"] {{
            color: {P_TEXT_MED};
            font-size: 0.86rem;
            letter-spacing: 0.2px;
          }}
          [data-testid="stMetricValue"] {{
            color: {P_TEXT_DARK};
            font-size: 1.62rem;
            font-weight: 700;
          }}
          .io-card {{
            border-left: 5px solid {COLOR_TEAL};
            border-radius: 10px;
            padding: 12px 14px;
            min-height: 96px;
          }}
          .io-card-label {{
            color: #C7D0D9;
            font-size: 0.86rem;
            letter-spacing: 0.2px;
            margin-bottom: 4px;
          }}
          .io-card-value {{
            color: {P_TEXT_DARK};
            font-size: 1.55rem;
            font-weight: 700;
            line-height: 1.1;
          }}
          .io-card-sub {{
            color: #9AA7B2;
            font-size: 0.78rem;
            margin-top: 6px;
          }}
          .stDataFrame {{
            border: 1px solid rgba(77,182,172,0.18);
            border-radius: 8px;
          }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-logo">
              <h2>IncidentOps</h2>
              <p>Process Mining Insights Hub</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("##### Navigate")
        page = st.radio(
            "Navigation",
            [
                "Executive Overview",
                "Process Explorer",
                "Bottlenecks",
                "Escalations & Handoffs",
                "Channel & Intake",
                "Knowledge & FCR",
                "Quality & CX",
                "Problem Candidates",
            ],
            label_visibility="collapsed",
        )

    db_url = get_db_url()
    if not db_url:
        st.warning(
            "Database config missing. Set DATABASE_URL_POOLED (preferred) or DATABASE_URL_DIRECT in .env or Streamlit secrets."
        )
        st.stop()

    if page == "Executive Overview":
        render_executive_overview(db_url)
    elif page == "Process Explorer":
        render_process_explorer(db_url)
    elif page == "Bottlenecks":
        render_bottlenecks(db_url)
    elif page == "Escalations & Handoffs":
        render_escalations_handoffs(db_url)
    elif page == "Channel & Intake":
        render_channel_intake(db_url)
    elif page == "Knowledge & FCR":
        render_knowledge_fcr(db_url)
    elif page == "Quality & CX":
        render_quality_cx(db_url)
    elif page == "Problem Candidates":
        render_problem_candidates(db_url)
    else:
        render_coming_soon(page)


if __name__ == "__main__":
    main()





