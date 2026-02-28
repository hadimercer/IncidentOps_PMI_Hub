from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import psycopg2
import streamlit as st
from dotenv import load_dotenv

ST_BG = "#0E1117"
ST_TEXT = "#FAFAFA"
ACCENT = "#4DB6AC"
CARD_BG = "#262730"


st.set_page_config(
    page_title="IncidentOps | PMI Hub",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    f"""
    <style>
      .io-header {{
        border-left: 6px solid {ACCENT};
        border-radius: 12px;
        padding: 18px 18px 16px 18px;
        margin: 0 0 16px 0;
        background: linear-gradient(90deg, rgba(77,182,172,0.20), rgba(38,39,48,0.92));
      }}
      .io-title {{
        color: {ST_TEXT};
        font-size: 1.65rem;
        font-weight: 700;
        margin: 0;
        line-height: 1.2;
      }}
      .io-subtitle {{
        color: #C7D0D9;
        font-size: 0.95rem;
        margin-top: 6px;
      }}
      .io-card {{
        border-left: 5px solid {ACCENT};
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
        color: {ST_TEXT};
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
        <div class="io-header">
          <div class="io-title">{title}</div>
          <div class="io-subtitle">{subtitle}</div>
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


def load_db_url() -> Optional[str]:
    load_dotenv(dotenv_path=Path(__file__).resolve().parent / ".env")
    return os.getenv("DATABASE_URL_POOLED") or os.getenv("DATABASE_URL_DIRECT")


@st.cache_resource(show_spinner=False)
def get_connection(db_url: str):
    return psycopg2.connect(db_url)


@st.cache_data(ttl=300, show_spinner=False)
def run_query(db_url: str, sql: str) -> pd.DataFrame:
    conn = get_connection(db_url)
    return pd.read_sql_query(sql, conn)


def render_coming_soon(page_name: str) -> None:
    header(page_name, "This module is planned next.")
    st.info("Coming soon. Executive Overview is currently active.")


def render_executive_overview(db_url: str) -> None:
    header("Executive Overview", "Cross-case operational KPIs and variant performance.")

    try:
        kpi_df = run_query(db_url, KPI_SQL)
        variant_df = run_query(db_url, VARIANT_SQL)
        range_df = run_query(db_url, RANGE_SQL)
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
            text_auto=True,
        )
        fig_cases.update_traces(marker_color=ACCENT, hovertemplate="Variant=%{x}<br>Cases=%{y}<extra></extra>")
        st.plotly_chart(
            clayout(fig_cases, title="Cases by Variant", xtitle="Variant", ytitle="Cases"),
            use_container_width=True,
        )

        fig_cycle = px.bar(
            plot_df,
            x="variant",
            y="avg_cycle_hours",
            text_auto=".2f",
        )
        fig_cycle.update_traces(
            marker_color="#7FE0D6",
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

    try:
        variant_df = run_query(db_url, VARIANT_SQL)
        transition_df = run_query(db_url, TRANSITION_SUMMARY_SQL)
        variant_transition_df = run_query(db_url, TRANSITION_BY_VARIANT_SQL)
        dwell_df = run_query(db_url, DWELL_SQL)
    except Exception as exc:
        st.error(f"Failed to query process explorer views: {exc}")
        st.stop()

    st.markdown("#### Variant Leaderboard")
    selected_variant: Optional[str] = None
    if variant_df.empty:
        st.info("No rows in im.v_variant_summary.")
    else:
        variants_sorted = variant_df.sort_values("cases", ascending=False).copy()
        variant_options = [
            str(v) for v in variants_sorted["variant"].tolist() if pd.notna(v) and str(v).strip() != ""
        ]
        if variant_options:
            selected_variant = st.selectbox("Variant drilldown", options=variant_options, index=0)
        else:
            st.info("No variant values available for drilldown.")

        display_variants = variants_sorted.copy()
        display_variants["cases"] = pd.to_numeric(display_variants["cases"], errors="coerce").astype("Int64")
        for col in ["avg_cycle_hours", "p90_cycle_hours", "avg_csat", "avg_escalations", "avg_resolver_changes"]:
            display_variants[col] = display_variants[col].map(lambda v: fmt_float(v, 2))
        for col in ["met_sla_rate", "reopen_rate", "reject_rate"]:
            display_variants[col] = display_variants[col].map(lambda v: fmt_pct(v, 1))

        st.dataframe(display_variants, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("#### Top Transitions (Overall)")
    if transition_df.empty:
        st.info("No rows in im.v_transition_summary.")
    else:
        transition_df = transition_df.copy()
        transition_df["transition_count"] = pd.to_numeric(transition_df["transition_count"], errors="coerce")
        for col in ["avg_delta_hours", "median_delta_hours", "p90_delta_hours"]:
            transition_df[col] = pd.to_numeric(transition_df[col], errors="coerce")
        transition_df = transition_df.dropna(subset=["transition_count"])
        transition_df = transition_df.sort_values("transition_count", ascending=False)

        col_table, col_sankey = st.columns([1.1, 1.2])

        with col_table:
            top20 = transition_df.head(20).copy()
            top20_display = top20.copy()
            for col in ["avg_delta_hours", "median_delta_hours", "p90_delta_hours"]:
                top20_display[col] = top20_display[col].map(lambda v: fmt_float(v, 2))
            top20_display["transition_count"] = top20_display["transition_count"].map(fmt_int)
            st.dataframe(top20_display, use_container_width=True, hide_index=True)

        with col_sankey:
            fig_overall = build_sankey(
                transitions=transition_df,
                title="Overall Flow Sankey (Top 25 Transitions)",
                max_rows=25,
                height=540,
            )
            if fig_overall is None:
                st.info("Not enough transition data to draw overall Sankey.")
            else:
                st.plotly_chart(fig_overall, use_container_width=True)

    st.divider()
    st.markdown("#### Variant Drilldown")
    if selected_variant is None:
        st.info("Select a variant in the leaderboard to view transition drilldown.")
    elif variant_transition_df.empty:
        st.info("No rows in im.v_transition_by_variant.")
    else:
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
            col_table_v, col_sankey_v = st.columns([1.1, 1.1])
            with col_table_v:
                top20_variant = selected_df.head(20).copy()
                top20_variant_display = top20_variant.copy()
                top20_variant_display["transition_count"] = top20_variant_display["transition_count"].map(fmt_int)
                top20_variant_display["avg_delta_hours"] = top20_variant_display["avg_delta_hours"].map(
                    lambda v: fmt_float(v, 2)
                )
                st.dataframe(
                    top20_variant_display[["from_event", "to_event", "transition_count", "avg_delta_hours"]],
                    use_container_width=True,
                    hide_index=True,
                )

            with col_sankey_v:
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
    st.markdown("#### Dwell Times")
    if dwell_df.empty:
        st.info("No rows in im.v_dwell_by_event.")
    else:
        dwell_df = dwell_df.copy()
        dwell_df["occurrences"] = pd.to_numeric(dwell_df["occurrences"], errors="coerce")
        for col in ["avg_dwell_hours", "median_dwell_hours", "p90_dwell_hours"]:
            dwell_df[col] = pd.to_numeric(dwell_df[col], errors="coerce")
        dwell_df = dwell_df.sort_values("avg_dwell_hours", ascending=False)

        col_dwell_table, col_dwell_chart = st.columns([1.1, 1.1])
        with col_dwell_table:
            dwell_display = dwell_df.copy()
            dwell_display["occurrences"] = dwell_display["occurrences"].map(fmt_int)
            for col in ["avg_dwell_hours", "median_dwell_hours", "p90_dwell_hours"]:
                dwell_display[col] = dwell_display[col].map(lambda v: fmt_float(v, 2))
            st.dataframe(dwell_display, use_container_width=True, hide_index=True)

        with col_dwell_chart:
            top10_dwell = dwell_df.head(10).sort_values("avg_dwell_hours", ascending=True)
            fig_dwell = px.bar(
                top10_dwell,
                x="avg_dwell_hours",
                y="event",
                orientation="h",
                text_auto=".2f",
            )
            fig_dwell.update_traces(
                marker_color=ACCENT,
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


def render_bottlenecks(db_url: str) -> None:
    header("Bottlenecks", "Identify where time is lost across steps and handoffs.")
    st.caption("Dwell time = time spent in a step before the next event.")
    st.caption("Transition time = delay between specific handoffs.")

    try:
        dwell_df = run_query(db_url, DWELL_SQL)
        transition_df = run_query(db_url, TRANSITION_SUMMARY_SQL)
    except Exception as exc:
        st.error(f"Failed to query bottleneck views: {exc}")
        st.stop()

    st.divider()
    st.markdown("#### Top Dwell Time Steps")
    if dwell_df.empty:
        st.info("No rows in im.v_dwell_by_event.")
    else:
        dwell_df = dwell_df.copy()
        dwell_df["occurrences"] = pd.to_numeric(dwell_df["occurrences"], errors="coerce")
        for col in ["avg_dwell_hours", "median_dwell_hours", "p90_dwell_hours"]:
            dwell_df[col] = pd.to_numeric(dwell_df[col], errors="coerce")
        dwell_df = dwell_df.sort_values("avg_dwell_hours", ascending=False)

        col_dwell_table, col_dwell_chart = st.columns([1.1, 1.1])
        with col_dwell_table:
            dwell_display = dwell_df.copy()
            dwell_display["occurrences"] = dwell_display["occurrences"].map(fmt_int)
            for col in ["avg_dwell_hours", "median_dwell_hours", "p90_dwell_hours"]:
                dwell_display[col] = dwell_display[col].map(lambda v: fmt_float(v, 2))
            st.dataframe(
                dwell_display[["event", "occurrences", "avg_dwell_hours", "median_dwell_hours", "p90_dwell_hours"]],
                use_container_width=True,
                hide_index=True,
            )

        with col_dwell_chart:
            top10_dwell = dwell_df.head(10).copy().sort_values("avg_dwell_hours", ascending=True)
            fig_dwell = px.bar(
                top10_dwell,
                x="event",
                y="avg_dwell_hours",
                text_auto=".2f",
            )
            fig_dwell.update_traces(
                marker_color=ACCENT,
                hovertemplate="Event=%{x}<br>Avg Dwell=%{y:.2f} hrs<extra></extra>",
            )
            st.plotly_chart(
                clayout(
                    fig_dwell,
                    title="Top 10 Dwell Steps by Avg Hours",
                    xtitle="Event",
                    ytitle="Avg Dwell (hrs)",
                    h=500,
                ),
                use_container_width=True,
            )

    st.divider()
    st.markdown("#### Slowest Transitions")
    if transition_df.empty:
        st.info("No rows in im.v_transition_summary.")
    else:
        transition_df = transition_df.copy()
        transition_df["transition_count"] = pd.to_numeric(transition_df["transition_count"], errors="coerce")
        for col in ["avg_delta_hours", "median_delta_hours", "p90_delta_hours"]:
            transition_df[col] = pd.to_numeric(transition_df[col], errors="coerce")
        transition_df = transition_df.dropna(subset=["transition_count", "p90_delta_hours"])

        col_transition_tables, col_transition_chart = st.columns([1.2, 1.0])

        with col_transition_tables:
            tab_slowest, tab_common = st.tabs(["Top by P90 Delay", "Top by Frequency"])

            with tab_slowest:
                slowest15 = transition_df.sort_values("p90_delta_hours", ascending=False).head(15).copy()
                slowest15_display = slowest15.copy()
                slowest15_display["transition_count"] = slowest15_display["transition_count"].map(fmt_int)
                for col in ["avg_delta_hours", "median_delta_hours", "p90_delta_hours"]:
                    slowest15_display[col] = slowest15_display[col].map(lambda v: fmt_float(v, 2))
                st.dataframe(
                    slowest15_display[
                        [
                            "from_event",
                            "to_event",
                            "transition_count",
                            "avg_delta_hours",
                            "median_delta_hours",
                            "p90_delta_hours",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

            with tab_common:
                common15 = transition_df.sort_values("transition_count", ascending=False).head(15).copy()
                common15_display = common15.copy()
                common15_display["transition_count"] = common15_display["transition_count"].map(fmt_int)
                for col in ["avg_delta_hours", "median_delta_hours", "p90_delta_hours"]:
                    common15_display[col] = common15_display[col].map(lambda v: fmt_float(v, 2))
                st.dataframe(
                    common15_display[
                        [
                            "from_event",
                            "to_event",
                            "transition_count",
                            "avg_delta_hours",
                            "median_delta_hours",
                            "p90_delta_hours",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )

        with col_transition_chart:
            top10_slowest = transition_df.sort_values("p90_delta_hours", ascending=False).head(10).copy()
            if top10_slowest.empty:
                st.info("Not enough transition data for slowest-transition chart.")
            else:
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
                fig_slow.update_traces(
                    marker_color="#7FE0D6",
                    hovertemplate="Transition=%{y}<br>P90 Delay=%{x:.2f} hrs<extra></extra>",
                )
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
    st.markdown("#### Insights")
    insights: list[str] = []
    if not dwell_df.empty:
        dwell_num = dwell_df.copy()
        dwell_num["avg_dwell_hours"] = pd.to_numeric(dwell_num["avg_dwell_hours"], errors="coerce")
        dwell_num = dwell_num.dropna(subset=["avg_dwell_hours"])
        if not dwell_num.empty:
            top_step = dwell_num.sort_values("avg_dwell_hours", ascending=False).iloc[0]
            insights.append(f"Longest dwell step: {top_step['event']} (~{float(top_step['avg_dwell_hours']):.2f} hrs avg)")

    if not transition_df.empty:
        trans_num = transition_df.copy()
        trans_num["p90_delta_hours"] = pd.to_numeric(trans_num["p90_delta_hours"], errors="coerce")
        trans_num = trans_num.dropna(subset=["p90_delta_hours"])
        if not trans_num.empty:
            top_transition = trans_num.sort_values("p90_delta_hours", ascending=False).iloc[0]
            insights.append(
                "Highest p90 transition: "
                f"{top_transition['from_event']} -> {top_transition['to_event']} "
                f"(~{float(top_transition['p90_delta_hours']):.2f} hrs p90)"
            )
        if len(trans_num) > 1:
            top_common = trans_num.sort_values("transition_count", ascending=False).iloc[0]
            insights.append(
                "Most frequent transition: "
                f"{top_common['from_event']} -> {top_common['to_event']} "
                f"({int(top_common['transition_count']):,} transitions)"
            )

    if insights:
        st.markdown("\n".join([f"- {item}" for item in insights[:3]]))
    else:
        st.info("No callouts available yet. Load data to compute dwell/transition metrics.")

    with st.expander("Data Notes"):
        st.write(
            "These metrics are computed from event timestamps and should be used to identify process friction, not individual performance."
        )


def render_escalations_handoffs(db_url: str) -> None:
    header("Escalations & Handoffs", "Analyze handoff tax, escalation intensity, and ping-pong behavior.")
    st.caption("Handoffs = resolver_changes")
    st.caption("Escalations = escalation_count")
    st.caption("Ping-pong = level 2 <-> level 3 bouncing inferred from event transitions")

    try:
        handoff_df = run_query(db_url, HANDOFF_SUMMARY_SQL)
        pingpong_df = run_query(db_url, PINGPONG_CASES_SQL)
        worst_df = run_query(db_url, WORST_HANDOFF_SQL)
        pingpong_kpi_df = run_query(db_url, PINGPONG_KPI_SQL)
        overall_kpi_df = run_query(db_url, OVERALL_HANDOFF_BASELINE_SQL)
    except Exception as exc:
        st.error(f"Failed to query escalation/handoff views: {exc}")
        st.stop()

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
            compare_metrics = [
                ("Cases", p["pingpong_cases"], o["overall_cases"], fmt_int),
                ("Avg Cycle (hrs)", p["avg_cycle_hours_pingpong"], o["avg_cycle_hours_overall"], fmt_float),
                ("Avg CSAT", p["avg_csat_pingpong"], o["avg_csat_overall"], fmt_float),
                ("SLA Met %", p["met_sla_rate_pingpong"], o["met_sla_rate_overall"], fmt_pct),
                (
                    "Avg Resolver Changes",
                    p["avg_resolver_changes_pingpong"],
                    o["avg_resolver_changes_overall"],
                    fmt_float,
                ),
                ("Avg Escalations", p["avg_escalations_pingpong"], o["avg_escalations_overall"], fmt_float),
            ]

            row_a = st.columns(3)
            row_b = st.columns(3)
            for idx, (label, ping_val, base_val, formatter) in enumerate(compare_metrics):
                target_col = row_a[idx] if idx < 3 else row_b[idx - 3]
                with target_col:
                    metric_tile(
                        label,
                        formatter(ping_val),
                        sublabel=f"Ping-pong cohort | Overall {formatter(base_val)}",
                    )

    with tab_hotspots:
        st.markdown("#### Handoff Hotspots")
        if handoff_df.empty:
            st.info("No rows in im.v_handoff_summary.")
        else:
            handoff_num = handoff_df.copy()
            num_cols = [
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
            ]
            for col in num_cols:
                handoff_num[col] = pd.to_numeric(handoff_num[col], errors="coerce")

            ranked = handoff_num.sort_values(["high_handoff_rate", "cases"], ascending=[False, False])
            top25 = ranked.head(25).copy()

            col_table, col_chart = st.columns([1.25, 1.0])
            with col_table:
                display_cols = [
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
                    "pingpong_rate",
                ]
                top25_display = top25[display_cols].copy()
                top25_display["cases"] = top25_display["cases"].map(fmt_int)
                for col in ["avg_cycle_hours", "p90_cycle_hours", "avg_csat", "avg_resolver_changes", "avg_escalations"]:
                    top25_display[col] = top25_display[col].map(lambda v: fmt_float(v, 2))
                for col in ["met_sla_rate", "handoff_rate", "high_handoff_rate", "pingpong_rate"]:
                    top25_display[col] = top25_display[col].map(lambda v: fmt_pct(v, 1))

                st.dataframe(top25_display, use_container_width=True, hide_index=True)

            with col_chart:
                top10_hot = ranked.head(10).copy()
                top10_hot["label"] = top10_hot["variant"].astype(str) + " | " + top10_hot["issue_type"].astype(str)
                top10_hot = top10_hot.sort_values("high_handoff_rate", ascending=True)
                fig_hot = px.bar(
                    top10_hot,
                    x="high_handoff_rate",
                    y="label",
                    orientation="h",
                    text_auto=".1%",
                )
                fig_hot.update_traces(
                    marker_color=ACCENT,
                    hovertemplate="Group=%{y}<br>High Handoff=%{x:.1%}<extra></extra>",
                )
                st.plotly_chart(
                    clayout(
                        fig_hot,
                        title="Top 10 High-Handoff Groups",
                        xtitle="High Handoff Rate",
                        ytitle="Variant | Issue Type",
                        h=560,
                    ),
                    use_container_width=True,
                )

    with tab_pingpong:
        st.markdown("#### Ping-Pong Explorer")
        if pingpong_df.empty:
            st.info("No rows in im.v_pingpong_cases.")
        else:
            cases_df = pingpong_df.copy()
            numeric_cols = [
                "cycle_hours",
                "customer_satisfaction",
                "escalation_count",
                "resolver_changes",
                "pingpong_transitions",
            ]
            for col in numeric_cols:
                cases_df[col] = pd.to_numeric(cases_df[col], errors="coerce")

            filter_col1, filter_col2, filter_col3 = st.columns([1.0, 1.2, 0.9])
            with filter_col1:
                priority_opts = sorted([str(v) for v in cases_df["priority"].dropna().unique().tolist()])
                selected_priorities = st.multiselect("Priority", options=priority_opts, default=priority_opts)
            with filter_col2:
                issue_opts = sorted([str(v) for v in cases_df["issue_type"].dropna().unique().tolist()])
                selected_issues = st.multiselect("Issue Type", options=issue_opts, default=issue_opts)
            with filter_col3:
                sla_filter = st.selectbox("SLA", options=["All", "Met", "Missed"], index=0)

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

            csv_pingpong = filtered.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download filtered ping-pong CSV",
                data=csv_pingpong,
                file_name="pingpong_cases_filtered.csv",
                mime="text/csv",
            )

            top50 = filtered.head(50).copy()
            top50_display = top50.copy()
            top50_display["cycle_hours"] = top50_display["cycle_hours"].map(lambda v: fmt_float(v, 2))
            top50_display["customer_satisfaction"] = top50_display["customer_satisfaction"].map(
                lambda v: fmt_float(v, 2)
            )
            top50_display["resolver_changes"] = top50_display["resolver_changes"].map(fmt_int)
            top50_display["escalation_count"] = top50_display["escalation_count"].map(fmt_int)
            top50_display["pingpong_transitions"] = top50_display["pingpong_transitions"].map(fmt_int)

            st.dataframe(
                top50_display[
                    [
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
                ],
                use_container_width=True,
                hide_index=True,
            )

            if filtered.empty:
                st.info("No rows match current filter selection.")
            else:
                plot_df = filtered.dropna(subset=["resolver_changes", "cycle_hours", "pingpong_transitions"]).copy()
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
            worklist = worklist.sort_values(
                ["resolver_changes", "escalation_count", "cycle_hours"],
                ascending=[False, False, False],
            )

            csv_worklist = worklist.to_csv(index=False).encode("utf-8")
            st.download_button(
                "Download worklist CSV",
                data=csv_worklist,
                file_name="worst_handoff_cases.csv",
                mime="text/csv",
            )

            worklist_display = worklist.copy()
            worklist_display["cycle_hours"] = worklist_display["cycle_hours"].map(lambda v: fmt_float(v, 2))
            worklist_display["customer_satisfaction"] = worklist_display["customer_satisfaction"].map(
                lambda v: fmt_float(v, 2)
            )
            worklist_display["escalation_count"] = worklist_display["escalation_count"].map(fmt_int)
            worklist_display["resolver_changes"] = worklist_display["resolver_changes"].map(fmt_int)

            st.dataframe(worklist_display, use_container_width=True, hide_index=True)


def render_quality_cx(db_url: str) -> None:
    header("Quality & CX", "Track closure quality, feedback coverage, and reopen-after-close outcomes.")

    try:
        cx_summary_df = run_query(db_url, CX_SUMMARY_SQL)
        cx_breakdown_df = run_query(db_url, CX_BREAKDOWN_SQL)
        closure_df = run_query(db_url, CLOSURE_COMPLIANCE_SQL)
    except Exception as exc:
        st.error(f"Failed to query quality/CX views: {exc}")
        st.stop()

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
                    marker_color=ACCENT,
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
                    marker_color="#7FE0D6",
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

    with tab_cohorts:
        st.markdown("#### Cohort Explorer")
        if closure_df.empty:
            st.info("No rows in im.v_closure_compliance.")
        else:
            df = closure_df.copy()
            for col in ["cycle_hours", "customer_satisfaction", "reopen_after_close_hours"]:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            for col in [
                "met_sla",
                "has_reject",
                "closed_without_feedback",
                "reopened_within_3h",
                "reopened_within_24h",
            ]:
                df[col] = df[col].fillna(False).astype(bool)

            f1, f2, f3 = st.columns([1.2, 1.2, 1.0])
            with f1:
                only_no_feedback = st.checkbox("Closed without feedback only", value=False)
                only_reopen_3h = st.checkbox("Reopened within 3 hours only", value=False)
                only_reopen_24h = st.checkbox("Reopened within 24 hours only", value=False)
            with f2:
                only_rejected = st.checkbox("Rejected only", value=False)
                only_missed_sla = st.checkbox("Missed SLA only", value=False)
                priority_opts = sorted([str(v) for v in df["priority"].dropna().unique().tolist()])
                selected_priorities = st.multiselect("Priority", options=priority_opts, default=priority_opts)
            with f3:
                issue_opts = sorted([str(v) for v in df["issue_type"].dropna().unique().tolist()])
                selected_issue_types = st.multiselect("Issue Type", options=issue_opts, default=issue_opts)

            filtered = df.copy()
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
            top200 = filtered.head(200).copy()

            st.download_button(
                "Download filtered cohort CSV",
                data=filtered.to_csv(index=False).encode("utf-8"),
                file_name="quality_cx_cohort_filtered.csv",
                mime="text/csv",
            )

            top200_display = top200.copy()
            top200_display["cycle_hours"] = top200_display["cycle_hours"].map(lambda v: fmt_float(v, 2))
            top200_display["customer_satisfaction"] = top200_display["customer_satisfaction"].map(
                lambda v: fmt_float(v, 2)
            )
            top200_display["reopen_after_close_hours"] = top200_display["reopen_after_close_hours"].map(
                lambda v: fmt_float(v, 2)
            )

            st.dataframe(
                top200_display[
                    [
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
                ],
                use_container_width=True,
                hide_index=True,
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
            bdf = bdf.sort_values("cases", ascending=False)

            display = bdf.copy()
            display["cases"] = display["cases"].map(fmt_int)
            for col in ["avg_csat", "avg_cycle_hours"]:
                display[col] = display[col].map(lambda v: fmt_float(v, 2))
            for col in ["met_sla_rate", "closed_without_feedback_rate", "reopened_within_3h_rate", "reject_rate"]:
                display[col] = display[col].map(lambda v: fmt_pct(v, 1))
            st.dataframe(display, use_container_width=True, hide_index=True)

            closed_by_issue = (
                bdf.dropna(subset=["issue_type", "cases", "closed_without_feedback_rate"])
                .assign(weighted_no_feedback=lambda d: d["closed_without_feedback_rate"] * d["cases"])
                .groupby("issue_type", as_index=False)
                .agg(cases=("cases", "sum"), weighted_no_feedback=("weighted_no_feedback", "sum"))
            )
            closed_by_issue["closed_without_feedback_rate"] = (
                closed_by_issue["weighted_no_feedback"] / closed_by_issue["cases"]
            )
            closed_by_issue = closed_by_issue.drop(columns=["weighted_no_feedback"])

            if closure_df.empty:
                reopen24_by_issue = pd.DataFrame(columns=["issue_type", "reopened_within_24h_rate"])
            else:
                reopen24_by_issue = (
                    closure_df[["issue_type", "reopened_within_24h"]]
                    .dropna(subset=["issue_type"])
                    .assign(reopened_within_24h=lambda d: d["reopened_within_24h"].fillna(False).astype(float))
                    .groupby("issue_type", as_index=False)
                    .agg(reopened_within_24h_rate=("reopened_within_24h", "mean"))
                )

            issue_weight = closed_by_issue.merge(reopen24_by_issue, on="issue_type", how="outer")

            chart_cols = st.columns(2)
            with chart_cols[0]:
                top_feedback_gap = issue_weight.sort_values("closed_without_feedback_rate", ascending=False).head(10)
                fig_gap = px.bar(
                    top_feedback_gap,
                    x="issue_type",
                    y="closed_without_feedback_rate",
                    text_auto=".1%",
                )
                fig_gap.update_traces(
                    marker_color=ACCENT,
                    hovertemplate="Issue=%{x}<br>No Feedback=%{y:.1%}<extra></extra>",
                )
                st.plotly_chart(
                    clayout(
                        fig_gap,
                        title="Closed Without Feedback Rate by Issue Type",
                        xtitle="Issue Type",
                        ytitle="Rate",
                        h=460,
                    ),
                    use_container_width=True,
                )

            with chart_cols[1]:
                top_reopen24 = issue_weight.sort_values("reopened_within_24h_rate", ascending=False).head(10)
                fig_reopen = px.bar(
                    top_reopen24,
                    x="issue_type",
                    y="reopened_within_24h_rate",
                    text_auto=".1%",
                )
                fig_reopen.update_traces(
                    marker_color="#7FE0D6",
                    hovertemplate="Issue=%{x}<br>Reopened <=24h=%{y:.1%}<extra></extra>",
                )
                st.plotly_chart(
                    clayout(
                        fig_reopen,
                        title="Reopened Within 24h Rate by Issue Type",
                        xtitle="Issue Type",
                        ytitle="Rate",
                        h=460,
                    ),
                    use_container_width=True,
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
            )

        if closure_df.empty:
            st.info("No rows in im.v_closure_compliance to export.")
        else:
            st.download_button(
                "Download v_closure_compliance CSV",
                data=closure_df.to_csv(index=False).encode("utf-8"),
                file_name="v_closure_compliance.csv",
                mime="text/csv",
            )
            st.caption("v_closure_compliance can be large depending on case volume.")

        st.write(
            "Use the cohort export for focused remediation work and the full exports for deeper offline analysis "
            "in BI tools or notebooks."
        )

    with st.expander("Data Notes"):
        st.write(
            "These metrics are for process improvement and customer outcomes, not individual performance evaluation."
        )


def main() -> None:
    with st.sidebar:
        st.title("IncidentOps")
        st.caption("Process Mining Insights Hub")
        page = st.radio(
            "Navigation",
            [
                "Executive Overview",
                "Process Explorer",
                "Bottlenecks",
                "Escalations & Handoffs",
                "Quality & CX",
                "Problem Candidates (coming soon)",
            ],
        )

    db_url = load_db_url()
    if not db_url:
        st.warning("Set DATABASE_URL_POOLED or DATABASE_URL_DIRECT in .env to load dashboard data.")
        st.stop()

    try:
        get_connection(db_url)
    except Exception as exc:
        st.error(f"Database connection failed: {exc}")
        st.stop()

    if page == "Executive Overview":
        render_executive_overview(db_url)
    elif page == "Process Explorer":
        render_process_explorer(db_url)
    elif page == "Bottlenecks":
        render_bottlenecks(db_url)
    elif page == "Escalations & Handoffs":
        render_escalations_handoffs(db_url)
    elif page == "Quality & CX":
        render_quality_cx(db_url)
    else:
        render_coming_soon(page)


if __name__ == "__main__":
    main()
