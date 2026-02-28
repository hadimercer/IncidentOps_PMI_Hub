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
                "Escalations & Handoffs (coming soon)",
                "Quality & CX (coming soon)",
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
    else:
        render_coming_soon(page)


if __name__ == "__main__":
    main()
