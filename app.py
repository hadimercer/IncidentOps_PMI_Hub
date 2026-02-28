from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import pandas as pd
import plotly.express as px
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


def main() -> None:
    with st.sidebar:
        st.title("IncidentOps")
        st.caption("Process Mining Insights Hub")
        page = st.radio(
            "Navigation",
            [
                "Executive Overview",
                "Process Explorer (coming soon)",
                "Bottlenecks (coming soon)",
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
    else:
        render_coming_soon(page)


if __name__ == "__main__":
    main()
