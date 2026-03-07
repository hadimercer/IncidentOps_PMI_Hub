from __future__ import annotations

import io
from datetime import date
from typing import Iterable

import pandas as pd
import streamlit as st


def sidebar_brand() -> None:
    st.markdown(
        """
        <div class="ops-sidebar-brand">
          <div class="eyebrow">Modern Ops Command</div>
          <h2>IncidentOps</h2>
          <p>Executive-first process mining, operational quality, and improvement planning.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def page_masthead(title: str, objective: str, what_matters: str, badge: str | None = None) -> None:
    badge_html = f'<span class="ops-chip">{badge}</span>' if badge else ""
    st.markdown(
        f"""
        <section class="ops-masthead">
          <div class="ops-eyebrow">Operations Review Surface</div>
          <div style="display:flex;justify-content:space-between;gap:1rem;align-items:flex-start;flex-wrap:wrap;">
            <div>
              <h1>{title}</h1>
              <div class="objective">{objective}</div>
              <div class="ops-what-matters">
                <span class="label">What matters now</span>
                <span class="ops-chip">{what_matters}</span>
              </div>
            </div>
            <div>{badge_html}</div>
          </div>
        </section>
        """,
        unsafe_allow_html=True,
    )


def section_header(title: str, description: str, eyebrow: str = "Analysis") -> None:
    st.markdown(
        f"""
        <div class="ops-panel-header">
          <div>
            <div class="eyebrow">{eyebrow}</div>
            <h3>{title}</h3>
            <p>{description}</p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def metric_strip(metrics: list[dict[str, str]]) -> None:
    if not metrics:
        return
    st.markdown('<div class="ops-kpi-strip">', unsafe_allow_html=True)
    cols = st.columns(len(metrics))
    for col, metric in zip(cols, metrics):
        with col:
            accent_style = metric.get("accent_style", "")
            st.markdown(
                f"""
                <div class="ops-kpi-card" style="{accent_style}">
                  <div class="label">{metric['label']}</div>
                  <div class="value">{metric['value']}</div>
                  <div class="meta">{metric.get('meta', '')}</div>
                  <div class="signal"></div>
                </div>
                """,
                unsafe_allow_html=True,
            )
    st.markdown('</div>', unsafe_allow_html=True)


def narrative_callouts(items: Iterable[str]) -> None:
    for item in items:
        st.markdown(f'<div class="ops-narrative">{item}</div>', unsafe_allow_html=True)


def insight_callout(text: str) -> None:
    st.markdown(f'<div class="ops-insight">{text}</div>', unsafe_allow_html=True)


def empty_state(message: str) -> None:
    st.markdown(f'<div class="ops-empty">{message}</div>', unsafe_allow_html=True)


def _build_column_config(df: pd.DataFrame, *, percent_cols: set[str] | None = None, number_decimals: int = 2):
    percent_cols = percent_cols or set()
    config = {}
    for column in df.columns:
        if column in percent_cols:
            config[column] = st.column_config.ProgressColumn(
                column.replace("_", " ").title(),
                min_value=0.0,
                max_value=1.0,
                format="%.1f%%",
            )
        elif pd.api.types.is_numeric_dtype(df[column]):
            config[column] = st.column_config.NumberColumn(
                column.replace("_", " ").title(),
                format=f"%.{number_decimals}f",
            )
        else:
            config[column] = st.column_config.TextColumn(column.replace("_", " ").title())
    return config


def _coerce_chips(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    for column in output.columns:
        if pd.api.types.is_bool_dtype(output[column]):
            if "reopen" in column:
                output[column] = output[column].map({True: "Reopened", False: "Stable"})
            elif "feedback" in column:
                output[column] = output[column].map({True: "Feedback", False: "Feedback Missing"})
            elif "reject" in column:
                output[column] = output[column].map({True: "Rejected", False: "Accepted"})
            elif "sla" in column:
                output[column] = output[column].map({True: "Met", False: "Missed"})
            else:
                output[column] = output[column].map({True: "Yes", False: "No"})
    return output


def summary_table(df: pd.DataFrame, *, percent_cols: set[str] | None = None, height: int = 420, hide_index: bool = True) -> None:
    if df.empty:
        empty_state("No rows match the current filters.")
        return
    display_df = _coerce_chips(df)
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=hide_index,
        height=height,
        column_config=_build_column_config(df, percent_cols=percent_cols),
    )


def worklist_table(df: pd.DataFrame, *, percent_cols: set[str] | None = None, height: int = 420, hide_index: bool = True) -> None:
    if df.empty:
        empty_state("No drilldown rows match the current filters.")
        return
    display_df = _coerce_chips(df)
    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=hide_index,
        height=height,
        column_config=_build_column_config(df, percent_cols=percent_cols),
    )


def export_frame(label: str, df: pd.DataFrame, *, key: str) -> None:
    buffer = io.BytesIO()
    df.to_csv(buffer, index=False)
    st.download_button(
        label,
        data=buffer.getvalue(),
        file_name=f"{key}.csv",
        mime="text/csv",
        key=f"download_{key}",
    )


def ensure_filter_state(min_date: date, max_date: date) -> None:
    defaults = {
        "global_date_range": (min_date, max_date),
        "global_priority": [],
        "global_issue_type": [],
        "global_variant": [],
        "global_channel": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def global_filter_bar(*, min_date: date, max_date: date, priorities: list[str], issue_types: list[str], variants: list[str], channels: list[str]) -> dict[str, object]:
    ensure_filter_state(min_date, max_date)
    st.markdown(
        """
        <div class="ops-filter-shell">
          <div class="ops-filter-title">
            <span class="label">Global Scope</span>
            <span class="hint">These filters persist across workstreams.</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns([1.2, 1, 1, 1, 1, 0.55])
    with cols[0]:
        st.date_input(
            "Date range",
            min_value=min_date,
            max_value=max_date,
            key="global_date_range",
        )
    with cols[1]:
        st.multiselect("Priority", priorities, key="global_priority")
    with cols[2]:
        st.multiselect("Issue Type", issue_types, key="global_issue_type")
    with cols[3]:
        st.multiselect("Variant", variants, key="global_variant")
    with cols[4]:
        st.multiselect("Channel", channels, key="global_channel")
    with cols[5]:
        st.write("")
        if st.button("Reset", use_container_width=True):
            st.session_state["global_date_range"] = (min_date, max_date)
            st.session_state["global_priority"] = []
            st.session_state["global_issue_type"] = []
            st.session_state["global_variant"] = []
            st.session_state["global_channel"] = []
            st.rerun()

    resolved_range = st.session_state["global_date_range"]
    if isinstance(resolved_range, tuple):
        start_date, end_date = resolved_range
    else:
        start_date = resolved_range[0]
        end_date = resolved_range[-1] if len(resolved_range) > 1 else resolved_range[0]

    return {
        "date_range": (start_date, end_date),
        "priority": st.session_state["global_priority"],
        "issue_type": st.session_state["global_issue_type"],
        "variant": st.session_state["global_variant"],
        "channel": st.session_state["global_channel"],
    }
