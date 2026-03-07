from __future__ import annotations

import streamlit as st

TOKENS = {
    "font_ui": "'IBM Plex Sans', 'Segoe UI', sans-serif",
    "font_mono": "'IBM Plex Mono', 'Cascadia Code', monospace",
    "bg": "#09131A",
    "bg_alt": "#0D1B24",
    "surface": "#11222C",
    "surface_alt": "#152B37",
    "surface_soft": "#1B3542",
    "border": "rgba(125, 164, 185, 0.18)",
    "border_strong": "rgba(125, 164, 185, 0.34)",
    "text": "#EAF4F8",
    "text_muted": "#A7BBC7",
    "text_soft": "#7F96A3",
    "accent": "#36C2B4",
    "accent_soft": "rgba(54, 194, 180, 0.16)",
    "warning": "#E9A63A",
    "warning_soft": "rgba(233, 166, 58, 0.14)",
    "danger": "#F06D5E",
    "danger_soft": "rgba(240, 109, 94, 0.16)",
    "success": "#74D8A6",
    "success_soft": "rgba(116, 216, 166, 0.16)",
    "shadow": "0 24px 60px rgba(0, 0, 0, 0.22)",
    "radius_xl": "24px",
    "radius_l": "18px",
    "radius_m": "14px",
    "radius_s": "10px",
    "space_1": "0.35rem",
    "space_2": "0.6rem",
    "space_3": "0.9rem",
    "space_4": "1.2rem",
    "space_5": "1.6rem",
    "space_6": "2.1rem",
    "chart_primary": "#36C2B4",
    "chart_warning": "#E9A63A",
    "chart_danger": "#F06D5E",
    "chart_success": "#74D8A6",
    "chart_neutral": "#6D8594",
    "chart_grid": "rgba(234, 244, 248, 0.10)",
    "chart_axis": "rgba(234, 244, 248, 0.22)",
}


def apply_theme() -> None:
    css = f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
      :root {{
        --io-bg: {TOKENS['bg']};
        --io-bg-alt: {TOKENS['bg_alt']};
        --io-surface: {TOKENS['surface']};
        --io-surface-alt: {TOKENS['surface_alt']};
        --io-surface-soft: {TOKENS['surface_soft']};
        --io-border: {TOKENS['border']};
        --io-border-strong: {TOKENS['border_strong']};
        --io-text: {TOKENS['text']};
        --io-text-muted: {TOKENS['text_muted']};
        --io-text-soft: {TOKENS['text_soft']};
        --io-accent: {TOKENS['accent']};
        --io-accent-soft: {TOKENS['accent_soft']};
        --io-warning: {TOKENS['warning']};
        --io-warning-soft: {TOKENS['warning_soft']};
        --io-danger: {TOKENS['danger']};
        --io-danger-soft: {TOKENS['danger_soft']};
        --io-success: {TOKENS['success']};
        --io-success-soft: {TOKENS['success_soft']};
      }}
      html, body, [class*="css"] {{
        font-family: {TOKENS['font_ui']};
      }}
      .stApp {{
        background:
          radial-gradient(circle at top left, rgba(54,194,180,0.10), transparent 28%),
          radial-gradient(circle at top right, rgba(233,166,58,0.09), transparent 22%),
          linear-gradient(180deg, {TOKENS['bg_alt']} 0%, {TOKENS['bg']} 100%);
        color: {TOKENS['text']};
      }}
      [data-testid="stHeader"] {{
        background: rgba(9, 19, 26, 0.72);
      }}
      [data-testid="stSidebar"] > div:first-child {{
        background:
          linear-gradient(180deg, rgba(17,34,44,0.96) 0%, rgba(13,27,36,0.98) 100%);
        border-right: 1px solid {TOKENS['border']};
      }}
      [data-testid="stSidebarNav"] {{ display: none; }}
      [data-testid="stMetricLabel"], [data-testid="stMetricValue"] {{
        color: {TOKENS['text']};
      }}
      .block-container {{
        padding-top: 1.2rem;
        padding-bottom: 3rem;
        max-width: 1520px;
      }}
      .ops-sidebar-brand {{
        padding: 1rem 1rem 0.9rem 1rem;
        border-radius: {TOKENS['radius_l']};
        margin-bottom: 1rem;
        border: 1px solid {TOKENS['border_strong']};
        background:
          linear-gradient(135deg, rgba(54,194,180,0.18), rgba(17,34,44,0.88));
        box-shadow: {TOKENS['shadow']};
      }}
      .ops-sidebar-brand .eyebrow {{
        color: {TOKENS['accent']};
        font-family: {TOKENS['font_mono']};
        font-size: 0.74rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
      }}
      .ops-sidebar-brand h2 {{
        margin: 0.25rem 0 0 0;
        font-size: 1.25rem;
        color: {TOKENS['text']};
      }}
      .ops-sidebar-brand p {{
        margin: 0.4rem 0 0 0;
        color: {TOKENS['text_muted']};
        font-size: 0.88rem;
        line-height: 1.45;
      }}
      .ops-masthead {{
        position: relative;
        overflow: hidden;
        padding: 1.35rem 1.45rem 1.25rem 1.45rem;
        border-radius: {TOKENS['radius_xl']};
        border: 1px solid {TOKENS['border_strong']};
        background:
          linear-gradient(135deg, rgba(54,194,180,0.16) 0%, rgba(17,34,44,0.94) 32%, rgba(9,19,26,0.97) 100%);
        box-shadow: {TOKENS['shadow']};
        margin-bottom: 1rem;
      }}
      .ops-masthead::after {{
        content: "";
        position: absolute;
        inset: auto -5% -45% auto;
        width: 280px;
        height: 280px;
        border-radius: 999px;
        background: radial-gradient(circle, rgba(233,166,58,0.18), transparent 65%);
        pointer-events: none;
      }}
      .ops-eyebrow {{
        color: {TOKENS['accent']};
        font-family: {TOKENS['font_mono']};
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        margin-bottom: 0.4rem;
      }}
      .ops-masthead h1 {{
        margin: 0;
        color: {TOKENS['text']};
        font-size: 2rem;
        line-height: 1.05;
      }}
      .ops-masthead .objective {{
        margin-top: 0.55rem;
        color: {TOKENS['text_muted']};
        font-size: 1rem;
        max-width: 58rem;
      }}
      .ops-what-matters {{
        margin-top: 0.95rem;
        display: inline-flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        align-items: center;
      }}
      .ops-what-matters .label {{
        color: {TOKENS['text_soft']};
        font-family: {TOKENS['font_mono']};
        font-size: 0.74rem;
        text-transform: uppercase;
        letter-spacing: 0.10em;
      }}
      .ops-chip {{
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        padding: 0.24rem 0.68rem;
        border: 1px solid {TOKENS['border_strong']};
        background: rgba(17,34,44,0.92);
        color: {TOKENS['text_muted']};
        font-size: 0.82rem;
      }}
      .ops-chip.status-good {{ background: {TOKENS['success_soft']}; color: {TOKENS['success']}; border-color: rgba(116,216,166,0.30); }}
      .ops-chip.status-bad {{ background: {TOKENS['danger_soft']}; color: {TOKENS['danger']}; border-color: rgba(240,109,94,0.30); }}
      .ops-chip.status-warn {{ background: {TOKENS['warning_soft']}; color: {TOKENS['warning']}; border-color: rgba(233,166,58,0.30); }}
      .ops-filter-shell {{
        position: sticky;
        top: 4.25rem;
        z-index: 30;
        padding: 0.85rem 1rem 0.3rem 1rem;
        margin-bottom: 1rem;
        border: 1px solid {TOKENS['border']};
        border-radius: {TOKENS['radius_l']};
        background: rgba(9,19,26,0.90);
        backdrop-filter: blur(12px);
      }}
      .ops-filter-title {{
        display: flex;
        justify-content: space-between;
        align-items: center;
        margin-bottom: 0.6rem;
      }}
      .ops-filter-title .label {{
        color: {TOKENS['text']};
        font-family: {TOKENS['font_mono']};
        font-size: 0.78rem;
        text-transform: uppercase;
        letter-spacing: 0.12em;
      }}
      .ops-filter-title .hint {{
        color: {TOKENS['text_soft']};
        font-size: 0.78rem;
      }}
      .ops-kpi-card {{
        border: 1px solid {TOKENS['border']};
        background: linear-gradient(180deg, rgba(21,43,55,0.96), rgba(13,27,36,0.96));
        border-radius: {TOKENS['radius_l']};
        padding: 1rem 1rem 0.95rem 1rem;
        min-height: 176px;
        height: 176px;
        display: flex;
        flex-direction: column;
        box-shadow: {TOKENS['shadow']};
      }}
      .ops-kpi-card .label {{
        color: {TOKENS['text_soft']};
        text-transform: uppercase;
        letter-spacing: 0.10em;
        font-size: 0.73rem;
        font-family: {TOKENS['font_mono']};
      }}
      .ops-kpi-card .value {{
        margin-top: 0.45rem;
        color: {TOKENS['text']};
        font-size: 1.85rem;
        line-height: 1;
        font-weight: 700;
      }}
      .ops-kpi-card .meta {{
        margin-top: 0.6rem;
        color: {TOKENS['text_muted']};
        font-size: 0.84rem;
        line-height: 1.45;
        min-height: 3.65rem;
        overflow: hidden;
        display: -webkit-box;
        -webkit-line-clamp: 3;
        -webkit-box-orient: vertical;
      }}
      .ops-kpi-card .signal {{
        margin-top: auto;
        height: 4px;
        border-radius: 999px;
        background: linear-gradient(90deg, rgba(54,194,180,0.12), rgba(54,194,180,0.72));
      }}
      .ops-panel {{
        border: 1px solid {TOKENS['border']};
        border-radius: {TOKENS['radius_l']};
        background: linear-gradient(180deg, rgba(17,34,44,0.96), rgba(9,19,26,0.98));
        padding: 1rem 1rem 0.8rem 1rem;
        box-shadow: {TOKENS['shadow']};
        margin-bottom: 1rem;
      }}
      .ops-panel.tier-1 {{ border-color: {TOKENS['border_strong']}; }}
      .ops-panel.tier-3 {{ background: linear-gradient(180deg, rgba(15,28,36,0.96), rgba(9,19,26,0.98)); }}
      .ops-panel-header {{
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        align-items: flex-start;
        margin-bottom: 0.75rem;
      }}
      .ops-panel-header h3 {{
        margin: 0.15rem 0 0 0;
        color: {TOKENS['text']};
        font-size: 1.08rem;
      }}
      .ops-panel-header p {{
        margin: 0.35rem 0 0 0;
        color: {TOKENS['text_muted']};
        font-size: 0.88rem;
        line-height: 1.45;
      }}
      .ops-panel-header .eyebrow {{
        color: {TOKENS['text_soft']};
        font-family: {TOKENS['font_mono']};
        font-size: 0.72rem;
        letter-spacing: 0.10em;
        text-transform: uppercase;
      }}
      .ops-narrative {{
        border: 1px solid {TOKENS['border']};
        border-radius: {TOKENS['radius_m']};
        padding: 0.8rem 0.9rem;
        background: rgba(17,34,44,0.76);
        color: {TOKENS['text_muted']};
        font-size: 0.88rem;
        line-height: 1.45;
        margin-bottom: 0.7rem;
      }}
      .ops-insight {{
        border-left: 3px solid {TOKENS['accent']};
        background: rgba(54,194,180,0.08);
        padding: 0.75rem 0.9rem;
        border-radius: 0 {TOKENS['radius_s']} {TOKENS['radius_s']} 0;
        color: {TOKENS['text_muted']};
        margin-bottom: 0.8rem;
      }}
      .ops-empty {{
        border: 1px dashed {TOKENS['border_strong']};
        border-radius: {TOKENS['radius_m']};
        padding: 1rem;
        color: {TOKENS['text_soft']};
        background: rgba(17,34,44,0.55);
      }}
      .stDataFrame, [data-testid="stDataFrame"] {{
        border: 1px solid {TOKENS['border']} !important;
        border-radius: {TOKENS['radius_m']};
      }}
      .stDownloadButton button, .stButton button {{
        border-radius: 999px;
        border: 1px solid {TOKENS['border_strong']};
        background: rgba(17,34,44,0.92);
        color: {TOKENS['text']};
      }}
      .stSelectbox label, .stMultiSelect label, .stDateInput label {{
        color: {TOKENS['text_soft']};
        font-size: 0.76rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
      }}
      .stCaption {{ color: {TOKENS['text_soft']}; }}
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)
