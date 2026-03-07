from __future__ import annotations

import plotly.graph_objects as go

from ui.theme import TOKENS

COLORWAY = [
    TOKENS["chart_primary"],
    TOKENS["chart_warning"],
    TOKENS["chart_neutral"],
    TOKENS["chart_success"],
    TOKENS["chart_danger"],
]


def style_figure(fig: go.Figure, *, title: str = "", xtitle: str = "", ytitle: str = "", height: int = 420) -> go.Figure:
    fig.update_layout(
        title=dict(text=title, font=dict(size=18, color=TOKENS["text"])),
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=COLORWAY,
        font=dict(color=TOKENS["text_muted"], size=13, family=TOKENS["font_ui"]),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(0,0,0,0)",
            font=dict(color=TOKENS["text_muted"]),
            title=None,
        ),
        margin=dict(l=24, r=20, t=64, b=24),
        hoverlabel=dict(
            bgcolor=TOKENS["surface_alt"],
            bordercolor=TOKENS["border_strong"],
            font=dict(color=TOKENS["text"]),
        ),
    )
    fig.update_xaxes(
        title=xtitle,
        gridcolor=TOKENS["chart_grid"],
        zeroline=False,
        linecolor=TOKENS["chart_axis"],
        tickfont=dict(color=TOKENS["text_muted"]),
        title_font=dict(color=TOKENS["text_soft"]),
    )
    fig.update_yaxes(
        title=ytitle,
        gridcolor=TOKENS["chart_grid"],
        zeroline=False,
        linecolor=TOKENS["chart_axis"],
        tickfont=dict(color=TOKENS["text_muted"]),
        title_font=dict(color=TOKENS["text_soft"]),
    )
    return fig


def annotate(fig: go.Figure, text: str, *, x: float = 0.99, y: float = 1.12) -> go.Figure:
    fig.add_annotation(
        text=text,
        xref="paper",
        yref="paper",
        x=x,
        y=y,
        xanchor="right",
        showarrow=False,
        font=dict(color=TOKENS["text_soft"], size=12),
    )
    return fig
