"""
Sentiment and market association dashboard (Streamlit PoC).

From repo root:
  pip install -r requirements.txt -r requirements-ui.txt
  # optional: pip install -r requirements-ml.txt  requirements-llm.txt
  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import html
import math
import re
import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from csv_infer import infer_market_columns, infer_review_columns  # noqa: E402
from executive_copy import (  # noqa: E402
    CHART_LEGEND_LEXICON,
    CHART_LEGEND_NEURAL,
    LABEL_LEXICON_SCORER,
    LABEL_NEURAL_SCORER,
    TECH_STRIP_TITLE,
    association_explanation_html,
    best_day_review_samples,
    build_chart_callouts,
    build_insight_facts,
    build_market_association_beats,
    build_market_stability_beats,
    build_market_trajectory_beats,
    build_surface_low_bullets,
    build_technical_strip_association,
    build_technical_strip_holdout,
    build_technical_strip_overview,
    build_technical_strip_reviews,
    build_technical_strip_stability,
    build_technical_strip_themes,
    build_technical_strip_trajectory,
    chart_summary_stats,
    fallback_executive_narrative,
    holdout_intro_markdown,
    landing_subscription_value_html,
    LAG_EXPLAINER,
    insight_association_section_markdown,
    insight_holdout_section_markdown,
    insight_stability_section_markdown,
    insight_trajectory_section_markdown,
    narrative_cache_key,
    narrative_payload,
    overview_ir_markdown,
    stability_summary_stats,
    themes_interpretation_markdown,
    try_executive_narrative_markdown,
    worst_day_review_samples,
)
from insights import actionable_insights, plain_language_insights  # noqa: E402
from llm_chat import build_chat_context, chat_complete, llm_client_configured  # noqa: E402
from pipeline import dedupe_disagreement_quotes, run_pipeline  # noqa: E402
from report_pdf import build_pdf_bytes  # noqa: E402

SAMPLE = ROOT / "sample_data"
COMMITTED_REV = SAMPLE / "reviews_demo.csv"
COMMITTED_MKT = SAMPLE / "market_demo.csv"

# Presentation-aligned palette (dark navy + teal–cyan)
C_TEAL = "#14b8a6"
C_TEAL_BRIGHT = "#2dd4bf"
C_TEAL_DARK = "#0f766e"
C_NAVY = "#0f172a"
C_NAVY_DEEP = "#020617"
C_INDIGO = "#6366f1"
C_SLATE = "#64748b"
C_AMBER = "#d97706"
C_CREAM = "#f5f5f0"
C_CARD = "#fafaf8"


def _md_inline(text: str) -> str:
    """Convert **bold**, *italic*, and `code` to HTML for use inside one wrapper element."""
    text = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"\*(.+?)\*", r"<em>\1</em>", text)
    text = re.sub(r"`([^`]+)`", r'<code class="rs-inline-code">\1</code>', text)
    return text


def _key_findings_stack_html(lines: list[str]) -> str:
    """Stacked cards — each finding is one flex row so inner <strong> does not fragment layout."""
    parts: list[str] = []
    for i, line in enumerate(lines, start=1):
        body = _md_inline(line)
        parts.append(
            f'<div class="rs-finding-card">'
            f'<span class="rs-finding-num">{i}</span>'
            f'<div class="rs-finding-body">{body}</div>'
            f"</div>"
        )
    return f'<div class="rs-findings-stack">{"".join(parts)}</div>'


def _fmt_num(x: object | None, nd: int = 4) -> str:
    if x is None:
        return "n/a"
    try:
        v = float(x)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(v):
        return "n/a"
    return f"{v:.{nd}f}"


def _format_metric_date_span(d0: str | None, d1: str | None, *, empty: str = "—") -> str:
    """Readable date span for metrics (shorter than ISO + arrows)."""
    if not d0 or not d1:
        return empty
    s0, s1 = str(d0).strip(), str(d1).strip()
    if not s0 or not s1 or s0 == "—" or s1 == "—":
        return empty
    try:
        a = pd.Timestamp(s0)
        b = pd.Timestamp(s1)

        def mdy(ts: pd.Timestamp) -> str:
            return f"{ts.strftime('%b')} {int(ts.day)}, {ts.year}"

        if a.normalize() == b.normalize():
            return mdy(a)
        return f"{mdy(a)} to {mdy(b)}"
    except (ValueError, TypeError, OSError):
        return f"{s0.replace('-', '/')} to {s1.replace('-', '/')}"


def _inject_brand_style(*, landing: bool = False) -> None:
    dashboard_css = ""
    if not landing:
        dashboard_css = """
        /* Push first content below Streamlit header / browser chrome (was padding-top: 0) */
        .block-container {
            padding-top: 2.75rem !important;
        }
        """
    landing_css = ""
    if landing:
        landing_css = f"""
        /* Seamless gradient across the whole landing page */
        [data-testid="stAppViewContainer"] {{
            background: linear-gradient(160deg, {C_NAVY_DEEP} 0%, #091825 45%, #0d2235 100%) !important;
        }}
        .block-container {{
            padding-top: 0.35rem !important;
            padding-bottom: 1rem !important;
        }}
        section[data-testid="stMain"] .stAlert {{
            padding: 0.55rem 0.75rem !important;
            margin-bottom: 0.35rem !important;
        }}
        /* Hero is transparent — the page gradient shows through */
        .rs-landing-wrap {{
            background: transparent !important;
            border-radius: 0 !important;
            padding-bottom: 1.2rem !important;
            margin-bottom: 0 !important;
        }}
        div[data-testid="stVerticalBlockBorderWrapper"] > div {{
            background: rgba(255,255,255,0.04) !important;
            border-color: rgba(255,255,255,0.1) !important;
            border-radius: 16px !important;
            padding: 0.65rem 1rem 0.85rem !important;
        }}
        .rs-form-label {{
            margin: 0.75rem 0 0.2rem 0 !important;
        }}
        .rs-form-label:first-of-type {{ margin-top: 0.1rem !important; }}
        """
    st.markdown(
        f"""
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:ital,opsz,wght@0,9..40,300;0,9..40,400;0,9..40,600;0,9..40,700;1,9..40,400&display=swap');

        html, body, [data-testid="stAppViewContainer"] {{
            font-family: "DM Sans", ui-sans-serif, system-ui, sans-serif;
            background: {C_NAVY_DEEP} !important;
        }}

        /* Hide sidebar and Streamlit toolbar */
        section[data-testid="stSidebar"] {{ display: none !important; }}
        div[data-testid="stToolbar"] {{ display: none !important; }}
        #MainMenu {{ display: none !important; }}
        footer {{ display: none !important; }}

        /* Main container */
        .block-container {{
            padding-top: 0 !important;
            padding-bottom: 3rem !important;
            max-width: 1100px !important;
        }}
        {landing_css}
        {dashboard_css}

        /* ─── LANDING (compact hero — fits one viewport with form) ─── */
        .rs-landing-wrap {{
            position: relative;
            min-height: 0;
            border-radius: 0 0 18px 18px;
            padding: 1rem 1.2rem 0.85rem 1.2rem;
            margin: 0 -1rem 0.35rem -1rem;
            background: linear-gradient(150deg, {C_NAVY_DEEP} 0%, #091825 40%, #0d2235 72%, #0f2d45 100%);
            overflow: hidden;
        }}
        .rs-landing-wrap::before {{
            content: "";
            position: absolute;
            width: 380px; height: 380px;
            top: -140px; right: -80px;
            background: radial-gradient(circle, rgba(45,212,191,0.2) 0%, transparent 68%);
            pointer-events: none;
        }}
        .rs-landing-wrap::after {{
            content: "";
            position: absolute;
            width: 320px; height: 320px;
            bottom: -120px; left: -60px;
            background: radial-gradient(circle, rgba(20,184,166,0.16) 0%, transparent 66%);
            pointer-events: none;
        }}
        .rs-landing-inner {{
            position: relative;
            z-index: 1;
            max-width: 100%;
        }}
        .rs-eyebrow {{
            color: {C_TEAL_BRIGHT};
            font-size: 0.65rem;
            font-weight: 600;
            letter-spacing: 0.22em;
            text-transform: uppercase;
            margin-bottom: 0.35rem;
        }}
        .rs-landing-title {{
            color: #ffffff;
            font-size: clamp(1.55rem, 4.2vw, 2.2rem);
            font-weight: 700;
            letter-spacing: -0.035em;
            line-height: 1.08;
            margin: 0 0 0.4rem 0;
        }}
        .rs-landing-title span {{ color: {C_TEAL_BRIGHT}; }}
        .rs-landing-sub {{
            color: rgba(226,232,240,0.75);
            font-size: 0.84rem;
            line-height: 1.42;
            max-width: 100%;
            margin: 0;
        }}
        .rs-form-label {{
            color: {C_TEAL};
            font-size: 0.68rem;
            font-weight: 700;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            margin: 0.75rem 0 0.2rem 0;
            display: block;
        }}
        .rs-form-label:first-child {{ margin-top: 0; }}

        /* ─── DASHBOARD ─── */
        /* Top header title (left column of the header row) */
        .rs-hdr-title {{
            margin: 0;
            padding: 0.15rem 0 0.35rem 0;
            line-height: 1.55;
        }}
        .rs-topbar-badge {{
            color: {C_TEAL};
            font-size: 0.72rem;
            font-weight: 700;
            letter-spacing: 0.18em;
            text-transform: uppercase;
            margin: 0;
        }}
        .rs-topbar-company {{
            color: #ffffff;
            font-size: 1rem;
            font-weight: 600;
            letter-spacing: -0.01em;
            margin: 0;
        }}
        .rs-topbar-sep {{
            color: rgba(255,255,255,0.25);
            font-size: 0.9rem;
            margin: 0;
        }}
        /* Signal callout box */
        .rs-signal-box {{
            background: rgba(20,184,166,0.07);
            border: 1px solid rgba(20,184,166,0.2);
            border-radius: 12px;
            padding: 0.9rem 1.1rem;
            margin: 0.5rem 0 1.5rem 0;
            color: #cbd5e1;
            font-size: 0.92rem;
            line-height: 1.55;
        }}
        .rs-signal-box strong {{ color: {C_TEAL_BRIGHT}; }}
        /* Key findings — one card per takeaway (body is a single flex child; avoids word-by-word flex gaps) */
        .rs-findings-stack {{
            display: flex;
            flex-direction: column;
            gap: 0.65rem;
        }}
        .rs-finding-card {{
            display: flex;
            flex-direction: row;
            align-items: flex-start;
            gap: 0.85rem;
            padding: 0.85rem 1rem;
            background: rgba(15,23,42,0.72);
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
            border-left: 3px solid {C_TEAL};
        }}
        .rs-finding-num {{
            flex-shrink: 0;
            min-width: 1.65rem;
            height: 1.65rem;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 8px;
            background: rgba(20,184,166,0.14);
            color: {C_TEAL_BRIGHT};
            font-size: 0.78rem;
            font-weight: 700;
            line-height: 1;
        }}
        .rs-finding-body {{
            flex: 1;
            min-width: 0;
            color: #cbd5e1;
            font-size: 0.9rem;
            line-height: 1.58;
        }}
        .rs-finding-body strong {{ color: #f1f5f9; font-weight: 600; }}
        .rs-finding-body em {{ color: #94a3b8; }}
        .rs-inline-code {{
            font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, monospace;
            font-size: 0.84em;
            padding: 0.12rem 0.35rem;
            border-radius: 4px;
            background: rgba(0,0,0,0.35);
            color: {C_TEAL_BRIGHT};
        }}

        /* Section titles */
        .rs-section {{
            margin: 2.25rem 0 1rem 0;
            padding-bottom: 0.6rem;
            border-bottom: 1px solid rgba(20,184,166,0.18);
        }}
        .rs-section:first-child {{ margin-top: 1rem; }}
        .rs-section h3 {{
            margin: 0;
            color: #f1f5f9;
            font-size: 1.15rem;
            font-weight: 700;
            letter-spacing: -0.025em;
        }}
        .rs-section p {{
            margin: 0.3rem 0 0 0;
            color: #64748b;
            font-size: 0.875rem;
        }}

        /* Metric cards */
        div[data-testid="stMetric"] {{
            background: {C_NAVY} !important;
            border: 1px solid rgba(255,255,255,0.08) !important;
            border-radius: 14px !important;
            padding: 1rem 1.1rem !important;
            box-shadow: 0 1px 3px rgba(0,0,0,0.3) !important;
        }}
        div[data-testid="stMetric"] label {{
            color: #64748b !important;
            font-size: 0.8rem !important;
            font-weight: 500 !important;
            white-space: normal !important;
            line-height: 1.3 !important;
        }}
        div[data-testid="stMetric"] [data-testid="stMetricValue"] {{
            color: #f1f5f9 !important;
            font-size: 1rem !important;
            font-weight: 600 !important;
            line-height: 1.4 !important;
            word-break: break-word !important;
            overflow-wrap: anywhere !important;
        }}

        /* Tab bar */
        [data-baseweb="tab-list"] {{
            gap: 0.15rem !important;
            background: transparent !important;
            border-bottom: 1px solid rgba(255,255,255,0.08) !important;
            padding-bottom: 0 !important;
            margin-bottom: 0.5rem !important;
        }}
        button[data-baseweb="tab"] {{
            font-size: 0.88rem !important;
            font-weight: 600 !important;
            color: #64748b !important;
            background: transparent !important;
            border-radius: 6px 6px 0 0 !important;
            padding: 0.6rem 1.1rem !important;
            border-bottom: 2px solid transparent !important;
            transition: color 0.15s ease !important;
        }}
        button[data-baseweb="tab"]:hover {{
            color: #cbd5e1 !important;
        }}
        button[data-baseweb="tab"][aria-selected="true"] {{
            color: {C_TEAL_BRIGHT} !important;
            border-bottom: 2px solid {C_TEAL} !important;
            background: rgba(20,184,166,0.06) !important;
        }}

        /* Horizontal rules */
        hr {{
            border: none !important;
            border-top: 1px solid rgba(255,255,255,0.07) !important;
            margin: 1.5rem 0 !important;
        }}

        /* Download / action buttons keep teal primary */
        div[data-testid="stDownloadButton"] > button[kind="primary"] {{
            background: {C_TEAL} !important;
            color: {C_NAVY_DEEP} !important;
            font-weight: 700 !important;
            border: none !important;
        }}

        /* Dataframe dark */
        div[data-testid="stDataFrame"] {{
            border: 1px solid rgba(255,255,255,0.07) !important;
            border-radius: 10px !important;
            overflow: hidden !important;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def _section_title(title: str, subtitle: str = "") -> None:
    sub = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f'<div class="rs-section"><h3>{title}</h3>{sub}</div>',
        unsafe_allow_html=True,
    )


def load_builtin_demo() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not COMMITTED_REV.is_file() or not COMMITTED_MKT.is_file():
        raise FileNotFoundError(
            f"Missing built-in CSVs in {SAMPLE} (reviews_demo.csv + market_demo.csv)."
        )
    return pd.read_csv(COMMITTED_REV), pd.read_csv(COMMITTED_MKT)


def _pick_col_idx(cols: list[str], name: str | None, fallback: int = 0) -> int:
    if not cols:
        return 0
    if name and name in cols:
        return cols.index(name)
    return min(fallback, len(cols) - 1)


def _pick_ret_default_col(cols: list[str]) -> int:
    for name in ("ret", "RET", "return", "Return", "daily_return"):
        if name in cols:
            return cols.index(name)
    return min(2, len(cols) - 1) if cols else 0


def fig_dual_sentiment(merged: pd.DataFrame) -> go.Figure:
    d = merged.sort_values("date")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=d["date"],
            y=d["avg_textblob"],
            mode="lines+markers",
            name=CHART_LEGEND_LEXICON,
            line=dict(width=2.5, color=C_TEAL),
            marker=dict(size=5, color=C_TEAL),
        )
    )
    if d["avg_transformer"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=d["date"],
                y=d["avg_transformer"],
                mode="lines+markers",
                name=CHART_LEGEND_NEURAL,
                line=dict(width=2.5, color=C_INDIGO, dash="dash"),
                marker=dict(size=5, color=C_INDIGO),
            )
        )
    fig.update_layout(
        margin=dict(l=20, r=20, t=48, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="right", x=1,
                    font=dict(color="#cbd5e1")),
        yaxis_title="Daily avg sentiment",
        xaxis_title="Date",
        height=440,
        paper_bgcolor=C_NAVY,
        plot_bgcolor=C_NAVY,
        font=dict(family="DM Sans, sans-serif", color="#94a3b8"),
        title=dict(text="Review tone over time (daily average)", font=dict(size=14, color="#f1f5f9")),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#1e293b", color="#64748b")
    fig.update_yaxes(showgrid=True, gridcolor="#1e293b", zeroline=True, zerolinecolor="#334155", color="#64748b")
    return fig


def fig_association_scatter(reg: pd.DataFrame, worst_day: str | None = None) -> go.Figure | None:
    """
    Scatter: lag-1 daily avg sentiment (x) vs next-day return (y).
    Each dot = one trading day. Colour fades from early (light) to late (teal).
    Worst day is highlighted in amber.
    """
    d = reg.dropna(subset=["avg_textblob_lag1", "ret"]).copy()
    if len(d) < 3:
        return None
    d = d.sort_values("date").reset_index(drop=True)

    # Colour gradient by time (index position)
    n = len(d)
    colors = [f"rgba(20,184,166,{0.35 + 0.65 * (i / max(n - 1, 1)):.2f})" for i in range(n)]

    # Date labels for hover
    date_labels = []
    for row in d.itertuples():
        ds = str(row.date.date()) if hasattr(row.date, "date") else str(row.date)
        date_labels.append(ds)

    # Separate worst day
    worst_mask = [dl == worst_day for dl in date_labels] if worst_day else [False] * n
    normal_idx = [i for i, m in enumerate(worst_mask) if not m]
    worst_idx = [i for i, m in enumerate(worst_mask) if m]

    fig = go.Figure()

    # Normal dots
    if normal_idx:
        fig.add_trace(go.Scatter(
            x=d["avg_textblob_lag1"].iloc[normal_idx],
            y=d["ret"].iloc[normal_idx],
            mode="markers",
            name="Trading day",
            marker=dict(
                size=8,
                color=[colors[i] for i in normal_idx],
                line=dict(width=0.5, color="rgba(255,255,255,0.15)"),
            ),
            text=[date_labels[i] for i in normal_idx],
            hovertemplate="<b>%{text}</b><br>Sentiment: %{x:.3f}<br>Next-day return: %{y:.4f}<extra></extra>",
        ))

    # Worst day highlighted
    if worst_idx:
        fig.add_trace(go.Scatter(
            x=d["avg_textblob_lag1"].iloc[worst_idx],
            y=d["ret"].iloc[worst_idx],
            mode="markers",
            name=f"Worst day ({worst_day})",
            marker=dict(size=13, color=C_AMBER, symbol="diamond",
                        line=dict(width=1.5, color="#fbbf24")),
            text=[date_labels[i] for i in worst_idx],
            hovertemplate="<b>%{text} — worst day</b><br>Sentiment: %{x:.3f}<br>Next-day return: %{y:.4f}<extra></extra>",
        ))

    # Trend line via simple linear regression
    import numpy as np
    x_vals = d["avg_textblob_lag1"].values
    y_vals = d["ret"].values
    mask_finite = ~(~pd.Series(x_vals).notna() | ~pd.Series(y_vals).notna())
    if mask_finite.sum() >= 3:
        coef = np.polyfit(x_vals, y_vals, 1)
        x_line = np.linspace(float(x_vals.min()), float(x_vals.max()), 60)
        y_line = coef[0] * x_line + coef[1]
        fig.add_trace(go.Scatter(
            x=x_line,
            y=y_line,
            mode="lines",
            name="Trend",
            line=dict(color="rgba(100,116,139,0.55)", width=1.5, dash="dot"),
            hoverinfo="skip",
        ))

    fig.add_hline(y=0, line=dict(color="#334155", width=1))

    fig.update_layout(
        margin=dict(l=20, r=20, t=48, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.06, xanchor="right", x=1,
                    font=dict(color="#cbd5e1")),
        xaxis_title="Yesterday's avg review sentiment →",
        yaxis_title="Today's stock return →",
        height=420,
        paper_bgcolor=C_NAVY,
        plot_bgcolor=C_NAVY,
        font=dict(family="DM Sans, sans-serif", color="#94a3b8"),
        title=dict(text="Do warmer reviews line up with better stock days?", font=dict(size=14, color="#f1f5f9")),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#1e293b", zeroline=True, zerolinecolor="#334155", color="#64748b")
    fig.update_yaxes(showgrid=True, gridcolor="#1e293b", zeroline=True, zerolinecolor="#334155", color="#64748b")
    return fig


def fig_correlation_stability(stab: pd.DataFrame | None, *, show_neural_traces: bool = False) -> go.Figure | None:
    if stab is None or stab.empty:
        return None
    d = stab.sort_values("date")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=d["date"],
            y=d["expanding_corr_textblob"],
            mode="lines",
            name=f"Expanding r · {CHART_LEGEND_LEXICON}",
            line=dict(width=2.5, color=C_TEAL),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=d["date"],
            y=d["rolling_corr_textblob"],
            mode="lines",
            name=f"Rolling 20d · {CHART_LEGEND_LEXICON}",
            line=dict(width=2, color=C_AMBER, dash="dash"),
        )
    )
    if show_neural_traces and "expanding_corr_roberta" in d.columns and d["expanding_corr_roberta"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=d["date"],
                y=d["expanding_corr_roberta"],
                mode="lines",
                name=f"Expanding r · {CHART_LEGEND_NEURAL}",
                line=dict(width=2, color=C_INDIGO, dash="dot"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=d["date"],
                y=d["rolling_corr_roberta"],
                mode="lines",
                name=f"Rolling 20d · {CHART_LEGEND_NEURAL}",
                line=dict(width=2, color="#a855f7", dash="longdash"),
            )
        )
    fig.update_layout(
        margin=dict(l=20, r=20, t=48, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.08, xanchor="right", x=1,
                    font=dict(color="#cbd5e1")),
        yaxis_title="Link strength (Pearson r)",
        xaxis_title="Date",
        height=420,
        paper_bgcolor=C_NAVY,
        plot_bgcolor=C_NAVY,
        font=dict(family="DM Sans, sans-serif", color="#94a3b8"),
        title=dict(text="How stable is the sentiment–return link over time?", font=dict(size=14, color="#f1f5f9")),
    )
    fig.update_xaxes(showgrid=True, gridcolor="#1e293b", color="#64748b")
    fig.update_yaxes(showgrid=True, gridcolor="#1e293b", zeroline=True, zerolinecolor="#334155", color="#64748b")
    return fig


def _landing_form_container():
    """Bordered container when Streamlit supports it (wraps widgets; raw HTML divs do not)."""
    try:
        return st.container(border=True)
    except TypeError:
        return st.container()


def render_landing_page() -> None:
    # ── Hero (compact — stays above the fold with the form) ───────────────────
    st.markdown(
        """
        <div class="rs-landing-wrap">
          <div class="rs-landing-inner">
            <div class="rs-eyebrow">Alternative Data &nbsp;·&nbsp; NLP</div>
            <h1 class="rs-landing-title">Review<span>Signal</span></h1>
            <p class="rs-landing-sub">
              Drop in fresh reviews and returns on whatever cadence you choose — each run tells you what moved,
              what to read first, and what to tell your teams. Same column mapping next time so week-over-week stays fair.
              Not trading advice.
            </p>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(landing_subscription_value_html(), unsafe_allow_html=True)

    # ── Resolve data state before building the form ────────────────────────────
    reviews_df = st.session_state.reviews_df
    market_df = st.session_state.market_df
    rg = infer_review_columns(reviews_df) if reviews_df is not None else None
    mg = infer_market_columns(market_df) if market_df is not None else None
    rev_cols = list(reviews_df.columns) if reviews_df is not None else ["date", "review_text", "ticker"]
    mkt_cols = list(market_df.columns) if market_df is not None else ["date", "ticker", "ret"]

    with _landing_form_container():
        st.markdown('<span class="rs-form-label">Company / issuer</span>', unsafe_allow_html=True)
        st.text_input(
            "Company",
            key="rs_company",
            label_visibility="collapsed",
            placeholder="e.g. LinguaLoop Ltd.",
            help="Used on the executive brief, PDF export, and Q&A context.",
        )

        st.markdown('<span class="rs-form-label">Data</span>', unsafe_allow_html=True)
        if reviews_df is not None and market_df is not None:
            mode = st.session_state.get("data_mode", "builtin")
            if mode == "builtin":
                st.success(
                    f"**Demo ready** — {len(reviews_df):,} reviews · {len(market_df):,} market rows. "
                    "**Run report** or replace with your CSVs."
                )
            else:
                st.info(f"**Your CSVs** — {len(reviews_df):,} reviews · {len(market_df):,} market rows.")
        else:
            st.warning("Load two CSVs below or restore the demo.")

        rcol, u1, u2 = st.columns([0.85, 1, 1])
        with rcol:
            if st.button("Restore demo", key="rs_restore_demo", width="stretch"):
                try:
                    st.session_state.reviews_df, st.session_state.market_df = load_builtin_demo()
                    st.session_state.data_mode = "builtin"
                    st.session_state.result = None
                    st.session_state.chat_messages = []
                    st.rerun()
                except FileNotFoundError as e:
                    st.error(str(e))
        with u1:
            up_rev = st.file_uploader("Reviews CSV", type=["csv"], key="rs_up_rev", help="Date, review text, ticker.")
        with u2:
            up_mkt = st.file_uploader("Market CSV", type=["csv"], key="rs_up_mkt", help="Date, ticker, daily return.")

        if up_rev is not None:
            st.session_state.reviews_df = pd.read_csv(up_rev)
            st.session_state.result = None
            st.session_state.data_mode = "custom"
            reviews_df = st.session_state.reviews_df
            rg = infer_review_columns(reviews_df)
            rev_cols = list(reviews_df.columns)
        if up_mkt is not None:
            st.session_state.market_df = pd.read_csv(up_mkt)
            st.session_state.result = None
            st.session_state.data_mode = "custom"
            market_df = st.session_state.market_df
            mg = infer_market_columns(market_df)
            mkt_cols = list(market_df.columns)

        det_parts: list[str] = []
        if rg:
            det_parts.append(
                f"Reviews: `{rg.date_col}`, `{rg.text_col}`, `{rg.ticker_col}` ({rg.confidence:.0%})"
            )
        if mg:
            det_parts.append(
                f"Market: `{mg.date_col}`, `{mg.ticker_col}`, `{mg.ret_col}` ({mg.confidence:.0%})"
            )
        if det_parts:
            st.caption("Detected columns — " + " · ".join(det_parts))
        if rg and rg.confidence < 0.42:
            st.warning("Review columns uncertain — check **Advanced overrides**.")
        if mg and mg.confidence < 0.42:
            st.warning("Market columns uncertain — check **Advanced overrides**.")

        with st.expander("Advanced column overrides", expanded=False):
            ac1, ac2 = st.columns(2)
            with ac1:
                st.selectbox(
                    "Review date", rev_cols, index=_pick_col_idx(rev_cols, rg.date_col if rg else None), key="ov_r_date"
                )
                st.selectbox(
                    "Review text", rev_cols, index=_pick_col_idx(rev_cols, rg.text_col if rg else None, 1), key="ov_r_text"
                )
                st.selectbox(
                    "Review ticker",
                    rev_cols,
                    index=_pick_col_idx(rev_cols, rg.ticker_col if rg else None, 2),
                    key="ov_r_ticker",
                )
            with ac2:
                st.selectbox(
                    "Market date", mkt_cols, index=_pick_col_idx(mkt_cols, mg.date_col if mg else None), key="ov_m_date"
                )
                st.selectbox(
                    "Market ticker",
                    mkt_cols,
                    index=_pick_col_idx(mkt_cols, mg.ticker_col if mg else None),
                    key="ov_m_ticker",
                )
                st.selectbox(
                    "Return column",
                    mkt_cols,
                    index=_pick_col_idx(mkt_cols, mg.ret_col if mg else None, _pick_ret_default_col(mkt_cols)),
                    key="ov_m_ret",
                )

        r_date = st.session_state.get("ov_r_date") or rev_cols[_pick_col_idx(rev_cols, rg.date_col if rg else None)]
        r_text = st.session_state.get("ov_r_text") or rev_cols[_pick_col_idx(rev_cols, rg.text_col if rg else None, 1)]
        r_ticker = st.session_state.get("ov_r_ticker") or rev_cols[_pick_col_idx(rev_cols, rg.ticker_col if rg else None, 2)]
        m_date = st.session_state.get("ov_m_date") or mkt_cols[_pick_col_idx(mkt_cols, mg.date_col if mg else None)]
        m_ticker = st.session_state.get("ov_m_ticker") or mkt_cols[_pick_col_idx(mkt_cols, mg.ticker_col if mg else None)]
        m_ret = st.session_state.get("ov_m_ret") or mkt_cols[_pick_col_idx(mkt_cols, mg.ret_col if mg else None, _pick_ret_default_col(mkt_cols))]

        st.markdown('<span class="rs-form-label">Options</span>', unsafe_allow_html=True)
        opt1, opt2, opt3 = st.columns(3)
        with opt1:
            run_transformer = st.toggle(
                f"Run {LABEL_NEURAL_SCORER.lower()} (neural model)",
                value=False,
                key="rs_run_tr",
                help="Requires `pip install -r requirements-ml.txt`. Adds contextual AI scorer and disagreement queue.",
            )
        with opt2:
            top_n = st.slider("Disagreement queue size", 3, 25, 10, key="rs_top_n")
        with opt3:
            polish = st.toggle(
                "Polish brief with LLM",
                value=False,
                key="rs_polish",
                help="Optional. Requires OPENAI_API_KEY. Same facts, richer phrasing.",
            )

        run = st.button("Run report", type="primary", width="stretch", key="rs_run_report")
    if not run:
        return

    reviews_df = st.session_state.reviews_df
    market_df = st.session_state.market_df
    if reviews_df is None or market_df is None:
        st.error("Load data first — use the built-in demo or upload both CSVs.")
        return

    company = (st.session_state.get("rs_company") or "").strip() or "LinguaLoop Ltd."
    with st.spinner("Scoring reviews, merging market data, computing associations…"):
        try:
            st.session_state.result = run_pipeline(
                reviews_df,
                market_df,
                company_name=company,
                review_date_col=r_date,
                review_text_col=r_text,
                review_ticker_col=r_ticker,
                market_date_col=m_date,
                market_ticker_col=m_ticker,
                market_ret_col=m_ret,
                run_transformer=run_transformer,
                top_disagreement=top_n,
                polish_brief_with_llm=polish,
            )
            st.session_state.report_company = company
            st.session_state.chat_messages = []
            st.rerun()
        except ValueError as e:
            st.session_state.result = None
            st.error(str(e))
        except Exception as e:
            st.session_state.result = None
            st.exception(e)




def render_results_dashboard() -> None:
    res = st.session_state.result
    assert res is not None
    company = st.session_state.get("report_company") or (st.session_state.get("rs_company") or "LinguaLoop Ltd.")
    company_safe = html.escape(str(company))

    # ── Header row (brand + company + back button) ─────────────────────────────
    try:
        hdr_label, hdr_btn = st.columns([5, 1], vertical_alignment="center")
    except TypeError:
        hdr_label, hdr_btn = st.columns([5, 1])
    with hdr_label:
        st.markdown(
            f'<p class="rs-hdr-title">'
            f'<span class="rs-topbar-badge">ReviewSignal</span>'
            f'&nbsp;<span class="rs-topbar-sep">/</span>&nbsp;'
            f'<span class="rs-topbar-company">{company_safe}</span>'
            f"</p>",
            unsafe_allow_html=True,
        )
    with hdr_btn:
        if st.button("← New run", key="rs_back", width="stretch"):
            st.session_state.result = None
            st.session_state.chat_messages = []
            st.session_state.rs_narr_key = ""
            st.session_state.rs_narr_md = None
            st.rerun()
    # Full-width border below the header row
    st.markdown(
        '<hr style="margin: 0 0 1.4rem 0; border: none; border-top: 1px solid rgba(20,184,166,0.18);">',
        unsafe_allow_html=True,
    )

    insight_facts = build_insight_facts(res)
    plain_insight_lines = plain_language_insights(res, company=company, insight_facts=insight_facts)
    technical_insight_lines = actionable_insights(res)
    insight_lines_for_pdf = plain_insight_lines + technical_insight_lines
    ctx_json = build_chat_context(company, res)
    disagreement_display = dedupe_disagreement_quotes(res.disagreement_quotes)
    chart_stats = chart_summary_stats(res.merged)
    stab_stats = stability_summary_stats(res.stability_df)
    chart_callouts = build_chart_callouts(res)

    span_review = _format_metric_date_span(res.stats.get("date_min"), res.stats.get("date_max"), empty="—")
    span_overlap = _format_metric_date_span(res.date_overlap_min, res.date_overlap_max, empty="—")
    help_review = (
        f"Full export range (ISO): {res.stats.get('date_min', '—')} through {res.stats.get('date_max', '—')}"
    )
    help_overlap = (
        f"Overlap dates (ISO): {res.date_overlap_min or '—'} through {res.date_overlap_max or '—'}"
    )

    tab_ov, tab_mkt, tab_rev, tab_thm, tab_brief, tab_qa = st.tabs(
        ["Overview", "Market & stats", "Reviews", "Themes", "Brief", "Q&A"]
    )

    # ── Overview ───────────────────────────────────────────────────────────────
    with tab_ov:
        # ── Surface-low warnings ─────────────────────────────────────────────
        surface_lows = build_surface_low_bullets(res, insight_facts)
        if surface_lows:
            st.markdown(
                '<div class="rs-signal-box" style="margin-bottom:0.75rem;border-color:rgba(217,119,6,0.5)">'
                "<strong>Worth a closer look on this run</strong></div>",
                unsafe_allow_html=True,
            )
            st.markdown("\n".join(f"- {b}" for b in surface_lows))

        # ── Headline signal ──────────────────────────────────────────────────
        r_val = res.corr_textblob_lag1
        if r_val is not None:
            direction_word = "positive" if r_val > 0 else "negative"
            strength = (
                "a pattern worth monitoring"
                if abs(r_val) >= 0.08
                else "a weak link — treat conclusions cautiously"
            )
            st.markdown(
                f'<div class="rs-signal-box">'
                f"<strong>{html.escape(company)}</strong> — <strong>{res.stats['n_reviews']:,} reviews</strong> over "
                f"<strong>{span_review}</strong> ({res.overlap_days} trading days overlapping your market file). "
                f"Review tone moves with next-day returns in a <strong>{direction_word}</strong> way "
                f"in this sample — {strength}. Association only — not proof that reviews caused price moves."
                f"</div>",
                unsafe_allow_html=True,
            )

        # ── Coverage metrics ──────────────────────────────────────────────────
        _section_title("Coverage", "Do we have enough reviews lined up with trading days?")
        r1a, r1b = st.columns(2)
        r1a.metric("Reviews", f"{res.stats['n_reviews']:,}")
        r1b.metric("Overlapping trading days", res.overlap_days)
        r2a, r2b = st.columns(2)
        r2a.metric("Review span", span_review, help=help_review)
        r2b.metric("Overlap window", span_overlap, help=help_overlap)

        if res.transformer_error:
            _terr = res.transformer_error
            if _terr.startswith("MISSING_PACKAGES"):
                _terr_msg = (
                    f"{LABEL_NEURAL_SCORER} is not installed. "
                    "Run `pip install -r requirements-ml.txt` in your terminal, then re-run the report."
                )
            elif _terr.startswith("NETWORK_BLOCKED"):
                _terr_msg = (
                    f"{LABEL_NEURAL_SCORER} was skipped — the HuggingFace model download was blocked "
                    "(proxy / firewall / 403). The packages are installed. "
                    "**Fix options:** (1) run on a network that allows huggingface.co, "
                    "(2) download the model on a different machine and set `TRANSFORMERS_OFFLINE=1`, "
                    "or (3) leave the toggle off — the TextBlob score is used for all analysis."
                )
            else:
                _terr_msg = (
                    f"{LABEL_NEURAL_SCORER} was skipped — {_terr}. "
                    "The TextBlob score is used for all analysis. Re-enable the toggle to retry."
                )
            st.warning(_terr_msg)

        # ── Key signals (4 cards) ──────────────────────────────────────────────
        _section_title("Key signals", "What this data is telling you")
        st.markdown(_key_findings_stack_html(plain_insight_lines), unsafe_allow_html=True)

        # ── Downloads ─────────────────────────────────────────────────────────
        dl1, dl2 = st.columns(2)
        with dl1:
            try:
                pdf_bytes = build_pdf_bytes(company, res, insight_lines_for_pdf)
                st.download_button(
                    "Download PDF report",
                    data=pdf_bytes,
                    file_name="reviewsignal_report.pdf",
                    mime="application/pdf",
                    width="stretch",
                    type="primary",
                )
            except Exception as e:
                st.warning(f"PDF unavailable ({e}). Install `pip install -r requirements-ui.txt`")
        with dl2:
            if disagreement_display:
                dq_csv = pd.DataFrame(disagreement_display).to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download disagreement queue (CSV)",
                    data=dq_csv,
                    file_name="disagreement_queue.csv",
                    mime="text/csv",
                    width="stretch",
                )

        # ── Best / Worst Day panel ─────────────────────────────────────────────
        _section_title("Best & worst days", "The market days that matter most — with the reviews from those days")
        best_day_str, best_ret_val, best_samples = best_day_review_samples(res, k=4)
        w_samples = worst_day_review_samples(res, k=4)

        col_best, col_worst = st.columns(2)
        with col_best:
            if best_day_str and best_ret_val is not None:
                bv_pct = float(best_ret_val) * 100
                st.markdown(
                    f'<div style="background:rgba(20,184,166,0.12);border:1px solid rgba(20,184,166,0.35);'
                    f'border-radius:8px;padding:0.75rem 1rem;margin-bottom:0.5rem;">'
                    f'<strong style="font-size:0.85rem;color:#94a3b8;">BEST DAY</strong><br>'
                    f'<span style="font-size:1.4rem;font-weight:700;color:#2dd4bf;">{best_day_str}</span><br>'
                    f'<span style="color:#5eead4;font-size:1rem;">+{bv_pct:.2f}%</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                with st.expander("Read what customers said ↗", expanded=False):
                    if best_samples:
                        for s in best_samples:
                            st.markdown(
                                f'<div style="border-left:3px solid rgba(20,184,166,0.5);padding:0.4rem 0.75rem;'
                                f'margin-bottom:0.5rem;font-size:0.88rem;color:#cbd5e1;">{html.escape(s["excerpt"])}</div>',
                                unsafe_allow_html=True,
                            )
                    else:
                        st.caption("No reviews found for this exact date — check the Reviews tab for nearby rows.")
            else:
                st.info("Best day could not be identified.")

        with col_worst:
            if res.worst_return_day and res.worst_return_value is not None:
                try:
                    wv_pct = float(res.worst_return_value) * 100
                    st.markdown(
                        f'<div style="background:rgba(217,119,6,0.1);border:1px solid rgba(217,119,6,0.4);'
                        f'border-radius:8px;padding:0.75rem 1rem;margin-bottom:0.5rem;">'
                        f'<strong style="font-size:0.85rem;color:#94a3b8;">WORST DAY</strong><br>'
                        f'<span style="font-size:1.4rem;font-weight:700;color:#f59e0b;">{res.worst_return_day}</span><br>'
                        f'<span style="color:#fbbf24;font-size:1rem;">{wv_pct:.2f}%</span>'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
                except (TypeError, ValueError):
                    st.markdown(f"**Worst day:** {res.worst_return_day}")
                with st.expander("Read what customers said ↘", expanded=False):
                    if w_samples:
                        for s in w_samples:
                            st.markdown(
                                f'<div style="border-left:3px solid rgba(217,119,6,0.5);padding:0.4rem 0.75rem;'
                                f'margin-bottom:0.5rem;font-size:0.88rem;color:#cbd5e1;">{html.escape(s["excerpt"])}</div>',
                                unsafe_allow_html=True,
                            )
                    else:
                        st.caption("No reviews found for this exact date — check adjacent dates in Reviews.")
            else:
                st.info("Worst day could not be identified.")

    # ── Market & stats ──────────────────────────────────────────────────────────
    with tab_mkt:
        # ── Tone over time ────────────────────────────────────────────────────
        tq, td, tc = build_market_trajectory_beats(company, chart_stats, insight_facts)
        _section_title("Review tone over time", "Did customer mood drift over the window?")
        st.plotly_chart(fig_dual_sentiment(res.merged), width="stretch")
        if chart_callouts.get("trajectory"):
            st.info(chart_callouts["trajectory"])
        st.markdown(tc)
        with st.expander("How we read this chart", expanded=False):
            st.markdown(tq)
            st.markdown(f"*{td}*")
            st.caption(LAG_EXPLAINER)
            st.markdown(insight_trajectory_section_markdown(insight_facts))
            st.markdown(build_technical_strip_trajectory(res, chart_stats))

        # ── Association ───────────────────────────────────────────────────────
        aq, ad, ac = build_market_association_beats(company, res, insight_facts)
        _section_title(
            "Does mood line up with stock moves?",
            "When reviews were warmer yesterday, did the stock tend to do better today?",
        )
        _fig_scatter = fig_association_scatter(res.reg, worst_day=res.worst_return_day)
        if _fig_scatter is not None:
            st.plotly_chart(_fig_scatter, width="stretch")
        st.markdown(
            f'<div class="rs-signal-box" style="margin-top:0.35rem">{association_explanation_html(res.corr_textblob_lag1)}</div>',
            unsafe_allow_html=True,
        )
        if chart_callouts.get("association"):
            _assoc_callout = chart_callouts["association"].replace(
                "the dots tend to line up — warmer reviews, better next day",
                "each dot is one trading day. Dots in the upper-right = warm reviews + good stock day; lower-left = same pattern. Dots in the upper-left or lower-right are days the pattern broke"
            )
            st.info(_assoc_callout)
        st.markdown(ac)

        if res.corr_textblob_lag1 is None and res.regression_full_textblob is None and len(res.reg) >= 8:
            st.warning(
                "Numbers didn't compute — yesterday's average dictionary score has almost no variation. "
                "Check column overrides on the landing page and re-run."
            )

        rf = res.regression_full_textblob
        rf2 = res.regression_full_transformer
        with st.expander(f"{TECH_STRIP_TITLE} — exact numbers", expanded=False):
            a1, a2, a3, a4 = st.columns(4)
            a1.metric("Lag-1 r · dictionary", _fmt_num(res.corr_textblob_lag1, 4), help="Pearson correlation, full overlap window")
            a2.metric("Lag-1 r · contextual AI", _fmt_num(res.corr_transformer_lag1, 4))
            a3.metric("OLS slope · dictionary", _fmt_num(rf["coef"], 6) if rf else "n/a", help="Next-day ret ~ lag-1 sentiment; HC3 robust SEs")
            a4.metric("R² · dictionary", _fmt_num(rf["r2"], 4) if rf else "n/a")
            if rf:
                st.caption(
                    f"{LABEL_LEXICON_SCORER} — 95% CI [{_fmt_num(rf['ci_low'], 6)}, {_fmt_num(rf['ci_high'], 6)}]"
                    f"  ·  p = {_fmt_num(rf['p_value'], 4)}  ·  n = {rf['n']} days"
                )
            if rf2 or res.reg["avg_transformer_lag1"].notna().any():
                b1, b2 = st.columns(2)
                b1.metric("OLS slope · contextual AI", _fmt_num(rf2["coef"], 6) if rf2 else "n/a")
                b2.metric("R² · contextual AI", _fmt_num(rf2["r2"], 4) if rf2 else "n/a")
                if rf2:
                    st.caption(
                        f"{LABEL_NEURAL_SCORER} — 95% CI [{_fmt_num(rf2['ci_low'], 6)}, {_fmt_num(rf2['ci_high'], 6)}]"
                        f"  ·  p = {_fmt_num(rf2['p_value'], 4)}  ·  n = {rf2['n']} days"
                    )
            st.markdown(aq)
            st.markdown(f"*{ad}*")
            st.markdown(insight_association_section_markdown(insight_facts))
            st.markdown(build_technical_strip_association(res))

        # ── Stability ─────────────────────────────────────────────────────────
        sq, sd, sc = build_market_stability_beats(company, res, insight_facts, res.stability_df)
        _section_title(
            "Is the pattern stable?",
            "Has the link between reviews and returns been consistent, or did it shift recently?",
        )
        show_neural_stab = st.checkbox(
            "Show contextual AI stability curves (dictionary lines stay visible)",
            value=False,
            key="rs_stab_neural_traces",
        )
        f_stab = fig_correlation_stability(res.stability_df, show_neural_traces=show_neural_stab)
        if f_stab is not None:
            st.plotly_chart(f_stab, width="stretch")
        else:
            st.info("Not enough overlapping days for stability curves (need at least 15 after lag).")
        if chart_callouts.get("stability"):
            st.info(chart_callouts["stability"])
        st.markdown(sc)
        with st.expander("How we read this chart", expanded=False):
            st.markdown(sq)
            st.markdown(f"*{sd}*")
            st.markdown(insight_stability_section_markdown(insight_facts))
            st.markdown(build_technical_strip_stability(res, stab_stats))

        # ── Holdout sanity check ──────────────────────────────────────────────
        _section_title(
            "Did the pattern hold on the most recent days?",
            "We trained on older data, then tested on the newest — did the rule survive?",
        )
        hc1, hc2 = st.columns(2)
        with hc1:
            if res.holdout_textblob:
                h = res.holdout_textblob
                st.markdown(f"**{LABEL_LEXICON_SCORER}**")
                hc1a, hc1b = st.columns(2)
                hc1a.metric("Train days", h["n_train"])
                hc1b.metric("Holdout days", h["n_holdout"])
                hc1c, hc1d = st.columns(2)
                hc1c.metric("Slope p-value (HC3)", _fmt_num(h["p_value_lag_sentiment"], 4))
                hc1d.metric("Holdout corr(pred, actual)", _fmt_num(h.get("corr_pred_vs_actual_holdout"), 4))
                st.metric("Holdout RMSE", _fmt_num(h["rmse_holdout"], 6))
            else:
                st.info("Holdout not run (need sufficient history).")
        with hc2:
            if res.holdout_transformer:
                h = res.holdout_transformer
                st.markdown(f"**{LABEL_NEURAL_SCORER}**")
                hc2a, hc2b = st.columns(2)
                hc2a.metric("Train days", h["n_train"])
                hc2b.metric("Holdout days", h["n_holdout"])
                hc2c, hc2d = st.columns(2)
                hc2c.metric("Slope p-value (HC3)", _fmt_num(h["p_value_lag_sentiment"], 4))
                hc2d.metric("Holdout corr(pred, actual)", _fmt_num(h.get("corr_pred_vs_actual_holdout"), 4))
                st.metric("Holdout RMSE", _fmt_num(h["rmse_holdout"], 6))
            else:
                st.info(f"{LABEL_NEURAL_SCORER} holdout unavailable.")
        with st.expander("What do these numbers mean?", expanded=False):
            st.markdown(holdout_intro_markdown())
            st.markdown(insight_holdout_section_markdown(insight_facts))
            st.markdown(build_technical_strip_holdout(res))

    # ── Reviews ────────────────────────────────────────────────────────────────
    with tab_rev:
        _section_title(
            "Disagreement queue",
            f"Where {LABEL_LEXICON_SCORER} and {LABEL_NEURAL_SCORER} disagree most — usually the reviews worth reading first",
        )
        if disagreement_display:
            st.markdown(
                '<div class="rs-signal-box" style="margin-bottom:1rem">'
                f"These rows are your shortest path to nuance: when the <strong>{LABEL_LEXICON_SCORER}</strong> and "
                f"<strong>{LABEL_NEURAL_SCORER}</strong> disagree sharply, the text is often sarcastic, negated, "
                "or domain-specific — exactly where star counts mislead."
                "</div>",
                unsafe_allow_html=True,
            )
            dq = pd.DataFrame(disagreement_display)
            st.dataframe(
                dq[["date", "disagreement", "textblob", "transformer", "text"]],
                width="stretch",
                hide_index=True,
            )
            with st.expander(f"{TECH_STRIP_TITLE} — disagreement table columns", expanded=False):
                st.markdown(build_technical_strip_reviews())
        else:
            st.markdown(
                '<div class="rs-signal-box">'
                f"<strong>Turn on {LABEL_NEURAL_SCORER}</strong> on the start page to unlock this queue.<br>"
                "We rank reviews where the two scorers diverge most — the fastest way to find language that "
                "a simple word list would miss."
                "</div>",
                unsafe_allow_html=True,
            )

    # ── Themes ──────────────────────────────────────────────────────────────────
    with tab_thm:
        tm = res.text_mining
        if tm and (tm.get("bigrams") or tm.get("nmf_topics")):
            _section_title(
                "Themes in unusual days",
                tm.get("source_description", "Reviews flagged around big mover days and scorer disagreement"),
            )
            st.markdown(
                '<div class="rs-signal-box" style="margin-bottom:1rem">'
                f"<strong>{tm.get('n_docs', 0)} reviews</strong> in a focused corpus (high disagreement and/or "
                "extreme returns). <strong>Word pairs</strong> show what customers say together; "
                "<strong>topic clusters</strong> surface recurring storylines you can turn into support macros, FAQs, or release notes."
                "</div>",
                unsafe_allow_html=True,
            )
            st.markdown(themes_interpretation_markdown(company, tm))
            c_left, c_right = st.columns(2)
            with c_left:
                if tm.get("bigrams"):
                    st.markdown("**Frequent word pairs**")
                    st.dataframe(pd.DataFrame(tm["bigrams"]), width="stretch", hide_index=True)
            with c_right:
                if tm.get("nmf_topics"):
                    st.markdown("**Topic clusters (NMF)**")
                    st.dataframe(pd.DataFrame(tm["nmf_topics"]), width="stretch", hide_index=True)
            with st.expander(f"{TECH_STRIP_TITLE} — topic model & corpus", expanded=False):
                st.markdown(build_technical_strip_themes(tm))
        else:
            _section_title("Themes in unusual days", "Needs both scorers and enough flagged text")
            st.markdown(
                '<div class="rs-signal-box">'
                f"<strong>Enable {LABEL_NEURAL_SCORER}</strong> and re-run.<br>"
                "We build a small corpus where reviews sit next to big market moves or scorer fights, then "
                "extract word pairs and topics — operational language for product and support, not just charts."
                "</div>",
                unsafe_allow_html=True,
            )

    # ── Brief ───────────────────────────────────────────────────────────────────
    with tab_brief:
        _section_title("Brief", "The plain-English version — ready to share or paste into a meeting deck")
        # Compute narrative payload (LLM if key available, else plain-English template)
        _payload = narrative_payload(company, res, insight_facts=insight_facts)
        _nk = narrative_cache_key(_payload)
        if st.session_state.get("rs_narr_key") != _nk:
            st.session_state.rs_narr_key = _nk
            st.session_state.rs_narr_md = try_executive_narrative_markdown(_payload)
        _narr = st.session_state.get("rs_narr_md")

        # Primary content: plain-English 3-bullet card
        _plain_narrative = _narr if _narr else fallback_executive_narrative(_payload)
        st.markdown(_plain_narrative)
        if _narr:
            st.caption("Written by your AI key using only facts from this run — no invented statistics.")
        else:
            st.caption("Template summary — set OPENAI_API_KEY for an AI-written version using these same facts.")

        # Dig deeper expander
        with st.expander("Dig deeper — full statistical memo", expanded=False):
            st.markdown(overview_ir_markdown(company, res, insight_facts))
            st.markdown("---")
            st.markdown(res.brief_markdown)
            if res.brief_polished:
                st.markdown("---")
                st.markdown("**LLM-polished version (same facts):**")
                st.markdown(res.brief_polished)

        with st.expander(f"{TECH_STRIP_TITLE} — regression, HC3, holdout", expanded=False):
            for line in technical_insight_lines:
                st.markdown(f"- {line}")

    # ── Q&A ─────────────────────────────────────────────────────────────────────
    with tab_qa:
        _section_title("Grounded Q&A", "Answers use only this run — no invented statistics")
        if llm_client_configured():
            st.caption("LLM mode — answers generated from the run context via OpenAI-compatible API.")
        else:
            st.caption(
                "Rule-based mode — answers use the computed statistics from this run. "
                "Add `OPENAI_API_KEY` for richer, conversational responses."
            )
        with st.expander("Not sure what to ask? Try this →"):
            st.markdown(
                f"*Why do several top disagreement rows show the **{LABEL_LEXICON_SCORER}** near neutral "
                f"while **{LABEL_NEURAL_SCORER}** is strongly negative on the same text?*"
            )
        with st.expander(f"{TECH_STRIP_TITLE} — regression snapshot for Q&A", expanded=False):
            st.markdown(
                "Q&A pulls from the same JSON snapshot as the charts. "
                "For raw coefficients, p-values, and method names, use the expanders on each tab or the list below."
            )
            st.markdown(build_technical_strip_association(res))

        for msg in st.session_state.chat_messages:
            avatar = "📊" if msg["role"] == "assistant" else "💬"
            with st.chat_message(msg["role"], avatar=avatar):
                st.markdown(msg["content"])

        if prompt := st.chat_input("Ask about this run…"):
            st.session_state.chat_messages.append({"role": "user", "content": prompt})
            answer = chat_complete(ctx_json, prompt)
            st.session_state.chat_messages.append({"role": "assistant", "content": answer})
            st.rerun()


def main() -> None:
    st.set_page_config(
        page_title="ReviewSignal",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    if "result" not in st.session_state:
        st.session_state.result = None
    if "reviews_df" not in st.session_state:
        st.session_state.reviews_df = None
    if "market_df" not in st.session_state:
        st.session_state.market_df = None
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "data_mode" not in st.session_state:
        st.session_state.data_mode = "builtin"
    if "rs_company" not in st.session_state:
        st.session_state.rs_company = "LinguaLoop Ltd."
    if "report_company" not in st.session_state:
        st.session_state.report_company = "LinguaLoop Ltd."
    if "rs_narr_key" not in st.session_state:
        st.session_state.rs_narr_key = ""
    if "rs_narr_md" not in st.session_state:
        st.session_state.rs_narr_md = None

    if st.session_state.reviews_df is None and st.session_state.market_df is None:
        try:
            st.session_state.reviews_df, st.session_state.market_df = load_builtin_demo()
            st.session_state.data_mode = "builtin"
        except FileNotFoundError:
            pass

    _inject_brand_style(landing=st.session_state.result is None)

    if st.session_state.result is None:
        render_landing_page()
        return

    render_results_dashboard()


if __name__ == "__main__":
    main()
