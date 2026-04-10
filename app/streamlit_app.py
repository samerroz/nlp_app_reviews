"""
Sentiment and market association dashboard (Streamlit PoC).

From repo root:
  pip install -r requirements.txt -r requirements-ui.txt
  # optional: pip install -r requirements-ml.txt
  streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from pipeline import run_pipeline  # noqa: E402


def load_sample_pair() -> tuple[pd.DataFrame, pd.DataFrame]:
    rev = pd.read_csv(ROOT / "sample_data" / "reviews_sample.csv")
    mkt = pd.read_csv(ROOT / "sample_data" / "market_sample.csv")
    return rev, mkt


def fig_dual_sentiment(merged: pd.DataFrame) -> go.Figure:
    d = merged.sort_values("date")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=d["date"],
            y=d["avg_textblob"],
            mode="lines+markers",
            name="TextBlob (daily avg)",
            line=dict(width=2),
        )
    )
    if d["avg_transformer"].notna().any():
        fig.add_trace(
            go.Scatter(
                x=d["date"],
                y=d["avg_transformer"],
                mode="lines+markers",
                name="RoBERTa (daily avg)",
                line=dict(width=2, dash="dash"),
            )
        )
    fig.update_layout(
        margin=dict(l=20, r=20, t=36, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis_title="Sentiment",
        xaxis_title="Date",
        height=400,
        template="plotly_white",
    )
    return fig


def main() -> None:
    st.set_page_config(
        page_title="ReviewSignal",
        page_icon=":chart_with_upwards_trend:",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("ReviewSignal")
    st.caption(
        "Upload **review exports** + **daily returns** to get dual NLP sentiment, lag-1 association, "
        "and a **grounded** executive brief (template; optional API polish)."
    )

    if "result" not in st.session_state:
        st.session_state.result = None
    if "reviews_df" not in st.session_state:
        st.session_state.reviews_df = None
    if "market_df" not in st.session_state:
        st.session_state.market_df = None

    with st.sidebar:
        st.header("Run")
        company = st.text_input("Company name (for brief)", value="LinguaLoop Ltd.")
        use_sample = st.button("Load built-in sample data", use_container_width=True)
        if use_sample:
            st.session_state.reviews_df, st.session_state.market_df = load_sample_pair()
            st.success("Sample loaded.")

        st.divider()
        st.subheader("Or upload CSVs")
        up_rev = st.file_uploader("Reviews CSV", type=["csv"])
        up_mkt = st.file_uploader("Market CSV (date, ticker, ret)", type=["csv"])

        if up_rev is not None:
            st.session_state.reviews_df = pd.read_csv(up_rev)
        if up_mkt is not None:
            st.session_state.market_df = pd.read_csv(up_mkt)

        reviews_df = st.session_state.reviews_df
        market_df = st.session_state.market_df

        rev_cols = list(reviews_df.columns) if reviews_df is not None else []
        mkt_cols = list(market_df.columns) if market_df is not None else []

        st.subheader("Column mapping")
        r_date = st.selectbox(
            "Review date column",
            rev_cols,
            index=rev_cols.index("date") if "date" in rev_cols else 0,
        )
        r_text = st.selectbox(
            "Review text column",
            rev_cols,
            index=rev_cols.index("review_text") if "review_text" in rev_cols else 0,
        )
        r_ticker = st.selectbox(
            "Ticker column",
            rev_cols,
            index=rev_cols.index("ticker") if "ticker" in rev_cols else 0,
        )

        m_date = st.selectbox(
            "Market date column",
            mkt_cols,
            index=mkt_cols.index("date") if "date" in mkt_cols else 0,
        )
        m_ticker = st.selectbox(
            "Market ticker column",
            mkt_cols,
            index=mkt_cols.index("ticker") if "ticker" in mkt_cols else 0,
        )

        def pick_ret_default(cols: list[str]) -> int:
            for name in ("ret", "RET", "return", "Return"):
                if name in cols:
                    return cols.index(name)
            return min(2, len(cols) - 1) if cols else 0

        m_ret = st.selectbox(
            "Return column",
            mkt_cols,
            index=pick_ret_default(mkt_cols) if mkt_cols else 0,
        )

        run_transformer = st.toggle("Run RoBERTa sentiment (needs requirements-ml.txt)", value=True)
        top_n = st.slider("Disagreement queue size", 3, 25, 10)
        polish = st.toggle("Polish brief with OpenAI if OPENAI_API_KEY set", value=True)

        run = st.button("Run analysis", type="primary", use_container_width=True)

    if run:
        if reviews_df is None or market_df is None:
            st.error("Load sample data or upload both CSVs.")
        else:
            with st.spinner("Scoring reviews and merging with market data…"):
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
                except ValueError as e:
                    st.session_state.result = None
                    st.error(str(e))
                except Exception as e:
                    st.session_state.result = None
                    st.exception(e)

    res = st.session_state.result
    if res is None:
        st.info("Load data in the sidebar, then click **Run analysis**.")
        return

    st.subheader("Data health")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Reviews", f"{res.stats['n_reviews']:,}")
    c2.metric("Overlapping trading days", res.overlap_days)
    c3.metric("Review span", f"{res.stats['date_min']} → {res.stats['date_max']}")
    c4.metric("Overlap window", f"{res.date_overlap_min or '—'} → {res.date_overlap_max or '—'}")

    if res.transformer_error:
        st.warning(
            f"RoBERTa not available ({res.transformer_error}). "
            "Install: `pip install -r requirements-ml.txt` and set `HF_HOME` if needed."
        )

    st.subheader("Dual sentiment over time")
    st.plotly_chart(fig_dual_sentiment(res.merged), use_container_width=True)

    st.subheader("Association (lag-1 daily sentiment vs same-day return)")
    a1, a2 = st.columns(2)
    a1.metric(
        "Pearson · TextBlob",
        f"{res.corr_textblob_lag1:.4f}" if res.corr_textblob_lag1 is not None else "n/a",
    )
    a2.metric(
        "Pearson · RoBERTa",
        f"{res.corr_transformer_lag1:.4f}" if res.corr_transformer_lag1 is not None else "n/a",
    )
    st.caption(
        "Association is not causation. Small samples give unstable correlations — fine for workflow demo; "
        "cite your long-sample study in the course document."
    )

    st.subheader("Read these first (largest TextBlob vs RoBERTa gap)")
    if res.disagreement_quotes:
        dq = pd.DataFrame(res.disagreement_quotes)
        st.dataframe(
            dq[["date", "disagreement", "textblob", "transformer", "text"]],
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("No transformer scores — enable RoBERTa or check ML install.")

    st.subheader("Executive brief")
    st.markdown(res.brief_markdown)
    if res.brief_polished:
        with st.expander("API-polished wording (same facts)"):
            st.markdown(res.brief_polished)


if __name__ == "__main__":
    main()
