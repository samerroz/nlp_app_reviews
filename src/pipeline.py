"""
Importable review × market pipeline: score → daily → merge → lag → stats → brief.
Used by demo_pipeline CLI and Streamlit app.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from llm_brief import build_stats_payload, maybe_llm_polish_brief, render_template_brief
from sentiment_models import polarity_textblob, try_transformer_scores


@dataclass
class PipelineResult:
    reviews_scored: pd.DataFrame
    daily: pd.DataFrame
    merged: pd.DataFrame
    reg: pd.DataFrame
    corr_textblob_lag1: float | None
    corr_transformer_lag1: float | None
    mean_abs_disagreement: float | None
    worst_return_day: str | None
    worst_return_value: float | None
    disagreement_quotes: list[dict[str, Any]]
    stats: dict[str, Any]
    brief_markdown: str
    brief_polished: str | None
    transformer_error: str | None
    overlap_days: int
    date_overlap_min: str | None
    date_overlap_max: str | None


def prepare_reviews(
    df: pd.DataFrame,
    date_col: str = "date",
    text_col: str = "review_text",
    ticker_col: str = "ticker",
) -> pd.DataFrame:
    out = df.copy()
    if date_col != "date":
        out = out.rename(columns={date_col: "date"})
    if text_col != "review_text":
        out = out.rename(columns={text_col: "review_text"})
    if ticker_col != "ticker":
        out = out.rename(columns={ticker_col: "ticker"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out = out.dropna(subset=["date", "review_text", "ticker"])
    return out


def prepare_market(
    df: pd.DataFrame,
    date_col: str = "date",
    ticker_col: str = "ticker",
    ret_col: str = "ret",
) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    d, t, r = date_col.strip().lower(), ticker_col.strip().lower(), ret_col.strip().lower()
    missing = [x for x in (d, t, r) if x not in out.columns]
    if missing:
        raise ValueError(f"Market CSV missing columns: {missing}. Found: {list(out.columns)}")
    out = out.rename(columns={d: "date", t: "ticker", r: "ret"})
    out["date"] = pd.to_datetime(out["date"], errors="coerce").dt.normalize()
    out["ret"] = pd.to_numeric(out["ret"], errors="coerce")
    out = out.dropna(subset=["date", "ticker", "ret"])
    return out


def score_reviews_dataframe(
    reviews: pd.DataFrame,
    *,
    run_transformer: bool = True,
) -> tuple[pd.DataFrame, str | None]:
    """Add sentiment_textblob and sentiment_transformer columns. Returns (df, error_or_none)."""
    out = reviews.copy()
    texts = out["review_text"].astype(str).tolist()
    out["sentiment_textblob"] = [polarity_textblob(t) for t in texts]
    tr_err: str | None = None
    if run_transformer:
        tr_scores, err = try_transformer_scores(texts)
        if tr_scores is None:
            out["sentiment_transformer"] = np.nan
            tr_err = err
        else:
            out["sentiment_transformer"] = tr_scores
    else:
        out["sentiment_transformer"] = np.nan
    return out, tr_err


def daily_aggregates(reviews_scored: pd.DataFrame) -> pd.DataFrame:
    return reviews_scored.groupby(["date", "ticker"], as_index=False).agg(
        avg_textblob=("sentiment_textblob", "mean"),
        avg_transformer=("sentiment_transformer", "mean"),
        review_count=("review_text", "count"),
    )


def merge_with_market(daily: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    return daily.merge(market, on=["date", "ticker"], how="inner")


def add_lag1_sentiment(merged: pd.DataFrame) -> pd.DataFrame:
    out = merged.copy()
    for col in ("avg_textblob", "avg_transformer"):
        out[f"{col}_lag1"] = out.groupby("ticker")[col].shift(1)
    return out


def compute_correlations(reg: pd.DataFrame) -> tuple[float | None, float | None]:
    corr_tb: float | None = None
    corr_tr: float | None = None
    r = reg.dropna(subset=["avg_textblob_lag1"])
    if len(r) >= 3:
        corr_tb = float(r["avg_textblob_lag1"].corr(r["ret"]))
    sub = reg.dropna(subset=["avg_transformer_lag1"])
    if len(sub) >= 3:
        corr_tr = float(sub["avg_transformer_lag1"].corr(sub["ret"]))
    return corr_tb, corr_tr


def pick_disagreement_quotes(
    reviews_scored: pd.DataFrame,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    rev = reviews_scored.copy()
    if not rev["sentiment_transformer"].notna().any():
        return []
    rev["disagreement"] = (rev["sentiment_textblob"] - rev["sentiment_transformer"]).abs()
    top = rev.sort_values("disagreement", ascending=False).head(top_k)
    return [
        {
            "date": str(r["date"].date()),
            "text": str(r["review_text"])[:500],
            "textblob": float(r["sentiment_textblob"]),
            "transformer": float(r["sentiment_transformer"]),
            "disagreement": float(r["disagreement"]),
        }
        for _, r in top.iterrows()
    ]


def run_pipeline(
    reviews_raw: pd.DataFrame,
    market_raw: pd.DataFrame,
    *,
    company_name: str = "LinguaLoop Ltd.",
    review_date_col: str = "date",
    review_text_col: str = "review_text",
    review_ticker_col: str = "ticker",
    market_date_col: str = "date",
    market_ticker_col: str = "ticker",
    market_ret_col: str = "ret",
    run_transformer: bool = True,
    top_disagreement: int = 10,
    polish_brief_with_llm: bool = True,
) -> PipelineResult:
    reviews = prepare_reviews(
        reviews_raw,
        date_col=review_date_col,
        text_col=review_text_col,
        ticker_col=review_ticker_col,
    )
    market = prepare_market(
        market_raw,
        date_col=market_date_col,
        ticker_col=market_ticker_col,
        ret_col=market_ret_col,
    )

    reviews_scored, transformer_err = score_reviews_dataframe(reviews, run_transformer=run_transformer)
    daily = daily_aggregates(reviews_scored)
    merged = merge_with_market(daily, market)
    if merged.empty:
        raise ValueError(
            "No overlapping date+ticker rows between daily review aggregates and market data. "
            "Check that tickers match (e.g. DEMO vs AAPL) and calendar dates align."
        )
    merged = add_lag1_sentiment(merged)
    reg = merged.dropna(subset=["avg_textblob_lag1"])

    corr_tb, corr_tr = compute_correlations(merged)

    mean_dis: float | None = None
    if reviews_scored["sentiment_transformer"].notna().any():
        mean_dis = float(
            (reviews_scored["sentiment_textblob"] - reviews_scored["sentiment_transformer"])
            .abs()
            .mean()
        )

    worst_idx = reg["ret"].idxmin() if len(reg) else None
    worst_day = str(reg.loc[worst_idx, "date"].date()) if worst_idx is not None else None
    worst_ret = float(reg.loc[worst_idx, "ret"]) if worst_idx is not None else None

    quotes = pick_disagreement_quotes(reviews_scored, top_k=top_disagreement)

    overlap_min = merged["date"].min() if len(merged) else None
    overlap_max = merged["date"].max() if len(merged) else None

    stats = build_stats_payload(
        n_reviews=len(reviews_scored),
        date_min=str(reviews_scored["date"].min().date()) if len(reviews_scored) else "",
        date_max=str(reviews_scored["date"].max().date()) if len(reviews_scored) else "",
        corr_textblob_lag1=corr_tb,
        corr_transformer_lag1=corr_tr,
        mean_abs_disagreement=mean_dis,
        worst_return_day=worst_day,
        worst_return_value=worst_ret,
    )

    brief = render_template_brief(company_name, stats, quotes[:3])
    polished: str | None = None
    if polish_brief_with_llm:
        polished = maybe_llm_polish_brief(brief, stats, json.dumps(quotes[:3], ensure_ascii=False))

    return PipelineResult(
        reviews_scored=reviews_scored,
        daily=daily,
        merged=merged,
        reg=reg,
        corr_textblob_lag1=corr_tb,
        corr_transformer_lag1=corr_tr,
        mean_abs_disagreement=mean_dis,
        worst_return_day=worst_day,
        worst_return_value=worst_ret,
        disagreement_quotes=quotes,
        stats=stats,
        brief_markdown=brief,
        brief_polished=polished,
        transformer_error=transformer_err,
        overlap_days=len(merged),
        date_overlap_min=str(overlap_min.date()) if overlap_min is not None else None,
        date_overlap_max=str(overlap_max.date()) if overlap_max is not None else None,
    )
