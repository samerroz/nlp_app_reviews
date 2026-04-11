"""
Importable review × market pipeline: score → daily → merge → lag → stats → brief.
Used by demo_pipeline CLI and Streamlit app.
"""
from __future__ import annotations

import json
import math
import re
import unicodedata
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from llm_brief import build_stats_payload, maybe_llm_polish_brief, render_template_brief
from sentiment_models import polarity_textblob, try_transformer_scores
from text_mining import run_text_mining
from time_series_eval import build_stability_frame, full_sample_ols_hc3, holdout_ols_hc3


def _dedupe_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge duplicate column labels. Keeping only the *first* column often leaves an empty
    `review_text` and zeros out sentiment variation (NaN correlations / OLS). We coalesce:
    for `review_text`, prefer the column with the most non-empty characters; else fillna left-to-right.
    """
    if not df.columns.duplicated().any():
        return df
    pieces: dict[str, pd.Series] = {}
    for name in pd.unique(df.columns):
        block = df.loc[:, name]
        if isinstance(block, pd.DataFrame):
            if block.shape[1] == 0:
                continue
            if str(name) == "review_text":
                order = sorted(
                    range(block.shape[1]),
                    key=lambda j: (
                        block.iloc[:, j].astype(str).str.len().sum(),
                        int(block.iloc[:, j].notna().sum()),
                    ),
                    reverse=True,
                )
                s = block.iloc[:, order[0]].copy()
                for j in order[1:]:
                    oth = block.iloc[:, j]
                    blank = s.isna() | (s.astype(str).str.strip() == "") | (
                        s.astype(str).str.strip().str.lower() == "nan"
                    )
                    s = s.where(~blank, oth)
            else:
                s = block.iloc[:, 0].copy()
                for j in range(1, block.shape[1]):
                    s = s.fillna(block.iloc[:, j])
            pieces[str(name)] = s
        else:
            pieces[str(name)] = block
    return pd.DataFrame(pieces, index=df.index)


def _finite_pearson(sx: pd.Series, sy: pd.Series) -> float | None:
    """Pearson r only when both sides vary; else None (avoid NaN in UI)."""
    m = pd.DataFrame({"x": sx, "y": sy}).dropna()
    if len(m) < 3:
        return None
    std_x = float(m["x"].std(ddof=1))
    std_y = float(m["y"].std(ddof=1))
    if not math.isfinite(std_x) or not math.isfinite(std_y) or std_x < 1e-9 or std_y < 1e-9:
        return None
    c = m["x"].corr(m["y"])
    if c is None or not isinstance(c, (float, np.floating)) or not math.isfinite(float(c)):
        return None
    return float(c)


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
    stability_df: pd.DataFrame | None
    holdout_textblob: dict[str, Any] | None
    holdout_transformer: dict[str, Any] | None
    regression_full_textblob: dict[str, Any] | None
    regression_full_transformer: dict[str, Any] | None
    text_mining: dict[str, Any] | None


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
    out = _dedupe_columns(out)
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
    out = _dedupe_columns(out)
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
    rt = out["review_text"]
    if isinstance(rt, pd.DataFrame):
        order = sorted(
            range(rt.shape[1]),
            key=lambda j: (rt.iloc[:, j].astype(str).str.len().sum(), int(rt.iloc[:, j].notna().sum())),
            reverse=True,
        )
        s = rt.iloc[:, order[0]].copy()
        for j in order[1:]:
            oth = rt.iloc[:, j]
            blank = s.isna() | (s.astype(str).str.strip() == "") | (s.astype(str).str.strip().str.lower() == "nan")
            s = s.where(~blank, oth)
        rt = s
    texts = rt.astype(str).tolist()
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


def _sanitize_regression_dict(d: dict[str, Any] | None) -> dict[str, Any] | None:
    if not d:
        return None
    try:
        keys = ("coef", "ci_low", "ci_high", "p_value", "r2")
        if not all(math.isfinite(float(d[k])) for k in keys):
            return None
    except (KeyError, TypeError, ValueError):
        return None
    return d


def _sanitize_holdout_dict(h: dict[str, Any] | None) -> dict[str, Any] | None:
    if not h:
        return None
    try:
        if not math.isfinite(float(h["p_value_lag_sentiment"])):
            return None
        if not math.isfinite(float(h["coef_lag_sentiment"])):
            return None
    except (KeyError, TypeError, ValueError):
        return None
    return h


def add_lag1_sentiment(merged: pd.DataFrame) -> pd.DataFrame:
    # shift(1) must follow calendar order within each ticker
    out = merged.sort_values(["ticker", "date"], kind="mergesort").reset_index(drop=True).copy()
    for col in ("avg_textblob", "avg_transformer"):
        out[f"{col}_lag1"] = out.groupby("ticker", sort=False)[col].shift(1)
    return out


def compute_correlations(reg: pd.DataFrame) -> tuple[float | None, float | None]:
    r = reg.dropna(subset=["avg_textblob_lag1", "ret"])
    corr_tb = _finite_pearson(r["avg_textblob_lag1"], r["ret"])
    sub = reg.dropna(subset=["avg_transformer_lag1", "ret"])
    corr_tr = _finite_pearson(sub["avg_transformer_lag1"], sub["ret"])
    return corr_tb, corr_tr


def disagreement_text_key(raw: str) -> str:
    """Stable key for deduping review text (NFKC, strip invisible / bidi filler, collapse whitespace)."""
    t = unicodedata.normalize("NFKC", str(raw))
    t = re.sub(r"[\u200b-\u200f\u2028\u2029\ufeff\u2060\u00a0]", " ", t)
    t = re.sub(r"\s+", " ", t.strip().lower())
    return t[:500]


def dedupe_disagreement_quotes(quotes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep first occurrence per normalized text (e.g. stale Streamlit session or upstream duplicates)."""
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for q in quotes:
        k = disagreement_text_key(str(q.get("text", "")))
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(q)
    return out


def pick_disagreement_quotes(
    reviews_scored: pd.DataFrame,
    top_k: int = 10,
) -> list[dict[str, Any]]:
    """Rank by disagreement; keep up to top_k rows with **distinct** review text (demo CSVs often repeat templates)."""
    rev = reviews_scored.copy()
    if not rev["sentiment_transformer"].notna().any():
        return []
    rev["disagreement"] = (rev["sentiment_textblob"] - rev["sentiment_transformer"]).abs()
    rev = rev.sort_values("disagreement", ascending=False)
    seen_text: set[str] = set()
    out: list[dict[str, Any]] = []
    for _, r in rev.iterrows():
        raw = str(r["review_text"])
        key = disagreement_text_key(raw)
        if not key or key in seen_text:
            continue
        seen_text.add(key)
        out.append(
            {
                "date": str(r["date"].date()),
                "text": raw[:500],
                "textblob": float(r["sentiment_textblob"]),
                "transformer": float(r["sentiment_transformer"]),
                "disagreement": float(r["disagreement"]),
            }
        )
        if len(out) >= top_k:
            break
    return out


def sample_reviews_for_trading_date(
    reviews_scored: pd.DataFrame,
    trading_date_str: str | None,
    *,
    k: int = 3,
    max_chars: int = 240,
) -> list[dict[str, str]]:
    """
    Pull up to ``k`` review excerpts whose normalized calendar date matches the market trading day string.

    Review rows use the same normalized ``date`` as ``prepare_reviews`` (midnight). If no rows match
    the exact day, tries the prior calendar day (reviews sometimes cluster a day before the print).
    """
    if not trading_date_str or reviews_scored is None or reviews_scored.empty or "review_text" not in reviews_scored.columns:
        return []
    try:
        target = pd.to_datetime(trading_date_str).normalize()
    except (TypeError, ValueError):
        return []
    df = reviews_scored.copy()
    df["_d"] = pd.to_datetime(df["date"], errors="coerce").dt.normalize()
    sub = df[df["_d"] == target]
    if sub.empty:
        sub = df[df["_d"] == target - pd.Timedelta(days=1)]
    if sub.empty:
        return []
    out: list[dict[str, str]] = []
    for _, row in sub.head(k).iterrows():
        raw = str(row.get("review_text") or "")
        excerpt = (raw[:max_chars] + "…") if len(raw) > max_chars else raw
        dpart = row["_d"]
        ds = str(dpart.date()) if pd.notna(dpart) else ""
        out.append({"date": ds, "excerpt": excerpt})
    return out


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

    corr_tb, corr_tr = compute_correlations(reg)

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

    stability_df = None
    holdout_tb = None
    holdout_tr = None
    if len(reg) >= 15:
        try:
            stability_df = build_stability_frame(reg, rolling_window=20, expanding_min=15, rolling_min=15)
        except Exception:
            stability_df = None
        holdout_tb = _sanitize_holdout_dict(holdout_ols_hc3(reg, "avg_textblob_lag1", "ret"))
        if reg["avg_transformer_lag1"].notna().any():
            holdout_tr = _sanitize_holdout_dict(holdout_ols_hc3(reg, "avg_transformer_lag1", "ret"))
        else:
            holdout_tr = None

    reg_full_tb = _sanitize_regression_dict(full_sample_ols_hc3(reg, "avg_textblob_lag1", "ret"))
    reg_full_tr: dict[str, Any] | None = None
    if reg["avg_transformer_lag1"].notna().sum() >= 8:
        reg_full_tr = _sanitize_regression_dict(full_sample_ols_hc3(reg, "avg_transformer_lag1", "ret"))

    text_mining_result: dict[str, Any] | None = None
    if len(reviews_scored) >= 5:
        text_mining_result = run_text_mining(
            reviews_scored,
            reg,
            top_disagreement=max(top_disagreement * 5, 60),
        )

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

    brief_quotes = quotes[:5]
    brief = render_template_brief(company_name, stats, brief_quotes)
    polished: str | None = None
    if polish_brief_with_llm:
        polished = maybe_llm_polish_brief(brief, stats, json.dumps(brief_quotes, ensure_ascii=False))

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
        stability_df=stability_df,
        holdout_textblob=holdout_tb,
        holdout_transformer=holdout_tr,
        regression_full_textblob=reg_full_tb,
        regression_full_transformer=reg_full_tr,
        text_mining=text_mining_result,
    )
