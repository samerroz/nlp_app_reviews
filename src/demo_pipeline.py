"""
PoC: load sample CSVs → TextBlob + optional RoBERTa sentiment → daily merge with returns
→ lag-1 correlations → grounded executive brief (template; optional OpenAI polish).

Run from repo root:
  python src/demo_pipeline.py

Transformer layer (recommended for course NLP depth):
  pip install -r requirements-ml.txt
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from llm_brief import build_stats_payload, maybe_llm_polish_brief, render_template_brief
from sentiment_models import polarity_textblob, try_transformer_scores

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "sample_data"


def main() -> None:
    reviews = pd.read_csv(SAMPLE / "reviews_sample.csv")
    reviews["date"] = pd.to_datetime(reviews["date"]).dt.normalize()
    texts = reviews["review_text"].astype(str).tolist()

    reviews["sentiment_textblob"] = [polarity_textblob(t) for t in texts]
    tr_scores, tr_err = try_transformer_scores(texts)
    if tr_scores is None:
        print(f"[RoBERTa] Skipped ({tr_err or 'unknown error'}). Install: pip install -r requirements-ml.txt\n")
        reviews["sentiment_transformer"] = np.nan
    else:
        reviews["sentiment_transformer"] = tr_scores

    daily = reviews.groupby(["date", "ticker"], as_index=False).agg(
        avg_textblob=("sentiment_textblob", "mean"),
        avg_transformer=("sentiment_transformer", "mean"),
        review_count=("review_text", "count"),
    )

    market = pd.read_csv(SAMPLE / "market_sample.csv")
    market.columns = [c.lower() for c in market.columns]
    market["date"] = pd.to_datetime(market["date"]).dt.normalize()

    merged = daily.merge(market, on=["date", "ticker"], how="inner")
    if merged.empty:
        print("No overlapping dates/tickers after merge — check sample files.")
        return

    for col in ("avg_textblob", "avg_transformer"):
        merged[f"{col}_lag1"] = merged.groupby("ticker")[col].shift(1)

    reg = merged.dropna(subset=["avg_textblob_lag1"])
    print("Merged sample (lagged sentiment vs same-day return):\n", reg.to_string(index=False))

    corr_tb: float | None = None
    corr_tr: float | None = None
    if len(reg) >= 3:
        corr_tb = float(reg["avg_textblob_lag1"].corr(reg["ret"]))
        print(f"\nPearson (lag-1 TextBlob vs return): {corr_tb:.4f}")
    if len(reg.dropna(subset=["avg_transformer_lag1"])) >= 3:
        sub = reg.dropna(subset=["avg_transformer_lag1"])
        corr_tr = float(sub["avg_transformer_lag1"].corr(sub["ret"]))
        print(f"Pearson (lag-1 RoBERTa vs return): {corr_tr:.4f}")

    mean_dis: float | None = None
    if reviews["sentiment_transformer"].notna().any():
        mean_dis = float(
            (reviews["sentiment_textblob"] - reviews["sentiment_transformer"])
            .abs()
            .mean()
        )

    worst_idx = reg["ret"].idxmin() if len(reg) else None
    worst_day = str(reg.loc[worst_idx, "date"].date()) if worst_idx is not None else None
    worst_ret = float(reg.loc[worst_idx, "ret"]) if worst_idx is not None else None

    # Flag reviews where lexicon and transformer disagree most (feeds LLM / manual triage).
    rev = reviews.copy()
    if rev["sentiment_transformer"].notna().any():
        rev["disagreement"] = (rev["sentiment_textblob"] - rev["sentiment_transformer"]).abs()
        top = rev.sort_values("disagreement", ascending=False).head(3)
        quotes = [
            {
                "date": str(r["date"].date()),
                "text": str(r["review_text"])[:280],
                "textblob": float(r["sentiment_textblob"]),
                "transformer": float(r["sentiment_transformer"]),
                "disagreement": float(r["disagreement"]),
            }
            for _, r in top.iterrows()
        ]
    else:
        quotes = []

    stats = build_stats_payload(
        n_reviews=len(reviews),
        date_min=str(reviews["date"].min().date()),
        date_max=str(reviews["date"].max().date()),
        corr_textblob_lag1=corr_tb,
        corr_transformer_lag1=corr_tr,
        mean_abs_disagreement=mean_dis,
        worst_return_day=worst_day,
        worst_return_value=worst_ret,
    )

    company = "LinguaLoop Ltd."
    template = render_template_brief(company, stats, quotes)
    print("\n--- Grounded brief (template) ---\n")
    print(template)

    polished = maybe_llm_polish_brief(template, stats, json.dumps(quotes, ensure_ascii=False))
    if polished:
        print("\n--- Optional LLM polish (same facts; set OPENAI_API_KEY) ---\n")
        print(polished)

    print(
        "\nNote: Tiny sample = illustrative only. Full study: longer window + robust regression — see README and docs/CASE_STUDY.md."
    )


if __name__ == "__main__":
    main()
