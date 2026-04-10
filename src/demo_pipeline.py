"""
Minimal PoC: load tiny CSV samples, score review text, aggregate daily sentiment,
merge with daily returns, print correlation.

Run from repo root:
  python src/demo_pipeline.py
"""
from pathlib import Path

import pandas as pd
from textblob import TextBlob

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "sample_data"


def polarity(text: str) -> float:
    return float(TextBlob(str(text)).sentiment.polarity)


def main() -> None:
    reviews = pd.read_csv(SAMPLE / "reviews_sample.csv")
    reviews["date"] = pd.to_datetime(reviews["date"]).dt.normalize()
    reviews["sentiment"] = reviews["review_text"].map(polarity)
    daily = (
        reviews.groupby(["date", "ticker"], as_index=False)
        .agg(avg_sentiment=("sentiment", "mean"), review_count=("review_text", "count"))
    )

    market = pd.read_csv(SAMPLE / "market_sample.csv")
    market.columns = [c.lower() for c in market.columns]
    market["date"] = pd.to_datetime(market["date"]).dt.normalize()

    merged = daily.merge(market, on=["date", "ticker"], how="inner")
    if merged.empty:
        print("No overlapping dates/tickers after merge — check sample files.")
        return

    merged["avg_sentiment_lag1"] = merged.groupby("ticker")["avg_sentiment"].shift(1)
    reg = merged.dropna(subset=["avg_sentiment_lag1"])

    print("Merged sample (lagged sentiment vs same-day return):\n", reg.to_string(index=False))
    if len(reg) >= 3:
        corr = reg["avg_sentiment_lag1"].corr(reg["ret"])
        print(f"\nPearson correlation (lag-1 daily sentiment vs return): {corr:.4f}")
    print(
        "\nNote: With ~6 days this is illustrative only. "
        "Full analysis uses longer history and robust regression (see README)."
    )


if __name__ == "__main__":
    main()
