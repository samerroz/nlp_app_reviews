#!/usr/bin/env python3
"""
Generate synthetic reviews + market CSV pair for demos.

Run from repo root:
  python scripts/generate_realistic_sample.py
  python scripts/generate_realistic_sample.py --days 500 --reviews-min 15 --reviews-max 45 \\
    --out-rev .cache/reviews_demo_large.csv --out-mkt .cache/market_demo_large.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_REV = ROOT / "sample_data" / "reviews_demo.csv"
DEFAULT_OUT_MKT = ROOT / "sample_data" / "market_demo.csv"

TICKER = "LLGL"

POS = [
    "Love the new lessons streak keeps me coming back every day.",
    "Best language app I have tried clear explanations and fun challenges.",
    "Amazing update the app feels faster and smoother now.",
    "Really helpful for daily practice on my commute.",
    "Solid progress tracking motivates me to stay consistent.",
    "Five stars the pronunciation drills finally clicked for me.",
    "Great content and the community challenges are fun.",
    "Exceeded my expectations this week after the patch.",
]
NEG = [
    "Crashes on launch cannot open my course anymore.",
    "Terrible billing charged twice and support is slow.",
    "Frustrated with bugs after the last update fix this.",
    "Subscription price jump feels unfair for what we get.",
    "Audio cuts out constantly unusable for listening practice.",
    "Notifications are aggressive and drain my battery.",
    "Lost my streak due to a sync bug very disappointing.",
    "Support took forever to respond still not resolved.",
    "Video lessons freeze halfway through every session lately.",
    "Cannot cancel subscription in the app very shady flow.",
    "Translation hints are wrong for my target language frustrating.",
    "Leaderboard reset after update lost all my progress.",
    "Dark mode is broken text is unreadable at night.",
    "Microphone permission bug means speaking exercises never score.",
]
NEU = [
    "Okay app does the job nothing special.",
    "Mixed feelings some lessons great others feel repetitive.",
    "It works but the UI could be cleaner.",
    "Average experience might keep it might not.",
    "Some good features but onboarding was confusing.",
]
MIXED = [
    "Not bad but could be better if offline mode worked reliably.",
    "Great lessons but the paywall hits at annoying times.",
    "Really helpful for daily practice though ads are heavy.",
    "Fun when it works but login fails on wifi sometimes.",
    "Good vocabulary decks but speaking feedback feels random.",
    "Worth it on sale full price feels steep for what you get.",
]


def generate_synthetic(
    out_rev: Path,
    out_mkt: Path,
    *,
    seed: int = 42,
    days: int = 90,
    reviews_min: int = 7,
    reviews_max: int = 18,
    start_date: str = "2024-06-03",
    ticker: str = TICKER,
) -> tuple[int, int]:
    out_rev = Path(out_rev)
    out_mkt = Path(out_mkt)
    out_rev.parent.mkdir(parents=True, exist_ok=True)
    out_mkt.parent.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start_date, periods=days, freq="C")
    n_days = len(dates)

    latent = np.zeros(n_days)
    for i in range(n_days):
        latent[i] = 0.55 * np.sin(i / 11.0) + 0.25 * np.sin(i / 4.0) + rng.normal(0, 0.18)
    latent = np.clip(latent, -1.0, 1.0)

    rets = np.zeros(n_days)
    for i in range(n_days):
        lag = latent[i - 1] if i > 0 else 0.0
        rets[i] = 0.012 * lag + rng.normal(0, 0.0095)
        if i % 31 == 17:
            rets[i] -= 0.018
        if i % 41 == 9:
            rets[i] += 0.015

    market_rows = []
    for i, d in enumerate(dates):
        market_rows.append({"date": d.strftime("%Y-%m-%d"), "ticker": ticker, "ret": round(float(rets[i]), 6)})

    lo, hi = min(reviews_min, reviews_max), max(reviews_min, reviews_max)
    review_rows = []
    for i, d in enumerate(dates):
        p_pos = float((latent[i] + 1.0) / 2.0)
        p_pos = 0.15 + 0.7 * p_pos
        n_rev = int(rng.integers(lo, hi + 1))
        for _ in range(n_rev):
            u = rng.random()
            if u < p_pos * 0.55:
                text = rng.choice(POS)
                rating = int(rng.choice([4, 5, 5, 5]))
            elif u < p_pos * 0.55 + (1 - p_pos) * 0.45:
                text = rng.choice(NEG)
                rating = int(rng.choice([1, 2, 2, 3]))
            elif u < p_pos * 0.55 + (1 - p_pos) * 0.45 + 0.12:
                text = rng.choice(MIXED)
                rating = int(rng.choice([3, 4]))
            else:
                text = rng.choice(NEU)
                rating = int(rng.choice([3, 3, 4]))
            hour, minute = int(rng.integers(8, 21)), int(rng.integers(0, 60))
            ts = f"{d.strftime('%Y-%m-%d')} {hour:02d}:{minute:02d}:00"
            review_rows.append(
                {
                    "date": ts,
                    "rating": rating,
                    "review_text": text,
                    "ticker": ticker,
                }
            )

    pd.DataFrame(review_rows).to_csv(out_rev, index=False)
    pd.DataFrame(market_rows).to_csv(out_mkt, index=False)
    return len(review_rows), len(market_rows)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate synthetic LinguaLoop-style demo CSVs.")
    p.add_argument("--days", type=int, default=90, help="Trading days (business-day calendar)")
    p.add_argument("--reviews-min", type=int, default=7)
    p.add_argument("--reviews-max", type=int, default=18)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--start", type=str, default="2024-06-03", help="First business date YYYY-MM-DD")
    p.add_argument("--ticker", type=str, default=TICKER)
    p.add_argument("--out-rev", type=Path, default=DEFAULT_OUT_REV)
    p.add_argument("--out-mkt", type=Path, default=DEFAULT_OUT_MKT)
    args = p.parse_args()

    n_r, n_m = generate_synthetic(
        args.out_rev,
        args.out_mkt,
        seed=args.seed,
        days=args.days,
        reviews_min=args.reviews_min,
        reviews_max=args.reviews_max,
        start_date=args.start,
        ticker=args.ticker,
    )
    print(f"Wrote {n_r} reviews, {n_m} market days -> {args.out_rev}, {args.out_mkt}")


if __name__ == "__main__":
    main()
