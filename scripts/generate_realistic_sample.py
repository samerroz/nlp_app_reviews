#!/usr/bin/env python3
"""
Generate a larger synthetic reviews + market CSV pair for demos (~90 trading days).
Run from repo root: python scripts/generate_realistic_sample.py
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_REV = ROOT / "sample_data" / "reviews_demo.csv"
OUT_MKT = ROOT / "sample_data" / "market_demo.csv"

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
]


def main() -> None:
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2024-06-03", periods=90, freq="C")
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
        market_rows.append({"date": d.strftime("%Y-%m-%d"), "ticker": TICKER, "ret": round(float(rets[i]), 6)})

    review_rows = []
    rid = 0
    for i, d in enumerate(dates):
        p_pos = float((latent[i] + 1.0) / 2.0)
        p_pos = 0.15 + 0.7 * p_pos
        n_rev = int(rng.integers(7, 19))
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
                    "ticker": TICKER,
                }
            )
            rid += 1

    pd.DataFrame(review_rows).to_csv(OUT_REV, index=False)
    pd.DataFrame(market_rows).to_csv(OUT_MKT, index=False)
    print(f"Wrote {len(review_rows)} reviews, {len(market_rows)} market days -> {OUT_REV.name}, {OUT_MKT.name}")


if __name__ == "__main__":
    main()
