"""
PoC CLI: sample CSVs → full pipeline → print brief.

Run from repo root:
  python src/demo_pipeline.py

Transformer: pip install -r requirements-ml.txt
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from pipeline import run_pipeline

ROOT = Path(__file__).resolve().parents[1]
SAMPLE = ROOT / "sample_data"


def main() -> None:
    reviews = pd.read_csv(SAMPLE / "reviews_sample.csv")
    market = pd.read_csv(SAMPLE / "market_sample.csv")

    try:
        result = run_pipeline(
            reviews,
            market,
            company_name="LinguaLoop Ltd.",
            run_transformer=True,
            top_disagreement=10,
        )
    except ValueError as e:
        print(e)
        return

    if result.transformer_error:
        print(f"[RoBERTa] Skipped or partial: {result.transformer_error}\n")

    print("Merged sample (lagged sentiment vs same-day return):\n", result.reg.to_string(index=False))

    if result.corr_textblob_lag1 is not None:
        print(f"\nPearson (lag-1 TextBlob vs return): {result.corr_textblob_lag1:.4f}")
    if result.corr_transformer_lag1 is not None:
        print(f"Pearson (lag-1 RoBERTa vs return): {result.corr_transformer_lag1:.4f}")

    print("\n--- Grounded brief (template) ---\n")
    print(result.brief_markdown)

    if result.brief_polished:
        print("\n--- Optional LLM polish (OPENAI_API_KEY) ---\n")
        print(result.brief_polished)

    print(
        "\nNote: Tiny sample = illustrative only. Full study: longer window + robust regression — see README and docs/CASE_STUDY.md."
    )


if __name__ == "__main__":
    main()
