"""
Grounded executive brief: numbers and quotes come from the pipeline, not from free-form LLM guessing.

- Without API key: deterministic Markdown brief (fine for class PoC).
- With OPENAI_API_KEY + pip install openai: optional short LLM rewrite that must only use provided facts.
"""
from __future__ import annotations

import os
from typing import Any


def _fmt_float(x: float | None, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and x != x):  # nan
        return "n/a"
    return f"{x:.{nd}f}"


def build_stats_payload(
    *,
    n_reviews: int,
    date_min: str,
    date_max: str,
    corr_textblob_lag1: float | None,
    corr_transformer_lag1: float | None,
    mean_abs_disagreement: float | None,
    worst_return_day: str | None,
    worst_return_value: float | None,
) -> dict[str, Any]:
    return {
        "n_reviews": n_reviews,
        "date_min": date_min,
        "date_max": date_max,
        "corr_textblob_lag1": corr_textblob_lag1,
        "corr_transformer_lag1": corr_transformer_lag1,
        "mean_abs_disagreement": mean_abs_disagreement,
        "worst_return_day": worst_return_day,
        "worst_return_value": worst_return_value,
    }


def render_template_brief(
    company_name: str,
    stats: dict[str, Any],
    disagreement_quotes: list[dict[str, Any]],
) -> str:
    """Markdown suitable for slides or Streamlit."""
    lines = [
        f"## Executive brief — {company_name}",
        "",
        "### Quantified signals (from your data only)",
        f"- Reviews analyzed: **{stats['n_reviews']}** ({stats['date_min']} → {stats['date_max']})",
        f"- Lag-1 correlation (daily avg TextBlob vs next-day return): **{_fmt_float(stats.get('corr_textblob_lag1'))}**",
    ]
    if stats.get("corr_transformer_lag1") is not None:
        lines.append(
            f"- Lag-1 correlation (daily avg RoBERTa sentiment vs next-day return): **{_fmt_float(stats.get('corr_transformer_lag1'))}**"
        )
    if stats.get("mean_abs_disagreement") is not None:
        lines.append(
            f"- Mean |TextBlob − RoBERTa| per review: **{_fmt_float(stats.get('mean_abs_disagreement'))}** "
            "(high = informal/sarcastic/negated language — prioritize human read of flagged quotes)."
        )
    if stats.get("worst_return_day"):
        lines.append(
            f"- Worst market day in window: **{stats['worst_return_day']}** (return {_fmt_float(stats.get('worst_return_value'))})."
        )

    lines += ["", "### Reviews to read first (lexicon vs transformer disagree)", ""]
    if not disagreement_quotes:
        lines.append("_No disagreement rows in this tiny sample._")
    else:
        for i, q in enumerate(disagreement_quotes, 1):
            lines.append(f"{i}. **{_fmt_float(q.get('disagreement'), 3)}** apart — TextBlob {_fmt_float(q.get('textblob'), 3)}, "
                         f"RoBERTa {_fmt_float(q.get('transformer'), 3)} — {q.get('date', '')}")
            lines.append(f"   > {q.get('text', '')}")
            lines.append("")

    lines += [
        "",
        "### Suggested actions (hypothesis — validate with your teams)",
        "- **Product / eng:** If worst days align with release incidents, tie hotfix prioritization to themes in flagged reviews.",
        "- **IR / leadership:** Use the dual sentiment spread as a **quality check** before attributing moves to “social buzz.”",
        "- **Risk:** When RoBERTa is much more negative than TextBlob, escalate **manual review** (common for sarcasm and negation).",
        "",
        "_Association ≠ causation. Use this as monitoring + triage, not a trading mandate._",
    ]
    return "\n".join(lines)


def maybe_llm_polish_brief(
    template_markdown: str,
    stats: dict[str, Any],
    quotes_json: str,
) -> str | None:
    """
    If OPENAI_API_KEY is set, ask the model to tighten wording without adding facts.
    Returns None if skipped or on failure.
    """
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None

    client = OpenAI()
    system = (
        "You write executive briefings. Rules: "
        "Do not invent statistics, dates, correlations, or quotes. "
        "Only paraphrase the facts provided in the user message. "
        "Keep under 200 words. Output Markdown."
    )
    user = (
        "Facts JSON:\n"
        + str(stats)
        + "\n\nQuoted review snippets (verbatim allowed):\n"
        + quotes_json
        + "\n\nDraft to polish (keep all numbers identical):\n"
        + template_markdown
    )
    try:
        r = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.2,
        )
        return (r.choices[0].message.content or "").strip()
    except Exception:
        return None
