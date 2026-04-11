"""
Grounded executive brief: numbers and quotes come from the pipeline, not from free-form LLM guessing.

- Without API key: deterministic Markdown brief (fine for class PoC).
- With OPENAI_API_KEY + pip install openai: optional short LLM rewrite that must only use provided facts.
"""
from __future__ import annotations

import os
from typing import Any

from executive_copy import LABEL_LEXICON_SCORER, LABEL_NEURAL_SCORER, TECH_LEXICON, TECH_NEURAL


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
        "### What ReviewSignal answers",
        "- **Monitoring:** Whether average review tone on one day tends to line up with **next day’s** stock move in your `ret` column.",
        "- **Triage:** Which reviews deserve a human read because the **fast score** and **contextual AI score** disagree.",
        "",
        "### Signals from your upload (no outside data)",
        f"- Reviews analyzed: **{stats['n_reviews']}** ({stats['date_min']} → {stats['date_max']})",
        f"- **{LABEL_LEXICON_SCORER}** vs next-day return (lag-1): **{_fmt_float(stats.get('corr_textblob_lag1'))}**",
    ]
    if stats.get("corr_transformer_lag1") is not None:
        lines.append(
            f"- **{LABEL_NEURAL_SCORER}** vs next-day return (lag-1): **{_fmt_float(stats.get('corr_transformer_lag1'))}**"
        )
    if stats.get("mean_abs_disagreement") is not None:
        lines.append(
            f"- Average gap between the two scorers per review: **{_fmt_float(stats.get('mean_abs_disagreement'))}** "
            "(larger gaps often mean sarcasm, negation, or informal language — read those rows first)."
        )
    if stats.get("worst_return_day"):
        lines.append(
            f"- Worst market day in window: **{stats['worst_return_day']}** (return {_fmt_float(stats.get('worst_return_value'))})."
        )

    lines += [
        "",
        "### For your technical team — scorer IDs & disagreement table",
        f"- **{LABEL_LEXICON_SCORER}** = `{TECH_LEXICON}`; **{LABEL_NEURAL_SCORER}** = `{TECH_NEURAL}`.",
        "",
        "### Reviews to read first (scores disagree)",
        "",
    ]
    if not disagreement_quotes:
        lines.append("_No disagreement rows in this sample._")
    else:
        for i, q in enumerate(disagreement_quotes, 1):
            lines.append(
                f"{i}. **{_fmt_float(q.get('disagreement'), 3)}** apart — {TECH_LEXICON} {_fmt_float(q.get('textblob'), 3)}, "
                f"{TECH_NEURAL} {_fmt_float(q.get('transformer'), 3)} — {q.get('date', '')}"
            )
            lines.append(f"   > {q.get('text', '')}")
            lines.append("")

    lines += [
        "",
        "### Suggested next steps (validate internally)",
        "- **Product / eng:** If worst days align with incidents, map themes in flagged reviews to fixes and comms.",
        "- **IR / leadership:** Use score disagreement as a **quality check** before attributing moves to “app store buzz.”",
        f"- **Risk:** When **{LABEL_NEURAL_SCORER}** is much more negative than **{LABEL_LEXICON_SCORER}**, escalate **manual review**.",
        "",
        "_Association ≠ causation. Monitoring and triage only — not investment advice._",
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

    base = os.environ.get("OPENAI_BASE_URL", "").strip()
    kw: dict[str, str] = {"api_key": key}
    if base:
        kw["base_url"] = base
    client = OpenAI(**kw)
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
