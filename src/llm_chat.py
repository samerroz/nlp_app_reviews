"""
Grounded chat: system prompt embeds frozen pipeline JSON; model must not invent statistics.

- **No API key:** rule-based answers use only the JSON snapshot (no hallucinated numbers).
- **Any OpenAI-compatible endpoint:** set `OPENAI_API_KEY` and optionally `OPENAI_BASE_URL`
  (Groq, Together, LM Studio, Azure OpenAI-style gateways, etc.) plus `OPENAI_MODEL`.
"""
from __future__ import annotations

import json
import math
import os
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline import PipelineResult


def llm_client_configured() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY", "").strip())


def _openai_client():
    from openai import OpenAI

    key = os.environ.get("OPENAI_API_KEY", "").strip()
    base = os.environ.get("OPENAI_BASE_URL", "").strip()
    kw: dict[str, str] = {"api_key": key}
    if base:
        kw["base_url"] = base
    return OpenAI(**kw)


def build_chat_context(company_name: str, result: "PipelineResult") -> str:
    tm = result.text_mining
    payload: dict[str, Any] = {
        "company": company_name,
        "n_reviews": result.stats.get("n_reviews"),
        "overlap_trading_days": result.overlap_days,
        "date_overlap": [result.date_overlap_min, result.date_overlap_max],
        "corr_textblob_lag1": result.corr_textblob_lag1,
        "corr_transformer_lag1": result.corr_transformer_lag1,
        "mean_abs_disagreement": result.mean_abs_disagreement,
        "worst_return_day": result.worst_return_day,
        "worst_return_value": result.worst_return_value,
        "regression_full_textblob": result.regression_full_textblob,
        "regression_full_transformer": result.regression_full_transformer,
        "holdout_textblob": result.holdout_textblob,
        "holdout_transformer": result.holdout_transformer,
        "text_mining_bigrams": (tm or {}).get("bigrams"),
        "text_mining_topics": (tm or {}).get("nmf_topics"),
        "text_mining_n_docs": (tm or {}).get("n_docs"),
        "text_mining_source": (tm or {}).get("source_description"),
        "disagreement_quotes": result.disagreement_quotes[:12],
    }
    return json.dumps(payload, indent=2, default=str)


CHAT_SYSTEM = (
    "You are an analyst assistant for ReviewSignal. "
    "You ONLY use facts from the JSON block the user provides. "
    "Do not invent correlations, p-values, dates, or review text. "
    "If asked for something not in the JSON, say you do not have that in the run. "
    "Cite numbers exactly as given. Keep answers concise (under 180 words unless asked for detail). "
    "Format: a one-line title or **bold** lead, then 2-4 short bullets or paragraphs, then one line on limits "
    "(e.g. association is not causation) when relevant."
)


def chat_offline_reply(context_json: str, user_message: str) -> str:
    """
    Deterministic, grounded replies when no LLM is configured.
    Supports compound questions (multiple intents in one message).
    """
    try:
        ctx = json.loads(context_json)
    except json.JSONDecodeError:
        return "I only have this run's saved JSON snapshot, and it could not be read. Re-run **Run analysis**."

    msg_l = user_message.lower()
    co = ctx.get("company") or "this company"
    n_rev = ctx.get("n_reviews")
    days = ctx.get("overlap_trading_days")
    d0, d1 = ctx.get("date_overlap") or (None, None)
    c_tb = ctx.get("corr_textblob_lag1")
    c_tr = ctx.get("corr_transformer_lag1")
    dis = ctx.get("mean_abs_disagreement")
    worst_d = ctx.get("worst_return_day")
    worst_v = ctx.get("worst_return_value")
    quotes = ctx.get("disagreement_quotes") or []

    def _fmt_corr(x: Any) -> str:
        if x is None:
            return "n/a"
        try:
            xf = float(x)
        except (TypeError, ValueError):
            return "n/a"
        if not math.isfinite(xf):
            return "n/a"
        return f"{xf:.4f}"

    def _is_positive_corr(s: str) -> bool:
        try:
            return s != "n/a" and math.isfinite(float(s)) and float(s) > 0
        except (TypeError, ValueError):
            return False

    # ── Intent keyword sets ───────────────────────────────────────────────────
    # Each set uses substrings checked via `in msg_l`, so partial matches work.
    _TRANSFORMER_KWS = [
        "contextual ai", "contextual", "ai scorer", "ai score", "neural",
        "roberta", "transformer", "textblob", "lexicon", "disagree",
        "disagreement", "second opinion", "dual scor", "two model",
        "why zero", "near 0",
    ]
    _STABILITY_KWS = [
        "stabilize", "stabilise", "did the pattern", "pattern stab",
        "stable", "stability", "rolling", "expanding", "consistent",
        "drift", "over time", "change over", "has the link", "shift",
    ]
    _HOLDOUT_KWS = [
        "holdout", "pseudo", "forecast", "predict", "rmse",
        "out-of-sample", "sanity check", "recent week", "last week",
    ]
    _MARKET_KWS = [
        "market", "stock", "return", "correlation", "pearson", "ols",
        "slope", "line up", "mood", "association", "link",
    ]
    _THEME_KWS = [
        "bigram", "topic", "nmf", "theme", "word", "phrase",
        "complaint", "issue", "customer say", "customers say",
    ]
    _WORST_KWS = [
        "worst day", "worst return", "bad day", "august 14", "aug 14",
        "what happened", "what reviews",
    ]

    def _matches(kw_list: list) -> bool:
        return any(w in msg_l for w in kw_list)

    # ── Per-intent answer builders ────────────────────────────────────────────

    def _answer_transformer() -> str:
        parts = [
            f"**What is the Contextual AI scorer? ({co})**",
            "",
            "- **TextBlob** (the dictionary scorer) goes word by word. Short or polite phrasing often "
            "scores near 0 even when the review is clearly negative -- it misses negation, sarcasm, and context.",
            "- **RoBERTa** (the Contextual AI scorer) reads the whole sentence at once. It was trained on "
            "short informal social text, so it catches phrases like 'not resolved' or 'five stars but...' "
            "that TextBlob gets wrong.",
            "- When the two scores strongly disagree on the same review, that review is flagged -- "
            "a human should read it because neither model is clearly right.",
        ]
        if dis is not None:
            try:
                dis_f = float(dis)
                verdict = (
                    "high -- check the disagreement queue first."
                    if dis_f > 0.4
                    else "moderate -- the two models are broadly aligned."
                )
                parts.append(
                    f"- In this run the average gap between the two scorers is **{dis_f:.4f}** -- {verdict}"
                )
            except (TypeError, ValueError):
                pass
        if c_tr is not None:
            try:
                ctrf = float(c_tr)
                if math.isfinite(ctrf):
                    parts.append(
                        f"- RoBERTa's lag-1 link to next-day returns in this run: **{ctrf:.4f}** "
                        f"(vs TextBlob's **{_fmt_corr(c_tb)}**). The higher value is the stronger signal."
                    )
            except (TypeError, ValueError):
                pass
        if quotes:
            ex = quotes[0]
            t = str(ex.get("text", ""))[:160]
            ellipsis = "..." if len(str(ex.get("text", ""))) > 160 else ""
            parts.append(f'- Top disagreement example: "{t}{ellipsis}"')
        parts.append("\n_Association only -- this does not prove reviews caused stock moves._")
        return "\n".join(parts)

    def _answer_stability() -> str:
        lines = [
            f"**Did the sentiment-return pattern stay stable? ({co})**",
            "",
            "- The stability chart shows two lines over time: an **expanding** correlation (all data "
            "up to that point) and a **rolling 20-day** correlation (just the recent window).",
            "- When the rolling line tracks the expanding line closely, the pattern is **consistent**. "
            "When the rolling line swings away or crosses zero, the link broke down in that stretch.",
            "- A rising expanding line means the signal got stronger as more data came in. "
            "Flat or falling means it did not compound.",
        ]
        h_tb = ctx.get("holdout_textblob")
        if isinstance(h_tb, dict):
            ntr = h_tb.get("n_train")
            nho = h_tb.get("n_holdout")
            cpa = h_tb.get("corr_pred_vs_actual_holdout")
            end = h_tb.get("holdout_last_date")
            cpa_s = _fmt_corr(cpa)
            verdict = (
                "the pattern held on fresh data."
                if _is_positive_corr(cpa_s)
                else "the simple rule did not clearly hold on the most recent days."
            )
            lines.append(
                f"- **Sanity check:** trained on the first **{ntr}** days, tested on the last "
                f"**{nho}** (through **{end}**). Predicted-vs-actual correlation: **{cpa_s}** -- {verdict}"
            )
        lines.append(
            "\n_One holdout window is not full validation -- extend history and re-run for stronger evidence._"
        )
        return "\n".join(lines)

    def _answer_holdout() -> str:
        lines = [f"**What the holdout / sanity check numbers mean ({co})**", ""]
        found = False
        for label, key in (("TextBlob", "holdout_textblob"), ("RoBERTa", "holdout_transformer")):
            h = ctx.get(key)
            if not isinstance(h, dict):
                continue
            found = True
            ntr = h.get("n_train")
            nho = h.get("n_holdout")
            cpa = h.get("corr_pred_vs_actual_holdout")
            rmse = h.get("rmse_holdout")
            end = h.get("holdout_last_date")
            cpa_s = _fmt_corr(cpa)
            try:
                rmse_s = f"{float(rmse):.6f}" if rmse is not None else "n/a"
            except (TypeError, ValueError):
                rmse_s = "n/a"
            lines.append(
                f"- **{label}:** trained on the first **{ntr}** days, tested on the last **{nho}** "
                f"(through **{end}**). Predicted-vs-actual correlation: **{cpa_s}**; RMSE: **{rmse_s}**. "
                "Weak or negative = the simple rule did not hold on fresh data."
            )
        if not found:
            lines.append(
                "- Holdout stats not available for this run "
                "(need more history or stable variation in lagged sentiment)."
            )
        lines.append("\n_Limits: one window is not validation -- extend history and re-run._")
        return "\n".join(lines).strip()

    def _answer_market() -> str:
        lines = [
            f"**How sentiment links to next-day returns ({co})**",
            "",
            f"- **Lag-1 correlation:** Yesterday's avg TextBlob sentiment vs today's return = **{_fmt_corr(c_tb)}**.",
        ]
        if c_tr is not None:
            try:
                if math.isfinite(float(c_tr)):
                    lines.append(f"- RoBERTa lag-1 correlation: **{_fmt_corr(c_tr)}**.")
            except (TypeError, ValueError):
                pass
        lines.append(
            "- This is **association** over the overlapping window -- "
            "news, macro, and sector moves also drive returns."
        )
        if worst_d is not None and worst_v is not None:
            try:
                lines.append(
                    f"- **Worst return day:** **{worst_d}** at **{float(worst_v):.4f}** -- "
                    "worth checking what reviews said the day before."
                )
            except (TypeError, ValueError):
                pass
        return "\n".join(lines)

    def _answer_worst_day() -> str:
        if worst_d is None:
            return "No worst-day data is available in this run's snapshot."
        lines = [f"**Worst trading day in this run ({co})**", ""]
        try:
            lines.append(f"- Date: **{worst_d}**, return: **{float(worst_v):.4f}**.")
        except (TypeError, ValueError):
            lines.append(f"- Date: **{worst_d}**.")
        lines.append(
            "- Check the Overview tab for raw review excerpts from that day -- "
            "the 'Worst Day' panel shows what customers were actually writing."
        )
        lines.append(
            "- The day before (**lag-1**) is what the model uses as the sentiment signal for that return."
        )
        return "\n".join(lines)

    def _answer_themes() -> str:
        tm_data = ctx.get("text_mining_bigrams")
        lines = [f"**What topics show up most in flagged reviews ({co})**", ""]
        if isinstance(tm_data, list) and tm_data:
            top = ", ".join(
                f"**{b.get('phrase', '')}** ({b.get('count', '')})"
                for b in tm_data[:6]
                if isinstance(b, dict)
            )
            lines.append(f"- Top phrases: {top}.")
            lines.append(
                "- These come from reviews on high-disagreement days and extreme-return days only, "
                "not all reviews."
            )
            lines.append(
                "- Use them to label support themes or flag for product -- "
                "not automatic prioritization."
            )
        else:
            lines.append(
                "- No bigram data in this run's snapshot "
                "(RoBERTa may have been off, reducing the flagged corpus)."
            )
        return "\n".join(lines)

    # ── Detect intents and combine answers ────────────────────────────────────
    answers: list[str] = []

    if _matches(_TRANSFORMER_KWS):
        answers.append(_answer_transformer())

    if _matches(_STABILITY_KWS):
        answers.append(_answer_stability())
    elif _matches(_HOLDOUT_KWS):
        # Only add standalone holdout answer if stability didn't already cover it
        answers.append(_answer_holdout())

    if _matches(_WORST_KWS):
        answers.append(_answer_worst_day())

    if _matches(_THEME_KWS):
        answers.append(_answer_themes())

    # Only add market answer if nothing more specific already answered
    if not answers and _matches(_MARKET_KWS):
        answers.append(_answer_market())

    if answers:
        return "\n\n---\n\n".join(answers)

    # ── Default: short factual recap with better prompt hints ─────────────────
    recap = [
        f"**Snapshot for this run ({co})**",
        "",
        f"- **{n_rev}** reviews; **{days}** overlapping trading days ({d0} to {d1}).",
        f"- Lag-1 correlation: TextBlob **{_fmt_corr(c_tb)}**, RoBERTa **{_fmt_corr(c_tr)}**.",
    ]
    if worst_d is not None:
        recap.append(f"- Worst return day: **{worst_d}**.")
    if dis is not None:
        try:
            recap.append(f"- Mean scorer gap: **{float(dis):.4f}**.")
        except (TypeError, ValueError):
            pass
    recap.append(
        "\n_Try asking: \"explain the contextual AI scorer\", \"did the pattern stabilize?\", "
        "\"what happened on the worst day?\", or \"what themes came up?\"_"
    )
    return "\n".join(recap)


def chat_complete(context_json: str, user_message: str) -> str:
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not key:
        return chat_offline_reply(context_json, user_message)
    try:
        from openai import OpenAI  # noqa: F401
    except ImportError:
        return chat_offline_reply(context_json, user_message)

    client = _openai_client()
    user = f"Context (authoritative, do not contradict):\n```json\n{context_json}\n```\n\nQuestion:\n{user_message}"
    try:
        r = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": CHAT_SYSTEM},
                {"role": "user", "content": user},
            ],
            temperature=0.3,
        )
        return (r.choices[0].message.content or "").strip()
    except Exception as e:
        return (
            f"**Could not reach the API ({type(e).__name__}).** "
            "Here is the grounded answer from this run's data:\n\n"
            + chat_offline_reply(context_json, user_message)
        )
