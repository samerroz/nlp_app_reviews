"""
Rule-based actionable insight lines from pipeline output (no extra ML).

- ``plain_language_insights`` — 4 signal cards for non-technical readers.
- ``actionable_insights`` — denser technical notes (regression, HC3, NMF) for analysts.
"""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import pandas as pd

from executive_copy import LABEL_LEXICON_SCORER, LABEL_NEURAL_SCORER

if TYPE_CHECKING:
    from pipeline import PipelineResult


def _finite_f(x: Any) -> bool:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v)


def _p_value_label(pv: Any) -> str:
    if not _finite_f(pv):
        return "**p-value undefined** (no variation in lag-1 sentiment on the train window — check CSV columns / duplicates)"
    pvf = float(pv)
    sig = "not significant at 5%" if pvf >= 0.05 else "significant at 5%"
    return f"**{pvf:.4f}** ({sig})"


def _ols_block_valid(d: dict[str, Any]) -> bool:
    return all(_finite_f(d.get(k)) for k in ("coef", "ci_low", "ci_high", "p_value", "r2"))


def _ret_as_pct_str(x: float) -> str:
    """Format market `ret` as percentage (assumes decimal, e.g. 0.012 = 1.2%)."""
    return f"{float(x) * 100:.3f}%"


def _corr_strength_label(r: float) -> str:
    a = abs(float(r))
    if a < 0.12:
        return "very weak"
    if a < 0.28:
        return "weak"
    if a < 0.45:
        return "moderate"
    return "noticeable"


def _holdout_plain_language(h: dict[str, Any], scorer_display: str) -> str:
    """One sentence a non-quant can use; numbers only from dict."""
    ntr = h.get("n_train")
    nho = h.get("n_holdout")
    end = h.get("holdout_last_date") or "the end of the window"
    cpa = h.get("corr_pred_vs_actual_holdout")
    rmse = h.get("rmse_holdout")
    rmse_s = ""
    try:
        if rmse is not None and _finite_f(rmse):
            rmse_s = f" Typical prediction error was about **{_ret_as_pct_str(float(rmse))}** per day (same units as your **ret** column)."
    except (TypeError, ValueError):
        rmse_s = ""

    if cpa is None or not _finite_f(cpa):
        tail = " We could not score alignment in the holdout (often too few days or flat predictions)."
    else:
        cf = float(cpa)
        if cf < -0.05:
            tail = f" Out-of-sample, predicted and actual returns **disagreed** (correlation **{cf:.2f}**) — treat the pattern as **unstable**, not actionable."
        elif abs(cf) < 0.15:
            tail = f" Out-of-sample alignment was **almost flat** (correlation **{cf:.2f}**) — the simple rule did **not** reliably track returns on the last days.{rmse_s}"
        elif cf < 0.35:
            tail = f" Out-of-sample, predictions and returns lined up **weakly** (correlation **{cf:.2f}**). Interesting for research; not a mandate.{rmse_s}"
        else:
            tail = f" Out-of-sample alignment was **positive** (correlation **{cf:.2f}**), but one window is not validation — extend the history.{rmse_s}"

    return (
        f"**Recent-week sanity check ({scorer_display}):** We fit a simple rule on the **first {ntr}** trading days, "
        f"then checked it on the **last {nho}** days through **{end}**. {tail}"
    )


def plain_language_insights(
    result: "PipelineResult",
    *,
    company: str = "Your company",
    insight_facts: dict[str, Any] | None = None,
) -> list[str]:
    """
    4 opinionated signal cards for non-technical readers (IR, VP Product, leadership).
    Each card is one sentence that passes the standup test: could you say it out loud?
    """
    co = company.strip() or "Your company"
    ifacts = insight_facts or {}
    lines: list[str] = []

    # ── Card 1: Tone trend ────────────────────────────────────────────────────
    traj = ifacts.get("trajectory") or {}
    direction = traj.get("lexicon_direction", "flat")
    n_days = int(traj.get("n_days") or result.overlap_days or 0)
    volatility = traj.get("volatility_note", "unknown")
    direction_phrase = {
        "rose": "became more positive",
        "fell": "became more negative",
        "flat": "stayed roughly flat",
    }.get(direction, "held flat")
    vol_phrase = {
        "high": " Day-to-day swings were significant — don't over-index on any single day.",
        "moderate": " Day-to-day variation was moderate.",
        "low": " The trend was relatively smooth.",
    }.get(volatility, "")
    lines.append(
        f"**Customer sentiment {direction_phrase}** across the {n_days}-day window.{vol_phrase}"
    )

    # ── Card 2: Association with next-day stock moves ─────────────────────────
    tb = result.corr_textblob_lag1
    if tb is not None and _finite_f(tb):
        strength = _corr_strength_label(float(tb))
        dir_word = "better" if float(tb) > 0 else "worse"
        lines.append(
            f"**When reviews were warmer, the next trading day tended to be {dir_word}** — "
            f"a {strength} link. This is an observation, not a prediction — "
            "news, earnings, and macro events also move stock prices."
        )
    else:
        lines.append(
            "**No clear link** was found between review tone and next-day stock moves in this window. "
            "Sentiment may still matter for product and support decisions."
        )

    # ── Card 3: Reliability — stability + holdout combined ───────────────────
    stab = ifacts.get("stability") or {}
    hl = (ifacts.get("holdout") or {}).get("lexicon") or {}
    if stab.get("regime") == "recent_noisy":
        stab_sentence = "The pattern **shifted in recent weeks** — the full-window average may hide a recent change."
    elif stab.get("regime") == "stable":
        stab_sentence = "The pattern was **consistent** across the full window — no obvious regime change."
    else:
        stab_sentence = "The pattern was **mixed** — neither clearly stable nor clearly broken."
    if hl.get("available"):
        v = hl.get("verdict", "inconclusive")
        holdout_clause = {
            "confirms": " The most recent days **confirmed** the pattern — encouraging, though still just one window.",
            "weakens": " The most recent days **weakened** the pattern — down-weight the headline number when briefing leadership.",
            "partly_confirms": " The most recent days **partially confirmed** the pattern.",
        }.get(v, " The most recent days were **inconclusive** — not enough signal either way.")
    else:
        holdout_clause = ""
    lines.append(f"**Reliability:** {stab_sentence}{holdout_clause}")

    # ── Card 4: Single recommended action ─────────────────────────────────────
    fr = ifacts.get("focus_recommendation")
    if not fr:
        fr = (
            f"Walk through the worst-day reviews in **Reviews** with your product team, "
            f"then re-export {co} data on your usual cadence."
        )
    lines.append(fr)

    return lines


def actionable_insights(result: "PipelineResult") -> list[str]:
    """Short bullets for technical expanders; conservative wording."""
    lines: list[str] = []
    n = result.stats.get("n_reviews", 0)
    days = result.overlap_days
    tb = result.corr_textblob_lag1
    tr = result.corr_transformer_lag1
    dis = result.mean_abs_disagreement

    if days < 20:
        lines.append(
            f"**Sample depth:** Only **{days}** overlapping market days — treat correlations as **exploratory**. "
            "Aim for **60+** trading days in production runs."
        )
    elif days < 60:
        lines.append(
            f"**Sample depth:** **{days}** overlapping days is acceptable for a **pilot**; extend to a full quarter or year for stable monitoring."
        )
    else:
        lines.append(
            f"**Sample depth:** **{days}** overlapping trading days — enough for a **credible pilot** trend read (still not proof of causation)."
        )

    lines.append(f"**Volume:** **{n:,}** reviews in window — ensure this matches your internal export coverage (store, region, platform).")

    rf = result.regression_full_textblob
    if rf and _ols_block_valid(rf):
        lines.append(
            f"**Full-sample OLS (TextBlob, HC3):** slope **{rf['coef']:.6f}** "
            f"95% CI **[{rf['ci_low']:.6f}, {rf['ci_high']:.6f}]**, p **{rf['p_value']:.4f}**, R² **{rf['r2']:.4f}** "
            f"(*n*={rf['n']} days). Association ≠ causation."
        )
    elif len(result.reg) >= 8:
        lines.append(
            "**Full-sample OLS (TextBlob):** Not reported — lag-1 sentiment has **no usable variation** "
            "in this window (often duplicate/empty review columns). Check **Advanced column overrides**."
        )
    rf2 = result.regression_full_transformer
    if rf2 and _ols_block_valid(rf2):
        lines.append(
            f"**Full-sample OLS (RoBERTa, HC3):** slope **{rf2['coef']:.6f}** "
            f"95% CI **[{rf2['ci_low']:.6f}, {rf2['ci_high']:.6f}]**, p **{rf2['p_value']:.4f}**, R² **{rf2['r2']:.4f}**."
        )

    tm = result.text_mining
    if tm and tm.get("bigrams"):
        top = ", ".join(f"{b['phrase']} ({b['count']})" for b in tm["bigrams"][:5])
        lines.append(
            f"**Themes (bigrams on flagged reviews, *n*={tm['n_docs']}):** {top} — use for **release triage** wording."
        )
    if tm and tm.get("nmf_topics"):
        t0 = str(tm["nmf_topics"][0].get("top_words", ""))
        snippet = (t0[:120] + "…") if len(t0) > 120 else t0
        lines.append(
            f"**NMF topics (flagged corpus):** {len(tm['nmf_topics'])} components; e.g. *{snippet}*"
        )

    if tb is not None and tr is not None:
        gap = abs(tb - tr)
        if gap > 0.25:
            lines.append(
                "**Model divergence:** TextBlob and RoBERTa lag-1 correlations differ markedly — "
                "**prioritize the disagreement queue** and avoid betting the story on a single score."
            )
        else:
            lines.append(
                "**Model agreement:** Both scorers suggest a **similar directional** lag-1 association in this window — still validate with held-out periods."
            )

    if dis is not None:
        if dis > 0.45:
            lines.append(
                "**Language risk:** High mean |TextBlob − RoBERTa| — informal or ambiguous phrasing is common; **schedule a human read** of top disagreements weekly."
            )
        elif dis > 0.25:
            lines.append(
                "**Language risk:** Moderate scorer disagreement — use the ranked queue for **spot checks** after major releases."
            )
        else:
            lines.append(
                "**Language risk:** Scorers mostly agree on average — disagreement rows are still the fastest place to catch **edge cases**."
            )

    h = result.holdout_textblob
    if h:
        pv = h["p_value_lag_sentiment"]
        cpa = h.get("corr_pred_vs_actual_holdout")
        cpa_s = f"{cpa:.4f}" if cpa is not None and _finite_f(cpa) else "n/a"
        pv_part = _p_value_label(pv)
        lines.append(
            f"**Pseudo-forecasting (TextBlob):** OLS with **HC3** on first **{h['n_train']}** days; "
            f"**{h['n_holdout']}-day** holdout through **{h['holdout_last_date']}**. "
            f"Train-sample slope p-value {pv_part}. "
            f"Holdout corr(predicted, actual return) **{cpa_s}**; RMSE **{h['rmse_holdout']:.6f}**."
        )
    elif len(result.reg) >= 33:
        lines.append(
            "**Pseudo-forecasting (TextBlob):** Not shown — the train window did not support a reliable **HC3** slope "
            "(usually **no variation** in lag-1 sentiment). Fix columns/CSV as for full-sample OLS."
        )
    elif result.overlap_days >= 15:
        lines.append(
            "**Pseudo-forecasting:** Holdout skipped — need more overlapping days (defaults: **25+** train, **8+** holdout)."
        )

    if result.stability_df is not None and len(result.stability_df) > 0:
        ec = result.stability_df["expanding_corr_textblob"].dropna()
        rc = result.stability_df["rolling_corr_textblob"].dropna()
        if len(ec) and len(rc):
            lines.append(
                "**Rolling vs expanding *r*:** Compare the chart — if rolling correlation swings while expanding drifts, the sentiment–return link is **unstable over time** (common in alt data)."
            )

    h2 = result.holdout_transformer
    if h2:
        pv2 = h2["p_value_lag_sentiment"]
        cpa2 = h2.get("corr_pred_vs_actual_holdout")
        cpa2_s = f"{cpa2:.4f}" if cpa2 is not None and _finite_f(cpa2) else "n/a"
        pv2_part = _p_value_label(pv2)
        lines.append(
            f"**Pseudo-forecasting (RoBERTa):** Same time ordering — train **{h2['n_train']}**, holdout **{h2['n_holdout']}** through **{h2['holdout_last_date']}**. "
            f"Slope p-value {pv2_part}. Holdout corr(pred, actual) **{cpa2_s}**; RMSE **{h2['rmse_holdout']:.6f}**."
        )

    if result.worst_return_day and result.worst_return_value is not None:
        lines.append(
            f"**Drawdown focus:** Worst return **{result.worst_return_day}** (**{result.worst_return_value:.4f}**) — "
            "compare against **product changes**, **campaigns**, and **disagreement reviews** dated just before."
        )

    lines.append(
        "**Next step:** Export the **PDF report** for IR/product; re-run **weekly** on a fresh export with the same column mapping."
    )
    return lines
