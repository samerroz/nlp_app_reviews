"""
Business-facing labels, plain-English section intros, chart summaries, technical strips,
and optional grounded LLM executive narrative (same facts only).
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from typing import TYPE_CHECKING, Any

import pandas as pd

if TYPE_CHECKING:
    from pipeline import PipelineResult

# ── Primary-path vocabulary (avoid library names in headlines) ───────────────
LABEL_LEXICON_SCORER = "Fast dictionary-based score"
LABEL_NEURAL_SCORER = "Contextual AI score"
TECH_LEXICON = "TextBlob"
TECH_NEURAL = "RoBERTa"

# Short labels for chart legends (avoid wrapping)
CHART_LEGEND_LEXICON = "Dictionary-based"
CHART_LEGEND_NEURAL = "Contextual AI"

LAG_EXPLAINER = (
    "Each point is one trading day: **yesterday’s** average review tone vs **today’s** return in your `ret` column "
    "(one-day lag so the score cannot react to the same day’s price)."
)

TECH_STRIP_TITLE = "For your technical team — exact numbers"


def landing_subscription_value_html() -> str:
    """Landing: who it’s for + why recurring uploads matter (HTML fragment)."""
    return (
        '<div class="rs-signal-box" style="margin: 0.35rem 0 0.75rem 0;">'
        "<p style=\"margin:0 0 0.45rem 0;\"><strong>Who uses this — and why they come back</strong></p>"
        "<ul style=\"margin:0; padding-left:1.15rem; color:#cbd5e1; font-size:0.86rem; line-height:1.48;\">"
        "<li><strong>IR / execs:</strong> Check whether app-store tone and next-day moves line up "
        "before you attribute a swing to “reviews.”</li>"
        "<li><strong>Product &amp; engineering:</strong> See tone around releases and pull the "
        "<strong>exact reviews</strong> to read first (especially where scores disagree).</li>"
        "<li><strong>Every new export:</strong> Same questions answered again — <strong>what shifted</strong>, "
        "what to monitor, what to escalate — without rebuilding models yourself.</li>"
        "</ul>"
        "<p style=\"margin:0.45rem 0 0 0; font-size:0.8rem; color:#94a3b8;\">"
        "Ongoing use = <strong>consistent monitoring</strong>, not a one-off chart. Not trading advice.</p>"
        "</div>"
    )


def overview_value_loop_markdown() -> str:
    """Overview: neutral map of where answers live (no sales framing)."""
    return (
        "**Where to find answers in this report:**\n\n"
        "| Question | Where |\n"
        "|----------|-------|\n"
        "| Do we have enough data lined up with trading days? | **Coverage** |\n"
        "| What’s the headline for leadership, in plain English? | **For leadership / IR** |\n"
        "| What should product read first? | **For product & support** + **Reviews** |\n"
        "| What should we watch on the next export? | **For your next check-in** + **Key findings** |\n"
        "| How did tone and returns move — and stay stable? | **Market & stats** |\n"
        "| Word pairs and topics from rough periods? | **Themes** |\n"
        "| Memo for email / deck? | **Brief** + **Q&A** |\n\n"
        "_Use the **same column mapping** on the next export so metrics stay comparable._"
    )


def _focus_recommendation_from_blocks(
    assoc: dict[str, Any],
    stab_block: dict[str, Any],
    holdout_block: dict[str, Any],
) -> str:
    """Single prioritized action for busy readers — rules only, no new data."""
    hl = holdout_block.get("lexicon", {})
    if hl.get("available") and hl.get("verdict") == "weakens":
        return (
            "**Pull a fresh export and re-run** before anyone repeats the “reviews drove the stock” story — "
            "the latest days didn’t match the simple rule you fit on older data."
        )
    if stab_block.get("regime") == "recent_noisy":
        return (
            "**Talk about the last few weeks explicitly** in meetings — the headline full-window correlation hides "
            "swings that showed up in the rolling lines."
        )
    if assoc.get("both_scorers") and assoc.get("r_gap") is not None and float(assoc["r_gap"]) > 0.15:
        return (
            "**Read the disagreement queue** before leadership sees a single correlation number — "
            "the two scorers disagree on how strong the link is."
        )
    mad = assoc.get("mean_abs_disagreement")
    if mad is not None and _finite(mad) and float(mad) > 0.28:
        return (
            "**Calendar a short disagreement triage** (top rows) every cycle — that’s the fastest path to sarcasm, "
            "negation, and wording that star ratings miss."
        )
    if assoc.get("strength_lexicon") in ("negligible", "weak"):
        return (
            "**Lean on Themes and worst-day context** for product and support decisions — don’t over-sell a tight "
            "investor narrative off a weak same-sample link to returns."
        )
    if hl.get("available") and hl.get("verdict") in ("confirms", "partly_confirms"):
        return (
            "**Keep the same export cadence** — latest days didn’t obviously break the pattern you saw in-sample "
            "(still monitoring only, not a forecast)."
        )
    return (
        "**Use a steady re-export rhythm** (weekly or monthly) with the **same CSV columns** — "
        "that’s how you compare this window fairly to the next."
    )


def _finite(x: Any) -> bool:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return False
    return math.isfinite(v)


def chart_summary_stats(merged: pd.DataFrame) -> dict[str, Any]:
    """Aggregates only — safe to pass to LLM or show in prose."""
    out: dict[str, Any] = {"n_trading_days": int(len(merged)) if len(merged) else 0}
    if merged is None or merged.empty:
        return out
    d = merged.sort_values("date")
    for col, key in (("avg_textblob", "lexicon"), ("avg_transformer", "neural")):
        if col not in d.columns:
            continue
        s = pd.to_numeric(d[col], errors="coerce").dropna()
        if len(s) == 0:
            continue
        out[f"{key}_mean"] = float(s.mean())
        out[f"{key}_std"] = float(s.std(ddof=1)) if len(s) > 1 else None
        out[f"{key}_first"] = float(s.iloc[0])
        out[f"{key}_last"] = float(s.iloc[-1])
        out[f"{key}_min"] = float(s.min())
        out[f"{key}_max"] = float(s.max())
    return out


def trajectory_intro_markdown(stats: dict[str, Any]) -> str:
    """Plain-language blurb above the dual-sentiment chart."""
    n = stats.get("n_trading_days", 0)
    head = (
        f"**What this chart shows:** How average daily review tone moves over **{n}** overlapping market days. "
        f"**{LABEL_LEXICON_SCORER}** is quick and consistent; **{LABEL_NEURAL_SCORER}** reads full sentences and catches sarcasm better. "
        "When the two lines diverge, treat that as a cue to read those days’ reviews manually."
    )
    mid = ""
    if "lexicon_first" in stats and "lexicon_last" in stats:
        lf, ll = stats["lexicon_first"], stats["lexicon_last"]
        delta = ll - lf
        direction = "rose" if delta > 0.02 else "fell" if delta < -0.02 else "was roughly flat"
        mid = (
            f"\n\n**Read in one line:** The dictionary-based line **{direction}** from about **{lf:.2f}** to **{ll:.2f}** over the window."
        )
    tail = f"\n\n*{LAG_EXPLAINER}*"
    return f"{head}{mid}{tail}"


def stability_summary_stats(stab: pd.DataFrame | None) -> dict[str, Any]:
    out: dict[str, Any] = {}
    if stab is None or stab.empty:
        return out
    d = stab.sort_values("date")
    last = d.iloc[-1]
    out["last_date"] = str(last.get("date", ""))
    for col in (
        "expanding_corr_textblob",
        "rolling_corr_textblob",
        "expanding_corr_roberta",
        "rolling_corr_roberta",
    ):
        if col in d.columns and pd.notna(last.get(col)):
            try:
                out[col] = float(last[col])
            except (TypeError, ValueError):
                pass
    return out


def stability_intro_markdown() -> str:
    return (
        "**How this differs from the chart above:** The trajectory chart shows **average mood** each day. "
        "This chart shows whether the **strength of the link** between yesterday’s review tone and **next day’s** "
        f"return **changes as you add more days** — an “expanding” correlation uses all history to date; "
        "a **20-day rolling** line reacts faster to recent weeks. "
        "If rolling swings wildly while expanding drifts, the relationship is **unstable over time** — common with alt data."
    )


def holdout_intro_markdown() -> str:
    return (
        "**What this block is for:** We fit a simple straight-line rule on **older** trading days only, "
        "then score how well it lines up with **actual returns on the most recent** days (chronological holdout). "
        "That is a **sanity check** on whether the pattern persists — **not** a trading backtest and not investment advice. "
        f"**{LABEL_LEXICON_SCORER}** and **{LABEL_NEURAL_SCORER}** are scored separately when both exist."
    )


def _corr_strength_word(r: float | None) -> str:
    if r is None or not _finite(r):
        return "unavailable"
    a = abs(float(r))
    if a < 0.08:
        return "negligible"
    if a < 0.2:
        return "weak"
    if a < 0.35:
        return "modest"
    return "notable"


def _holdout_verdict(cpa: Any, headline_r: float | None) -> str:
    """Compare holdout corr(pred, actual) to full-sample headline r — template labels only."""
    if cpa is None or not _finite(cpa):
        return "unavailable"
    if headline_r is None or not _finite(headline_r):
        return "inconclusive"
    cf, rf = float(cpa), float(headline_r)
    if abs(cf) < 0.08 and abs(rf) < 0.12:
        return "inconclusive"
    same_sign = (cf > 0) == (rf > 0)
    if same_sign and abs(cf) >= 0.12 and abs(rf) >= 0.08:
        return "confirms"
    if same_sign and (abs(cf) >= 0.05 or abs(rf) >= 0.05):
        return "partly_confirms"
    if not same_sign and abs(cf) > 0.05 and abs(rf) > 0.08:
        return "weakens"
    return "inconclusive"


def build_insight_facts(res: "PipelineResult") -> dict[str, Any]:
    """
    Deterministic, JSON-friendly facts for 'so what' copy and optional LLM.
    Every qualitative label must trace to rules on pipeline outputs only.
    """
    cs = chart_summary_stats(res.merged)
    ss = stability_summary_stats(res.stability_df)

    traj: dict[str, Any] = {
        "n_days": int(cs.get("n_trading_days", 0)),
        "lexicon_direction": "flat",
        "lexicon_delta": 0.0,
        "neural_available": False,
        "gap_widened": None,
        "gap_narrowed": None,
        "lexicon_std": cs.get("lexicon_std"),
        "volatility_note": "unknown",
    }
    if "lexicon_first" in cs and "lexicon_last" in cs:
        lf, ll = float(cs["lexicon_first"]), float(cs["lexicon_last"])
        traj["lexicon_delta"] = ll - lf
        if ll - lf > 0.02:
            traj["lexicon_direction"] = "rose"
        elif ll - lf < -0.02:
            traj["lexicon_direction"] = "fell"
        else:
            traj["lexicon_direction"] = "flat"
    std_l = cs.get("lexicon_std")
    if std_l is not None and _finite(std_l):
        if float(std_l) > 0.12:
            traj["volatility_note"] = "high"
        elif float(std_l) > 0.06:
            traj["volatility_note"] = "moderate"
        else:
            traj["volatility_note"] = "low"

    m = res.merged
    if m is not None and len(m) and "avg_textblob" in m.columns:
        if "avg_transformer" in m.columns and m["avg_transformer"].notna().any():
            traj["neural_available"] = True
            sub = m.sort_values("date").dropna(subset=["avg_textblob", "avg_transformer"])
            if len(sub) >= 9:
                n = len(sub)
                k = max(3, n // 3)
                early = sub.iloc[:k]
                late = sub.iloc[-k:]
                gap_e = float((early["avg_textblob"] - early["avg_transformer"]).abs().mean())
                gap_l = float((late["avg_textblob"] - late["avg_transformer"]).abs().mean())
                if gap_l > gap_e + 0.04:
                    traj["gap_widened"] = True
                    traj["gap_narrowed"] = False
                elif gap_e > gap_l + 0.04:
                    traj["gap_widened"] = False
                    traj["gap_narrowed"] = True
                else:
                    traj["gap_widened"] = False
                    traj["gap_narrowed"] = False

    r_lex = res.corr_textblob_lag1
    r_neu = res.corr_transformer_lag1
    r_lex_f = float(r_lex) if r_lex is not None and _finite(r_lex) else None
    r_neu_f = float(r_neu) if r_neu is not None and _finite(r_neu) else None
    assoc: dict[str, Any] = {
        "r_lexicon": r_lex,
        "r_neural": r_neu,
        "strength_lexicon": _corr_strength_word(r_lex_f),
        "strength_neural": _corr_strength_word(r_neu_f),
        "r_gap": None,
        "both_scorers": False,
        "mean_abs_disagreement": res.mean_abs_disagreement,
        "mood_drift": traj["lexicon_direction"],
        "bridge_sentence": "",
    }
    if r_lex_f is not None and r_neu_f is not None:
        assoc["both_scorers"] = True
        assoc["r_gap"] = abs(r_lex_f - r_neu_f)
    drift_w = traj["lexicon_direction"]
    strength = assoc["strength_lexicon"]
    if r_lex_f is not None:
        if drift_w == "rose" and strength in ("negligible", "weak"):
            assoc["bridge_sentence"] = (
                f"Average tone **rose** over the window, but the lag-1 link to next-day returns stays **{strength}** "
                "— mood moved without a strong linear tie to `ret` in this sample."
            )
        elif drift_w == "fell" and strength in ("negligible", "weak"):
            assoc["bridge_sentence"] = (
                f"Average tone **fell** over the window, while the lag-1 link to returns is still **{strength}**."
            )
        elif drift_w in ("rose", "fell") and strength in ("modest", "notable"):
            assoc["bridge_sentence"] = (
                f"Tone **{drift_w}** over the window, and the lag-1 association with returns is **{strength}** — "
                "worth monitoring alongside product and news, not as a sole driver."
            )
        else:
            assoc["bridge_sentence"] = (
                f"Full-window lag-1 association is **{strength}** for the dictionary-based score "
                f"(r ≈ **{r_lex_f:.3f}**) — see the trajectory above for how daily averages moved."
            )
    else:
        assoc["bridge_sentence"] = "Lag-1 correlation could not be summarized reliably for this window."

    stab_df = res.stability_df
    stab_block: dict[str, Any] = {
        "available": stab_df is not None and not stab_df.empty,
        "regime": "unavailable",
        "gap_end_exp_roll_tb": None,
        "rolling_std_tb": None,
        "last_expanding_tb": ss.get("expanding_corr_textblob"),
        "last_rolling_tb": ss.get("rolling_corr_textblob"),
    }
    if stab_df is not None and not stab_df.empty and "rolling_corr_textblob" in stab_df.columns:
        d = stab_df.sort_values("date")
        both = d.dropna(subset=["rolling_corr_textblob", "expanding_corr_textblob"])
        if len(both) >= 3:
            gap_ser = (both["rolling_corr_textblob"] - both["expanding_corr_textblob"]).abs()
            stab_block["gap_end_exp_roll_tb"] = float(gap_ser.iloc[-1])
            stab_block["rolling_std_tb"] = (
                float(both["rolling_corr_textblob"].std(ddof=1))
                if len(both) > 1
                else None
            )
            ge = stab_block["gap_end_exp_roll_tb"]
            rs = stab_block["rolling_std_tb"]
            if ge is not None and ge > 0.18:
                stab_block["regime"] = "recent_noisy"
            elif rs is not None and rs > 0.14:
                stab_block["regime"] = "recent_noisy"
            elif ge is not None and ge < 0.12 and (rs is None or rs < 0.11):
                stab_block["regime"] = "stable"
            else:
                stab_block["regime"] = "inconclusive"
    elif stab_block["available"]:
        stab_block["regime"] = "inconclusive"

    def _one_holdout(h: dict[str, Any] | None, headline: float | None) -> dict[str, Any]:
        if not isinstance(h, dict):
            return {"available": False, "verdict": "unavailable", "cpa": None}
        cpa = h.get("corr_pred_vs_actual_holdout")
        return {
            "available": True,
            "verdict": _holdout_verdict(cpa, headline),
            "cpa": float(cpa) if cpa is not None and _finite(cpa) else None,
            "n_train": h.get("n_train"),
            "n_holdout": h.get("n_holdout"),
            "p_slope": h.get("p_value_lag_sentiment"),
        }

    holdout_block = {
        "lexicon": _one_holdout(res.holdout_textblob, r_lex_f),
        "neural": _one_holdout(res.holdout_transformer, r_neu_f),
    }

    exec_lines: list[str] = []
    if traj["n_days"]:
        exec_lines.append(
            f"**Mood path:** Dictionary-based daily tone **{traj['lexicon_direction']}** over **{traj['n_days']}** overlapping days"
            + (
                f" (day-to-day volatility **{traj['volatility_note']}**)."
                if traj["volatility_note"] != "unknown"
                else "."
            )
        )
    if r_lex_f is not None:
        exec_lines.append(
            f"**Link to returns:** Lag-1 association is **{assoc['strength_lexicon']}** (r ≈ {r_lex_f:.3f}) — monitoring signal only."
        )
    if assoc["both_scorers"] and assoc.get("r_gap") is not None and float(assoc["r_gap"]) > 0.15:  # type: ignore[arg-type]
        exec_lines.append(
            "**Two scores disagree on strength** — prioritize reading rows where dictionary vs contextual AI diverge."
        )
    elif res.mean_abs_disagreement is not None and _finite(res.mean_abs_disagreement) and float(res.mean_abs_disagreement) > 0.28:
        exec_lines.append(
            "**Review-level scorer gaps are elevated** — use the disagreement queue for spot checks."
        )
    if stab_block["regime"] == "recent_noisy":
        exec_lines.append(
            "**Stability:** Rolling vs expanding correlation diverge — treat the headline r as **time-sensitive**, not stable."
        )
    elif stab_block["regime"] == "stable" and stab_block["available"]:
        exec_lines.append(
            "**Stability:** Rolling and expanding correlations align at the end of the window — the link looks **less fragile** than in many alt-data pilots."
        )
    hl = holdout_block["lexicon"]
    if hl["available"] and hl["verdict"] == "confirms":
        exec_lines.append(
            "**Recent-week check:** Simple rule still lines up with latest returns — **supports** (not proves) the headline pattern."
        )
    elif hl["available"] and hl["verdict"] == "weakens":
        exec_lines.append(
            "**Recent-week check:** Latest days **don’t** match the old rule as well — **down-weight** the headline correlation for decisions."
        )
    if len(exec_lines) < 3 and assoc["bridge_sentence"]:
        bs = assoc["bridge_sentence"]
        if not any(bs[: min(30, len(bs))] in x for x in exec_lines):
            exec_lines.append(bs)
    if len(exec_lines) < 3:
        exec_lines.append(
            "_Association ≠ causation — combine with releases, support volume, and macro news._"
        )
    if len(exec_lines) < 3:
        exec_lines.append(
            "Extend the overlap window or re-run after major releases if you need a clearer read."
        )
    executive_summary = exec_lines[:3]
    focus_recommendation = _focus_recommendation_from_blocks(assoc, stab_block, holdout_block)

    return {
        "trajectory": traj,
        "association": assoc,
        "stability": stab_block,
        "holdout": holdout_block,
        "executive_summary": executive_summary,
        "focus_recommendation": focus_recommendation,
    }


def insight_trajectory_section_markdown(facts: dict[str, Any]) -> str:
    t = facts["trajectory"]
    lines = ["**What this means**", ""]
    if t["n_days"]:
        lines.append(
            f"- The lines summarize **how customers sounded on average** each trading day — **{t['n_days']}** days in view."
        )
    if t["lexicon_direction"] != "flat":
        lines.append(
            f"- Dictionary-based tone **{t['lexicon_direction']}** by about **{abs(t['lexicon_delta']):.2f}** points start → end."
        )
    else:
        lines.append("- Dictionary-based tone was **roughly flat** — no strong drift in average mood over the window.")
    if t["volatility_note"] in ("high", "moderate"):
        lines.append(
            f"- Day-to-day swings in that score are **{t['volatility_note']}** — expect a jagged line even when nothing “broke” in the product."
        )
    if t["neural_available"]:
        if t.get("gap_widened"):
            lines.append(
                f"- The **{LABEL_NEURAL_SCORER}** line **diverged more from the dictionary line toward the end** — recent reviews may be more nuanced or informal; read those dates."
            )
        elif t.get("gap_narrowed"):
            lines.append(
                "- The two scorers **agreed more toward the end** than at the start — language may have gotten simpler or sentiment clearer."
            )
        else:
            lines.append(
                f"- **{LABEL_NEURAL_SCORER}** is on — when it diverges from the dictionary line, treat that as a **triage signal**."
            )
    else:
        lines.append(
            f"- **{LABEL_NEURAL_SCORER}** is off this run — you only see the fast dictionary path; enable neural scoring on the start page for a second opinion."
        )
    return "\n".join(lines)


def insight_association_section_markdown(facts: dict[str, Any]) -> str:
    a = facts["association"]
    lines = ["**What this means**", ""]
    if a["bridge_sentence"]:
        lines.append(f"- {a['bridge_sentence']}")
    if a["both_scorers"] and a.get("r_gap") is not None:
        rg = float(a["r_gap"])
        if rg > 0.12:
            lines.append(
                f"- The two scorers disagree on **how strong** the lag-1 link is (|Δr| ≈ **{rg:.3f}**) — **do not** pick one score for a board narrative without reading the disagreement queue."
            )
        else:
            lines.append(
                f"- Both scorers point to a **similar-sized** lag-1 link (|Δr| ≈ **{rg:.3f}**) — a bit more confidence the pattern isn’t a lexicon artifact alone."
            )
    mad = a.get("mean_abs_disagreement")
    if mad is not None and _finite(mad) and float(mad) > 0.35:
        lines.append(
            f"- Typical per-review gap between scorers is **high** (~**{float(mad):.2f}**) — language is often ambiguous; manual reads are high leverage."
        )
    return "\n".join(lines)


def insight_stability_section_markdown(facts: dict[str, Any]) -> str:
    s = facts["stability"]
    lines = ["**What this means**", ""]
    if not s["available"] or s["regime"] == "unavailable":
        lines.append("- Not enough data to plot stability curves — skip this interpretation until the overlap is longer.")
    elif s["regime"] == "stable":
        lines.append(
            "- **Expanding** (all history so far) and **20-day rolling** correlations end in a **similar** place — the headline link is **less obviously a one-week fluke**."
        )
    elif s["regime"] == "recent_noisy":
        lines.append(
            "- **Rolling** correlation moves a lot vs **expanding** — the relationship **changes in recent weeks**. The full-window r above can **over-smooth** what just happened."
        )
    else:
        lines.append(
            "- Stability is **mixed** — neither clearly rock-solid nor clearly broken; extend the history or re-run after major events."
        )
    if s.get("gap_end_exp_roll_tb") is not None:
        lines.append(
            f"- End-of-window |rolling − expanding| ≈ **{float(s['gap_end_exp_roll_tb']):.3f}** for the dictionary-based series (smaller usually means calmer)."
        )
    return "\n".join(lines)


def insight_holdout_section_markdown(facts: dict[str, Any]) -> str:
    h = facts["holdout"]
    lines = ["**What this means**", ""]
    for key, label in (("lexicon", LABEL_LEXICON_SCORER), ("neural", LABEL_NEURAL_SCORER)):
        block = h.get(key, {})
        if not block.get("available"):
            lines.append(f"- **{label}:** No holdout — need more overlapping days or stable lag-1 score variation.")
            continue
        v = block.get("verdict", "inconclusive")
        cpa = block.get("cpa")
        cpa_s = f"**{cpa:.3f}**" if cpa is not None else "n/a"
        if v == "confirms":
            lines.append(
                f"- **{label}:** Holdout alignment **supports** the headline pattern (pred vs actual corr {cpa_s}) — still not a backtest."
            )
        elif v == "partly_confirms":
            lines.append(
                f"- **{label}:** Holdout is **directionally consistent** but soft (corr {cpa_s}) — treat as **weak** confirmation."
            )
        elif v == "weakens":
            lines.append(
                f"- **{label}:** Latest days **weaken** the story (holdout corr {cpa_s} vs your headline lag-1 r) — be cautious using the full-sample slope for forward talk."
            )
        else:
            lines.append(
                f"- **{label}:** Holdout is **inconclusive** (corr {cpa_s}) — too few days or too noisy to say if the rule survived."
            )
    return "\n".join(lines)


def overview_ir_markdown(company: str, res: "PipelineResult", facts: dict[str, Any]) -> str:
    """Short IR-facing verdict plus up to two executive_summary lines from insight_facts."""
    co = company.strip() or "This company"
    chunks: list[str] = []
    r = res.corr_textblob_lag1
    a = facts.get("association") or {}
    strength = a.get("strength_lexicon") or "weak"
    if r is None or not _finite(r):
        chunks.append(
            f"**{co}:** We could not summarize a dictionary-based **lag-1** link between average review tone and next-day **`ret`** "
            "in this window — often too little variation in daily average scores."
        )
    else:
        rf = float(r)
        dirw = "positive" if rf > 0 else "negative"
        chunks.append(
            f"**{co}:** Over **{res.overlap_days}** overlapping trading days, **yesterday’s** average review tone and **today’s** return "
            f"show a **{strength}**, **{dirw}** association (Pearson **r** ≈ **{rf:.3f}**). "
            "**Association ≠ causation** — use alongside earnings, product, and macro news."
        )
    for bullet in (facts.get("executive_summary") or [])[:2]:
        if isinstance(bullet, str) and bullet.strip():
            chunks.append(bullet)
    return "\n\n".join(chunks)


def worst_day_review_samples(res: "PipelineResult", k: int = 3) -> list[dict[str, str]]:
    """Lazy-import wrapper — avoids import cycle (pipeline → llm_brief → executive_copy)."""
    from pipeline import sample_reviews_for_trading_date

    return sample_reviews_for_trading_date(res.reviews_scored, res.worst_return_day, k=k)


def best_day_review_samples(res: "PipelineResult", k: int = 4) -> tuple[str | None, float | None, list[dict[str, str]]]:
    """
    Find the best return day and pull review excerpts for it.
    Returns (best_day_str, best_ret_value, samples_list).
    """
    from pipeline import sample_reviews_for_trading_date

    if res.merged is None or res.merged.empty or "ret" not in res.merged.columns:
        return None, None, []
    rets = pd.to_numeric(res.merged["ret"], errors="coerce")
    if not rets.notna().any():
        return None, None, []
    best_idx = rets.idxmax()
    best_val = float(rets.max())
    best_date = res.merged.loc[best_idx, "date"]
    best_day_str = str(best_date.date()) if hasattr(best_date, "date") else str(best_date)
    samples = sample_reviews_for_trading_date(res.reviews_scored, best_day_str, k=k)
    return best_day_str, best_val, samples


def build_chart_callouts(res: "PipelineResult") -> dict[str, str]:
    """
    Scan the merged DataFrame and stability DataFrame for specific, datable events.
    Returns plain-English callout strings for each chart section.
    Callouts are 1-2 sentences a non-technical reader can act on.
    """
    out: dict[str, str] = {}
    m = res.merged

    # ── Trajectory callout ────────────────────────────────────────────────────
    if m is not None and not m.empty and "avg_textblob" in m.columns:
        d = m.sort_values("date").copy()
        d["_tb"] = pd.to_numeric(d["avg_textblob"], errors="coerce")
        d["_tb_chg"] = d["_tb"].diff()

        traj_parts: list[str] = []

        # Worst stock day cross-reference
        if res.worst_return_day and res.worst_return_value is not None:
            wv_pct = abs(float(res.worst_return_value)) * 100
            traj_parts.append(
                f"The worst trading day in this window was **{res.worst_return_day}** (−{wv_pct:.1f}%) — "
                "look for a dip in the sentiment line around that date."
            )

        # Biggest single-day sentiment drop
        if d["_tb_chg"].notna().any():
            min_idx = d["_tb_chg"].idxmin()
            min_chg = float(d.loc[min_idx, "_tb_chg"])
            min_date = d.loc[min_idx, "date"]
            min_date_s = str(min_date.date()) if hasattr(min_date, "date") else str(min_date)
            if abs(min_chg) > 0.04:
                traj_parts.append(
                    f"Biggest single-day sentiment drop: **{min_date_s}** — if something happened to the app or support that day, this is where to look."
                )

        # Biggest gap between the two scorers
        if "avg_transformer" in d.columns and d["avg_transformer"].notna().any():
            d["_gap"] = (d["_tb"] - pd.to_numeric(d["avg_transformer"], errors="coerce")).abs()
            if d["_gap"].notna().any() and float(d["_gap"].max()) > 0.15:
                gap_idx = d["_gap"].idxmax()
                gap_date = d.loc[gap_idx, "date"]
                gap_date_s = str(gap_date.date()) if hasattr(gap_date, "date") else str(gap_date)
                traj_parts.append(
                    f"The two scoring methods disagreed most on **{gap_date_s}** — worth reading those reviews, as they may contain sarcasm or nuanced language the simpler method missed."
                )

        if traj_parts:
            out["trajectory"] = " ".join(traj_parts)

    # ── Association callout ───────────────────────────────────────────────────
    r = res.corr_textblob_lag1
    if r is not None and _finite(r):
        rf = float(r)
        dir_phrase = "tend to line up — warmer reviews, better next day" if rf > 0 else "move in opposite directions"
        strength = _corr_strength_word(rf)
        out["association"] = (
            f"Overall, the dots {dir_phrase} — a **{strength}** pattern across the full window. "
            "Outlier dots (top-left or bottom-right) are days the pattern broke — look for those clusters."
        )

    # ── Stability callout ─────────────────────────────────────────────────────
    stab = res.stability_df
    if stab is not None and not stab.empty and "rolling_corr_textblob" in stab.columns:
        d_s = stab.sort_values("date").copy()
        d_s["_rc"] = pd.to_numeric(d_s["rolling_corr_textblob"], errors="coerce")
        d_s["_rc_chg"] = d_s["_rc"].diff().abs()
        if d_s["_rc_chg"].notna().any() and float(d_s["_rc_chg"].max()) > 0.08:
            shift_idx = d_s["_rc_chg"].idxmax()
            shift_date = d_s.loc[shift_idx, "date"]
            shift_date_s = str(shift_date.date()) if hasattr(shift_date, "date") else str(shift_date)
            out["stability"] = (
                f"The link shifted most sharply around **{shift_date_s}**. "
                "If a product release, pricing change, or news event happened that week, that may explain it. "
                "Use the **Reviews** tab to read what customers were saying then."
            )
        else:
            out["stability"] = (
                "The link stayed relatively steady across the window — "
                "no single week shows a dramatic shift."
            )

    return out


def build_surface_low_bullets(res: "PipelineResult", facts: dict[str, Any]) -> list[str]:
    """Short, prominent risk / thin-data flags for the top of Overview."""
    lines: list[str] = []
    od = int(res.overlap_days or 0)
    if od < 20:
        lines.append(
            f"**Thin overlap:** Only **{od}** trading days line up — treat correlations as **exploratory**, not board-ready."
        )
    elif od < 45:
        lines.append(
            f"**Moderate depth:** **{od}** overlapping days — usable for internal monitoring; extend the window when you can."
        )

    hl = (facts.get("holdout") or {}).get("lexicon") or {}
    if hl.get("available") and hl.get("verdict") == "weakens":
        lines.append(
            "**Recent-week check:** The simple rule fits **older** days better than the **newest** returns — "
            "don’t lean on the full-sample lag-1 **r** as a forecast."
        )

    stab = facts.get("stability") or {}
    if stab.get("regime") == "recent_noisy":
        lines.append(
            "**Unstable link:** Short-window correlation **wanders** vs the long cumulative curve — see **Market & stats**."
        )

    mad = (facts.get("association") or {}).get("mean_abs_disagreement")
    if mad is not None and _finite(mad) and float(mad) > 0.28:
        lines.append(
            "**High scorer disagreement:** Average per-review gap is elevated — read the **Reviews** queue before attributing moves to “tone.”"
        )

    tb = res.corr_textblob_lag1
    if tb is not None and _finite(tb) and abs(float(tb)) < 0.08 and od >= 25:
        lines.append(
            "**Weak headline link:** Dictionary lag-1 correlation is **tiny** here — fine as a monitor, weak as a sole investor story."
        )

    return lines[:5]


def build_market_trajectory_beats(company: str, chart_stats: dict[str, Any], facts: dict[str, Any]) -> tuple[str, str, str]:
    traj = facts.get("trajectory") or {}
    n = int(chart_stats.get("n_trading_days") or 0)
    quick = (
        f"This answers: **Did average app-store tone drift** over the overlap window for **{company}** "
        f"({n} trading days), and did the two scorers agree on that path?"
    )
    data_line = (
        "We merged your reviews to market trading days, averaged each sentiment score per day, and plotted those daily means — "
        "the same series the correlation math uses one day later."
    )
    tdir = traj.get("lexicon_direction", "flat")
    insight = f"For **{company}**, dictionary-based daily tone **{tdir}** over this window"
    if traj.get("neural_available"):
        if traj.get("gap_widened"):
            insight += (
                f"; **{LABEL_NEURAL_SCORER}** **diverged more from the dictionary line toward the end** — "
                "when the lines split, open **Reviews** and read those dates (or the disagreement queue)."
            )
        elif traj.get("gap_narrowed"):
            insight += "; the two scorers **agreed more toward the end** than at the start."
        else:
            insight += (
                "; when the lines separate, use **Reviews** to spot sarcasm or wording the fast score missed."
            )
    else:
        insight += (
            f" ({LABEL_NEURAL_SCORER} was off this run — enable it on the start page for a second line.)"
        )
    return quick, data_line, insight


def build_market_association_beats(company: str, res: "PipelineResult", facts: dict[str, Any]) -> tuple[str, str, str]:
    quick = (
        "This answers: **When customers sounded happier or harsher yesterday (average review tone), "
        "did the stock tend to move a particular way today?** One-day lag so tone cannot react to the same day’s price."
    )
    data_line = (
        "We correlate **lag-1 daily mean sentiment** against **same-row `ret`** across every overlapping trading day in your files, "
        "then show regression detail under *For your technical team*."
    )
    a = facts.get("association") or {}
    r = res.corr_textblob_lag1
    strength = a.get("strength_lexicon") or "weak"
    if r is None or not _finite(r):
        ins = (
            f"For **{company}**, we couldn’t summarize a dictionary lag-1 link in this window — "
            "often too little day-to-day movement in average tone. Check column mapping and try a longer export."
        )
    else:
        rf = float(r)
        dirw = "positive" if rf > 0 else "negative"
        ins = (
            f"For **{company}**, the dictionary-based lag-1 association is **{strength}** and **{dirw}** "
            f"(Pearson **r** ≈ **{rf:.3f}**). Treat as **context** next to product and news — **not** the only driver of `ret`."
        )
    if a.get("both_scorers") and a.get("r_gap") is not None and float(a["r_gap"]) > 0.15:
        ins += (
            " The two scorers disagree on **how strong** that link is — skim **Reviews** before a leadership narrative."
        )
    return quick, data_line, ins


def stability_period_sentence(stab: pd.DataFrame | None) -> str:
    if stab is None or stab.empty or "rolling_corr_textblob" not in stab.columns:
        return ""
    d = stab.sort_values("date").dropna(subset=["rolling_corr_textblob"])
    if len(d) < 8:
        return ""
    d = d.assign(_r=pd.to_numeric(d["rolling_corr_textblob"], errors="coerce")).dropna(subset=["_r"])
    if len(d) < 8:
        return ""
    n = len(d)
    k = max(3, n // 4)
    early = float(d["_r"].iloc[:k].mean())
    late = float(d["_r"].iloc[-k:].mean())
    d0, d1 = d["date"].iloc[0], d["date"].iloc[-1]
    d0s = str(d0.date()) if hasattr(d0, "date") else str(d0)
    d1s = str(d1.date()) if hasattr(d1, "date") else str(d1)
    if abs(late - early) > 0.12:
        return (
            f"Rolling 20-day dictionary **r** moved from about **{early:.2f}** early on to **{late:.2f}** by **{d1s}** "
            f"— the link **shifted** between **{d0s}** and **{d1s}**."
        )
    return (
        f"Rolling 20-day dictionary **r** stayed near **{early:.2f}** early vs **{late:.2f}** late "
        f"from **{d0s}** through **{d1s}** — **no dramatic regime flip** in this sample."
    )


def build_market_stability_beats(
    company: str,
    res: "PipelineResult",
    facts: dict[str, Any],
    stab: pd.DataFrame | None,
) -> tuple[str, str, str]:
    quick = (
        "This answers: **Does the tone–return link stay steady as more trading days pile up, "
        "or does the last month tell a different story than the full sample?**"
    )
    data_line = (
        "**Expanding** correlation uses all overlapping days from the start through each date (min 15 after lag). "
        "**Rolling 20d** uses only the prior 20 trading days ending each date — it reacts faster to fresh weeks."
    )
    s = facts.get("stability") or {}
    per = stability_period_sentence(stab)
    if not s.get("available"):
        ins = f"For **{company}**, there aren’t enough overlapping days to plot stability paths yet."
    elif s.get("regime") == "recent_noisy":
        ins = (
            f"For **{company}**, **recent-window** correlation **bounces** compared with the long cumulative curve — "
            "treat the headline lag-1 **r** as **time-sensitive**."
        )
    elif s.get("regime") == "stable":
        ins = (
            f"For **{company}**, expanding and rolling dictionary paths **line up reasonably** at the end — "
            "the link looks **less like a one-week fluke** than in many alt-data pilots."
        )
    else:
        ins = (
            f"For **{company}**, stability is **mixed** — extend history or re-run after major releases for a cleaner read."
        )
    if per:
        ins += f" {per}"
    hl = (facts.get("holdout") or {}).get("lexicon") or {}
    if hl.get("available") and hl.get("verdict") == "weakens":
        ins += " Cross-check **Recent-week sanity check** below — it also stresses **newest** days."
    return quick, data_line, ins


# ── Theme cluster keywords ───────────────────────────────────────────────────
_SUPPORT_KWS = {"support", "respond", "response", "took", "forever", "wait",
                "waiting", "slow", "hours", "days", "reply", "ticket", "contacted"}
_UI_KWS = {"clicked", "click", "finally", "figured", "intuitive", "confusing",
           "confused", "navigation", "interface", "button", "screen", "menu",
           "difficult", "hard", "easy", "simple"}
_VALUE_KWS = {"worth", "price", "expensive", "cheap", "pay", "subscription",
              "premium", "free", "cost", "money", "refund", "cancel"}
_CRASH_KWS = {"crash", "bug", "freeze", "frozen", "error", "broken", "fix",
              "update", "glitch", "loading", "load", "stuck"}
_POSITIVE_KWS = {"love", "great", "amazing", "excellent", "perfect", "best",
                 "fantastic", "wonderful", "recommend", "brilliant"}


def _detect_theme_clusters(bigrams: list[dict]) -> list[str]:
    """Map bigram phrases to plain-English insight sentences."""
    all_words: set[str] = set()
    for b in bigrams:
        phrase = str(b.get("phrase", "")).lower()
        all_words.update(phrase.replace("-", " ").split())
    insights: list[str] = []
    if all_words & _SUPPORT_KWS:
        insights.append(
            "**Support response times** are a recurring theme — customers frequently mention waiting. "
            "Cross-reference with your support ticket data around this window."
        )
    if all_words & _UI_KWS:
        insights.append(
            "**App navigation or usability** comes up — some users describe a learning curve before things clicked. "
            "Consider whether onboarding copy or UI changes could reduce that friction."
        )
    if all_words & _VALUE_KWS:
        insights.append(
            "**Pricing or subscription value** is mentioned — worth checking whether recent price changes correlate with sentiment dips."
        )
    if all_words & _CRASH_KWS:
        insights.append(
            "**Stability or bug complaints** appear — cross-reference with your release notes and crash logs around these dates."
        )
    if not insights and all_words & _POSITIVE_KWS:
        insights.append(
            "**Positive language dominates** this corpus — customers describe satisfaction rather than frustration."
        )
    return insights


def themes_interpretation_markdown(company: str, tm: dict[str, Any] | None) -> str:
    if not tm or not (tm.get("bigrams") or tm.get("nmf_topics")):
        return ""
    bigs = tm.get("bigrams") or []
    clusters = _detect_theme_clusters(bigs) if bigs else []
    parts: list[str] = []
    if clusters:
        parts.append("**What customers are telling you:**")
        parts.append("")
        for c in clusters:
            parts.append(f"- {c}")
    topics = tm.get("nmf_topics") or []
    if topics:
        t0 = topics[0]
        tw = t0.get("top_words", "")
        if tw:
            parts.append(
                f"\nTop recurring words: **{tw}** — use these to brief your CS and product teams on storylines to investigate."
            )
    parts.append(
        "\n**Next step:** compare these themes to the **worst trading day** panel on Overview and the **Reviews** tab disagreement queue."
    )
    return "\n".join(parts)
def association_explanation_html(r: float | None) -> str:
    """HTML for signal box (no markdown ** inside parent HTML)."""
    if r is None:
        return (
            "<p>No usable link could be computed between yesterday’s average review score and next-day return "
            "for this window (often too little variation in scores).</p>"
        )
    a = abs(float(r))
    direction = "positive" if r > 0 else "negative"
    if a < 0.08:
        strength = "negligible"
    elif a < 0.2:
        strength = "weak"
    elif a < 0.35:
        strength = "modest"
    else:
        strength = "notable"
    return (
        f"<p><strong>{LABEL_LEXICON_SCORER}:</strong> The statistical link between <strong>yesterday’s</strong> "
        f"average review tone and <strong>next day’s</strong> return is <strong>{strength}</strong> and "
        f"<strong>{direction}</strong> in this sample (Pearson <em>r</em> = {float(r):.3f}, lag-1). "
        "That is an <strong>association</strong>, not proof that reviews caused the move.</p>"
    )


def build_technical_strip_overview(res: "PipelineResult") -> str:
    lines = [
        f"- **Reviews in export:** {res.stats.get('n_reviews', 0):,}",
        f"- **Overlap trading days:** {res.overlap_days}",
        f"- **Review date range (export):** {res.stats.get('date_min')} → {res.stats.get('date_max')}",
        f"- **Market overlap dates:** {res.date_overlap_min or '—'} → {res.date_overlap_max or '—'}",
        f"- **Lag-1 Pearson r ({TECH_LEXICON}):** {_fmt(res.corr_textblob_lag1)}",
        f"- **Lag-1 Pearson r ({TECH_NEURAL}):** {_fmt(res.corr_transformer_lag1)}",
    ]
    if res.transformer_error:
        lines.append(f"- **{TECH_NEURAL}:** skipped — {res.transformer_error}")
    return "\n".join(lines)


def build_technical_strip_trajectory(res: "PipelineResult", chart_stats: dict[str, Any]) -> str:
    lines = [
        f"- **Series:** daily mean `avg_textblob` ({TECH_LEXICON}), optional `avg_transformer` ({TECH_NEURAL})",
        f"- **Trading days in merge:** {chart_stats.get('n_trading_days', 0)}",
    ]
    for key, lab in (("lexicon", TECH_LEXICON), ("neural", TECH_NEURAL)):
        if f"{key}_mean" not in chart_stats:
            continue
        std = chart_stats.get(f"{key}_std")
        std_s = f"{float(std):.4f}" if std is not None and _finite(std) else "n/a"
        lines.append(
            f"- **{lab}:** mean {chart_stats[f'{key}_mean']:.4f}, std {std_s}, "
            f"first/last {chart_stats[f'{key}_first']:.4f} / {chart_stats[f'{key}_last']:.4f}"
        )
    lines.append(f"- **Lag definition:** {LAG_EXPLAINER.replace('**', '')}")
    return "\n".join(lines)


def _fmt(x: Any) -> str:
    if x is None:
        return "n/a"
    try:
        v = float(x)
    except (TypeError, ValueError):
        return "n/a"
    if not math.isfinite(v):
        return "n/a"
    return f"{v:.6f}"


def build_technical_strip_association(res: "PipelineResult") -> str:
    lines = [
        f"- **Lag-1 Pearson r · {TECH_LEXICON}:** {_fmt(res.corr_textblob_lag1)}",
        f"- **Lag-1 Pearson r · {TECH_NEURAL}:** {_fmt(res.corr_transformer_lag1)}",
        "- **Regression:** OLS of next-day `ret` on lag-1 average sentiment; **HC3** heteroskedasticity-robust SEs; full overlap window.",
    ]
    rf = res.regression_full_textblob
    if rf and all(_finite(rf.get(k)) for k in ("coef", "ci_low", "ci_high", "p_value", "r2", "n")):
        lines.append(
            f"- **{TECH_LEXICON} OLS:** slope {rf['coef']:.6f}, 95% CI [{rf['ci_low']:.6f}, {rf['ci_high']:.6f}], "
            f"p = {rf['p_value']:.4f}, R² = {rf['r2']:.4f}, n = {rf['n']} days"
        )
    rf2 = res.regression_full_transformer
    if rf2 and all(_finite(rf2.get(k)) for k in ("coef", "ci_low", "ci_high", "p_value", "r2", "n")):
        lines.append(
            f"- **{TECH_NEURAL} OLS:** slope {rf2['coef']:.6f}, 95% CI [{rf2['ci_low']:.6f}, {rf2['ci_high']:.6f}], "
            f"p = {rf2['p_value']:.4f}, R² = {rf2['r2']:.4f}, n = {rf2['n']} days"
        )
    return "\n".join(lines)


def build_technical_strip_stability(res: "PipelineResult", stab_stats: dict[str, Any]) -> str:
    lines = [
        "- **Expanding correlation:** uses all overlapping days from the start through each date (minimum 15 days after lag).",
        "- **Rolling 20d:** Pearson r on a 20-trading-day window ending each date.",
        f"- **{TECH_LEXICON} traces:** expanding / rolling as labeled in legend.",
        f"- **{TECH_NEURAL} traces:** same, when model scores exist.",
    ]
    if stab_stats.get("last_date"):
        lines.append(f"- **Last plotted date:** {stab_stats['last_date']}")
    for k, label in (
        ("expanding_corr_textblob", f"Last expanding r ({TECH_LEXICON})"),
        ("rolling_corr_textblob", f"Last rolling r ({TECH_LEXICON})"),
        ("expanding_corr_roberta", f"Last expanding r ({TECH_NEURAL})"),
        ("rolling_corr_roberta", f"Last rolling r ({TECH_NEURAL})"),
    ):
        if k in stab_stats:
            lines.append(f"- **{label}:** {stab_stats[k]:.4f}")
    return "\n".join(lines)


def build_technical_strip_holdout(res: "PipelineResult") -> str:
    lines: list[str] = []
    for label, key, tech in (
        (LABEL_LEXICON_SCORER, "holdout_textblob", TECH_LEXICON),
        (LABEL_NEURAL_SCORER, "holdout_transformer", TECH_NEURAL),
    ):
        h = getattr(res, key, None)
        if not isinstance(h, dict):
            continue
        lines.append(
            f"- **{tech} ({label}):** train n = {h.get('n_train')}, holdout n = {h.get('n_holdout')}, "
            f"through {h.get('holdout_last_date')}; slope p-value (HC3) = {_fmt(h.get('p_value_lag_sentiment'))}; "
            f"holdout corr(pred, actual) = {_fmt(h.get('corr_pred_vs_actual_holdout'))}; "
            f"RMSE = {_fmt(h.get('rmse_holdout'))}"
        )
    if not lines:
        lines.append("- Holdout not computed (insufficient history or variation in lag-1 sentiment).")
    return "\n".join(lines)


def build_technical_strip_reviews() -> str:
    return "\n".join(
        [
            f"- **Columns:** `disagreement` = |{TECH_LEXICON} − {TECH_NEURAL}|; scores in [`textblob`, `transformer`].",
            "- **Ranking:** highest disagreement first (most worth manual read).",
        ]
    )


def build_technical_strip_themes(tm: dict[str, Any] | None) -> str:
    if not tm:
        return "- No text-mining payload in this run."
    lines = [
        f"- **Flagged corpus n:** {tm.get('n_docs', 0)}",
        f"- **Source:** {tm.get('source_description', '—')}",
        "- **Bigrams:** token pairs (count) on flagged reviews.",
        "- **NMF topics:** latent Dirichlet-style topics (sklearn NMF); top words per topic.",
    ]
    return "\n".join(lines)


def build_technical_strip_brief_teaser() -> str:
    return (
        "- Full regression, stability, and holdout detail: see **Brief** tab → *Technical notes* expander, "
        "or the PDF export."
    )


def narrative_payload(
    company: str,
    res: "PipelineResult",
    *,
    insight_facts: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Single JSON object for optional executive LLM — numbers must match pipeline."""
    cs = chart_summary_stats(res.merged)
    ss = stability_summary_stats(res.stability_df)
    ifacts = insight_facts if insight_facts is not None else build_insight_facts(res)
    return {
        "company": company,
        "n_reviews": res.stats.get("n_reviews"),
        "overlap_trading_days": res.overlap_days,
        "corr_lexicon_lag1": res.corr_textblob_lag1,
        "corr_neural_lag1": res.corr_transformer_lag1,
        "chart_summary": cs,
        "stability_endpoints": ss,
        "worst_return_day": res.worst_return_day,
        "worst_return_value": res.worst_return_value,
        "worst_day_review_samples": worst_day_review_samples(res, k=3),
        "surface_low_bullets": build_surface_low_bullets(res, ifacts),
        "lexicon_label_business": LABEL_LEXICON_SCORER,
        "neural_label_business": LABEL_NEURAL_SCORER,
        "insight_facts": ifacts,
    }


def narrative_cache_key(payload: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:32]


def try_executive_narrative_markdown(payload: dict[str, Any]) -> str | None:
    """
    Optional LLM: 120–180 words, no new statistics, association ≠ causation.
    Returns None if no API key or failure.
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
        "You write a plain-English executive summary for ReviewSignal, a review-monitoring product. "
        "The reader is a non-technical VP or IR professional — they have 90 seconds. "
        "RULES: "
        "NEVER use: Pearson, r-value, OLS, HC3, lag-1, regression, correlation coefficient, p-value, TextBlob, RoBERTa. "
        "Instead say things like 'the stock tended to...' or 'when reviews were more positive...' or 'the pattern held / shifted'. "
        "Use ONLY facts from the JSON — company name, worst_return_day, bigram themes, mood direction, holdout verdict, overlap days. "
        "Do NOT invent numbers or statistics. "
        "If bigram themes are present in the JSON (look for text_mining or surface_low_bullets), "
        "synthesize what they mean: e.g. 'took forever' + 'support took' → 'customers frequently mention support wait times'. "
        "Output exactly three sections: "
        "1. **In plain English** — exactly 3 bullet points, one sentence each, spoken like a standup update: "
        "   Bullet 1: did customer mood go up, down, or sideways? "
        "   Bullet 2: did warmer reviews tend to line up with better stock days? How strong? "
        "   Bullet 3: did the pattern hold recently, or shift? Mention worst day if present in JSON. "
        "2. **One thing to do** — one sentence, concrete action mentioning dates or tabs. No jargon. "
        "3. End with exactly this italic line: _These observations come from app store reviews and market data only — not a prediction or investment advice._ "
        "Total: under 150 words. No title line. Output Markdown."
    )
    user = "Run facts (authoritative):\n```json\n" + json.dumps(payload, indent=2, default=str) + "\n```"
    try:
        r = client.chat.completions.create(
            model=os.environ.get("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.25,
        )
        return (r.choices[0].message.content or "").strip()
    except Exception:
        return None


def fallback_executive_narrative(payload: dict[str, Any]) -> str:
    """
    Plain-English 3-bullet talking-points card for non-technical readers.
    No jargon, no library names, no 'Pearson r' or 'OLS'.
    """
    co = str(payload.get("company", "this company"))
    d = payload.get("overlap_trading_days", 0)
    ifacts = payload.get("insight_facts") or {}
    traj = ifacts.get("trajectory") or {}
    assoc = ifacts.get("association") or {}
    stab = ifacts.get("stability") or {}

    direction = traj.get("lexicon_direction", "flat")
    direction_phrase = {
        "rose": "became more positive",
        "fell": "became more negative",
        "flat": "stayed roughly flat",
    }.get(direction, "held flat")

    r = payload.get("corr_lexicon_lag1")
    strength = assoc.get("strength_lexicon") or "weak"
    worst = payload.get("worst_return_day") or ""

    bullets: list[str] = []

    # Bullet 1: mood trend
    bullets.append(
        f"Over the {d}-day window, **customer reviews {direction_phrase}** for {co}."
    )

    # Bullet 2: link to stock moves
    if r is not None and _finite(r):
        rf = float(r)
        dir_word = "better" if rf > 0 else "worse"
        bullets.append(
            f"**On days when reviews were warmer than usual, the stock tended to do {dir_word} the next day** "
            f"— a {strength} but real pattern in this sample. This is context, not a trading signal."
        )
    else:
        bullets.append(
            "**No clear connection** was found between review tone and next-day stock moves "
            "in this window — that's still useful to know."
        )

    # Bullet 3: stability / worst day
    if stab.get("regime") == "recent_noisy":
        bullets.append(
            "**The pattern shifted in recent weeks** — the full-window average may be hiding a change. "
            "Treat recent data with extra care before sharing publicly."
        )
    elif worst:
        bullets.append(
            f"**The worst single trading day was {worst}.** "
            "Open the Reviews tab to see what customers were saying around that time."
        )
    else:
        bullets.append("The pattern was consistent across the full window — no obvious recent break.")

    fr = str((ifacts.get("focus_recommendation") or "").strip())
    # Strip any remaining technical jargon
    for jargon in ("lag-1", "HC3", "OLS", "Pearson", " r**", "r ≈"):
        fr = fr.replace(jargon, "")
    if not fr:
        fr = f"Re-export {co} data on your usual cadence and re-run — the next window will tell you if this pattern is holding."

    parts = ["**In plain English**", ""]
    for b in bullets:
        parts.append(f"- {b}")
    parts += ["", "**One thing to do**", "", fr, "",
              "_These observations come from app store reviews and market data only — not a prediction or investment advice._"]
    return "\n".join(parts)
