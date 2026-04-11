"""
Rule-based actionable insight lines from pipeline output (no extra ML).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline import PipelineResult


def actionable_insights(result: "PipelineResult") -> list[str]:
    """Short bullets for UI / PDF; conservative wording."""
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

    if result.worst_return_day and result.worst_return_value is not None:
        lines.append(
            f"**Drawdown focus:** Worst return **{result.worst_return_day}** (**{result.worst_return_value:.4f}**) — "
            "compare against **product changes**, **campaigns**, and **disagreement reviews** dated just before."
        )

    lines.append(
        "**Next step:** Export the **PDF report** for IR/product; re-run **weekly** on a fresh export with the same column mapping."
    )
    return lines
