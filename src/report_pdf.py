"""
Build a simple one-column PDF report (executive-style) from pipeline output.
Uses fpdf2; review snippets are ASCII-sanitized for built-in fonts.
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from pipeline import PipelineResult


def _ascii_safe(s: str, max_len: int = 400) -> str:
    """FPDF built-in Helvetica only supports Latin-1 subset; strip/replace Unicode."""
    t = s.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    t = t.replace("\u2014", "-").replace("\u2013", "-").replace("\u2212", "-")
    t = t.replace("\u2026", "...").replace("\u00d7", "x").replace("\u22c5", "*")
    t = t.replace("\u03b2", "beta").replace("\u00b2", "2").replace("\u00b3", "3")
    t = re.sub(r"[^\x20-\x7E\n]+", " ", t)
    out = (t[:max_len] + ("..." if len(t) > max_len else "")).strip()
    return out if out else " "


def _multicell(
    pdf: Any,
    h: float,
    text: str,
    *,
    max_text_len: int = 12000,
) -> None:
    """Full-width paragraph; reset x after write (fpdf2 default new_x=RIGHT breaks w=0 on next call)."""
    from fpdf.enums import WrapMode, XPos, YPos

    pdf.set_x(pdf.l_margin)
    pdf.multi_cell(
        w=pdf.epw,
        h=h,
        text=_ascii_safe(text, max_text_len),
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
        wrapmode=WrapMode.CHAR,
    )


def build_pdf_bytes(
    company_name: str,
    result: "PipelineResult",
    insight_lines: list[str],
) -> bytes:
    from fpdf import FPDF
    from fpdf.enums import XPos, YPos

    from pipeline import dedupe_disagreement_quotes

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.set_x(pdf.l_margin)
    pdf.cell(
        w=pdf.epw,
        h=10,
        text=_ascii_safe(f"ReviewSignal report - {company_name}", 80),
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )
    pdf.ln(2)
    brief_plain = re.sub(r"\*\*([^*]+)\*\*", r"\1", result.brief_markdown)
    brief_plain = re.sub(r"^#+\s*", "", brief_plain, flags=re.MULTILINE)
    pdf.set_font("Helvetica", "", 10)
    _multicell(pdf, 5, brief_plain, max_text_len=12000)
    pdf.ln(2)
    pdf.set_font("Helvetica", "B", 12)
    pdf.set_x(pdf.l_margin)
    pdf.cell(
        w=pdf.epw,
        h=8,
        text="Actionable insights",
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )
    pdf.set_font("Helvetica", "", 10)
    for line in insight_lines:
        clean = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
        _multicell(pdf, 5, "- " + clean, max_text_len=500)

    dq_pdf = dedupe_disagreement_quotes(result.disagreement_quotes)
    if dq_pdf:
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_x(pdf.l_margin)
        pdf.cell(
            w=pdf.epw,
            h=8,
            text="Top disagreement reviews (excerpt)",
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )
        pdf.set_font("Helvetica", "", 9)
        for i, q in enumerate(dq_pdf[:8], 1):
            pdf.set_font("Helvetica", "B", 9)
            pdf.set_x(pdf.l_margin)
            pdf.cell(
                w=pdf.epw,
                h=5,
                text=_ascii_safe(
                    f"{i}. {q.get('date', '')} | dis={float(q.get('disagreement', 0)):.3f}",
                    120,
                ),
                new_x=XPos.LMARGIN,
                new_y=YPos.NEXT,
            )
            pdf.set_font("Helvetica", "", 9)
            _multicell(pdf, 4, str(q.get("text", "")), max_text_len=320)

    pdf.add_page()
    pdf.set_font("Helvetica", "B", 14)
    pdf.set_x(pdf.l_margin)
    pdf.cell(
        w=pdf.epw,
        h=10,
        text=_ascii_safe("Quant + text mining (same run)", 60),
        new_x=XPos.LMARGIN,
        new_y=YPos.NEXT,
    )
    pdf.set_font("Helvetica", "", 9)
    _multicell(
        pdf,
        5,
        "Rolling/expanding correlation and chronological holdout OLS are time-honest checks "
        "(no random split). Full-sample OLS below uses HC3 robust SE; association is not causal.",
        max_text_len=2000,
    )
    pdf.ln(1)

    rf = result.regression_full_textblob
    if rf:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_x(pdf.l_margin)
        pdf.cell(
            w=pdf.epw,
            h=6,
            text=_ascii_safe("Full-sample OLS - TextBlob lag-1", 120),
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )
        pdf.set_font("Helvetica", "", 9)
        _multicell(
            pdf,
            5,
            f"n={rf['n']} coef={rf['coef']:.6f} 95pct CI [{rf['ci_low']:.6f}, {rf['ci_high']:.6f}] "
            f"p={rf['p_value']:.4f} R2={rf['r2']:.4f}",
            max_text_len=500,
        )
    rf2 = result.regression_full_transformer
    if rf2:
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_x(pdf.l_margin)
        pdf.cell(
            w=pdf.epw,
            h=6,
            text=_ascii_safe("Full-sample OLS - RoBERTa lag-1", 120),
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )
        pdf.set_font("Helvetica", "", 9)
        _multicell(
            pdf,
            5,
            f"n={rf2['n']} coef={rf2['coef']:.6f} 95pct CI [{rf2['ci_low']:.6f}, {rf2['ci_high']:.6f}] "
            f"p={rf2['p_value']:.4f} R2={rf2['r2']:.4f}",
            max_text_len=500,
        )
    tm = result.text_mining
    if tm and (tm.get("bigrams") or tm.get("nmf_topics")):
        pdf.set_font("Helvetica", "B", 11)
        pdf.set_x(pdf.l_margin)
        pdf.cell(
            w=pdf.epw,
            h=6,
            text="Text mining (flagged reviews only)",
            new_x=XPos.LMARGIN,
            new_y=YPos.NEXT,
        )
        pdf.set_font("Helvetica", "", 8)
        _multicell(pdf, 4, tm.get("source_description", ""), max_text_len=400)
        if tm.get("bigrams"):
            bg = "; ".join(f"{b['phrase']} x{b['count']}" for b in tm["bigrams"][:12])
            _multicell(pdf, 4, f"Top bigrams: {bg}", max_text_len=1200)
        if tm.get("nmf_topics"):
            for top in tm["nmf_topics"][:5]:
                _multicell(
                    pdf,
                    4,
                    f"Topic {top.get('id')}: {top.get('top_words', '')}",
                    max_text_len=500,
                )

    out = pdf.output(dest="S")
    if isinstance(out, (bytes, bytearray)):
        return bytes(out)
    return out.encode("latin-1", errors="replace")
