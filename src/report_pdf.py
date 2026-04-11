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
    t = s.replace("\u2019", "'").replace("\u201c", '"').replace("\u201d", '"')
    t = t.replace("\u2014", "-").replace("\u2212", "-")
    t = re.sub(r"[^\x20-\x7E\n]+", " ", t)
    return t[:max_len] + ("..." if len(t) > max_len else "")


def build_pdf_bytes(
    company_name: str,
    result: "PipelineResult",
    insight_lines: list[str],
) -> bytes:
    from fpdf import FPDF

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, _ascii_safe(f"ReviewSignal report — {company_name}", 80), ln=True)
    pdf.ln(2)
    brief_plain = re.sub(r"\*\*([^*]+)\*\*", r"\1", result.brief_markdown)
    brief_plain = re.sub(r"^#+\s*", "", brief_plain, flags=re.MULTILINE)
    pdf.set_font("Helvetica", "", 10)
    pdf.multi_cell(0, 5, _ascii_safe(brief_plain, 12000))
    pdf.ln(4)
    pdf.set_font("Helvetica", "B", 12)
    pdf.cell(0, 8, "Actionable insights", ln=True)
    pdf.set_font("Helvetica", "", 10)
    for line in insight_lines:
        clean = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)
        pdf.multi_cell(0, 5, "- " + _ascii_safe(clean, 500))
        pdf.ln(1)

    if result.disagreement_quotes:
        pdf.ln(2)
        pdf.set_font("Helvetica", "B", 12)
        pdf.cell(0, 8, "Top disagreement reviews (excerpt)", ln=True)
        pdf.set_font("Helvetica", "", 9)
        for i, q in enumerate(result.disagreement_quotes[:8], 1):
            pdf.set_font("Helvetica", "B", 9)
            pdf.cell(0, 5, f"{i}. {_ascii_safe(q.get('date', ''), 20)} | dis={q.get('disagreement', 0):.3f}", ln=True)
            pdf.set_font("Helvetica", "", 9)
            pdf.multi_cell(0, 4, _ascii_safe(str(q.get("text", "")), 320))
            pdf.ln(1)

    out = pdf.output(dest="S")
    if isinstance(out, (bytes, bytearray)):
        return bytes(out)
    return out.encode("latin-1", errors="replace")
