# Group document outline (maps to instructor PDF)

Use [group-work-information.pdf](group-work-information.pdf) as the master checklist. Below: **what to write** and **which figures** to drop in.

| PDF section | What to include (suggested) | Figures / artifacts |
|-------------|-----------------------------|---------------------|
| **Title** | Product + course, e.g. “ReviewSignal: NLP sentiment, market association, and grounded executive briefs for consumer-tech firms.” | — |
| **Introduction** | Problem: text volume + market data without repeatable scoring and triage; one-sentence solution. | Optional: hero screenshot of Streamlit. |
| **Context, need and scope** | BYOD exports (reviews CSV + daily returns); no scraping in core product; fictional case [CASE_STUDY.md](CASE_STUDY.md). Scope: PoC dashboard + pipeline code. | Data flow diagram (upload → NLP → merge → stats → brief). |
| **Challenges** | Informal language, sarcasm; calendar alignment; association vs causation; small-sample instability in demo; LLM hallucination risk (mitigate with grounded brief). | — |
| **Objectives** | Measurable: dual sentiment, daily merge, lag-1 correlation, disagreement queue, Markdown brief; stretch: topics/aspects. | — |
| **Solution** | ReviewSignal / pipeline description in plain language. | Link to GitHub repo. |
| **Solution approach** | Agile-ish: baseline TextBlob → add RoBERTa → add UI → optional OpenAI polish. | — |
| **Architecture** | Layers: UI (Streamlit), engine ([pipeline.py](../src/pipeline.py)), models ([sentiment_models.py](../src/sentiment_models.py)), brief ([llm_brief.py](../src/llm_brief.py)). | Same as teacher whiteboard + app screenshot. |
| **How the solution solves the problem** | Shared daily signal; dual scorer for triage; association for IR/product prioritization; brief for leadership. | Before/after narrative (LinguaLoop). |
| **Expected results** | Demo: correlations + disagreement table + brief on `sample_data/`; research: cite multi-firm / long-window work offline (Big Data folder, not in repo). | Table of metrics from one run. |
| **Project approach** | Iterations: document problem → implement engine → UI → rehearse presentation. | — |
| **MVP or PoC** | `streamlit run app/streamlit_app.py` + `python src/demo_pipeline.py`; **Extended demo** dataset; **PDF export** + insight bullets. | Screenshot + PDF sample + limitation statement. |
| **Project stages** | e.g. (1) Requirements (2) NLP engine (3) Quant merge (4) UI (5) Brief/LLM (6) Doc & deck. | Gantt (next section). |
| **Tasks definition** | Split: NLP, frontend, stats, writing, slides, recording. | Task list with owners. |
| **Team and estimated work** | Hours per person × weeks (honest student estimates). | — |
| **Planning (Gantt chart)** | 6–8 weeks bars: overlap allowed; milestone “teacher validation,” “MVP freeze,” “PDF slides due.” | Gantt image (Sheets, Notion, or Mermaid export). |
| **Team** | Names, emails. | — |
| **Roles** | PM, NLP lead, UI, quant, writer, presenter. | — |
| **Costs (human and tech)** | Human: hours × rough hourly rate or “student time.” Tech: OpenAI API (optional), HF free tier, laptops, $0 hosting if local demo. | Simple table. |

## Figures to generate once

1. **Architecture** — boxes: Reviews CSV, Market CSV, TextBlob, RoBERTa, Daily merge, Correlation, Disagreement ranker, Brief (template / API).  
2. **UI screenshot** — Streamlit after “Run analysis” on sample data.  
3. **Gantt** — export as PNG for PDF.  
4. **Optional** — regression table from long offline study (Duolingo etc.) as “validation appendix.”

## Rubric crosswalk (reminder)

- **NLP 35%:** Dual sentiment + disagreement + informal-text motivation.  
- **Innovation 35%:** Grounded LLM brief; product framing.  
- **Presentation 20%:** Live or recorded UI walkthrough ([DEMO_VIDEO_SCRIPT.md](DEMO_VIDEO_SCRIPT.md)).  
- **Topic 10%:** Clear corporate / alternative-data relevance.
