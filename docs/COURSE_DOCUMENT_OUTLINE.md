# Group document outline (maps to instructor PDF)

Use [group-work-information.pdf](group-work-information.pdf) as the master checklist. Below: **what to write** and **which figures** to drop in.

| PDF section | What to include (suggested) | Figures / artifacts |
|-------------|-----------------------------|---------------------|
| **Title** | Product + course, e.g. “ReviewSignal: NLP sentiment, market association, and grounded executive briefs for consumer-tech firms.” | — |
| **Introduction** | Problem: text volume + market data without repeatable scoring and triage; one-sentence solution (**ReviewSignal**). | Optional: **landing page** screenshot (start screen before **Run report**). |
| **Context, need and scope** | BYOD exports (reviews CSV + daily returns); no scraping in core product; fictional case [CASE_STUDY.md](CASE_STUDY.md). Scope: PoC dashboard + pipeline code. | Data flow diagram (upload → NLP → merge → stats → brief). |
| **Challenges** | Informal language, sarcasm; calendar alignment; association vs causation; small-sample instability in demo; LLM hallucination risk (mitigate with grounded brief). | — |
| **Objectives** | Use **O1–O4** table in [PRESENTATION_OUTLINE.md](PRESENTATION_OUTLINE.md): mood over time, market alignment (honest lag), prioritize human reading (disagreement), actionable outputs (brief/PDF/CSV/Q&A). | — |
| **Solution** | **Landing → Run report → six tabs** (Overview · Market & stats · Reviews · Themes · Brief · Q&A); pipeline in plain language. | Link to GitHub repo + [PRESENTATION_OUTLINE.md](PRESENTATION_OUTLINE.md). |
| **Solution approach** | Agile-ish: baseline TextBlob → add RoBERTa → add UI → optional OpenAI polish. | — |
| **Architecture** | Layers: UI (Streamlit), engine ([pipeline.py](../src/pipeline.py)), sentiment ([sentiment_models.py](../src/sentiment_models.py)), quant ([time_series_eval.py](../src/time_series_eval.py)), text mining ([text_mining.py](../src/text_mining.py)), CSV infer ([csv_infer.py](../src/csv_infer.py)), brief ([llm_brief.py](../src/llm_brief.py)), chat ([llm_chat.py](../src/llm_chat.py)). | Same as teacher whiteboard + app screenshot. |
| **How the solution solves the problem** | Shared daily signal; dual scorer for triage; association for IR/product prioritization; brief for leadership. | Before/after narrative (LinguaLoop). |
| **Expected results** | Demo: correlations, **OLS+CI**, stability chart, holdout metrics, **text-mining table**, disagreement queue, brief, **chat**, PDF; research: cite long-window work offline if applicable. | Table of metrics + screenshot of mining + chat. |
| **Project approach** | Iterations: document problem → implement engine → UI → rehearse presentation. | — |
| **MVP or PoC** | `streamlit run app/streamlit_app.py` + `python src/demo_pipeline.py`; built-in demo on **landing**; **Run report** → six-tab dashboard; **PDF** (2 pages); large synthetic CSVs via `scripts/generate_realistic_sample.py` → `.cache/`. | Landing screenshot + **Overview** or **Market & stats** tab + PDF sample + limitation statement. |
| **Project stages** | e.g. (1) Requirements (2) NLP engine (3) Quant merge (4) UI (5) Brief/LLM (6) Doc & deck. | Gantt (next section). |
| **Tasks definition** | Split: NLP, frontend, stats, writing, slides, recording. | Task list with owners. |
| **Team and estimated work** | Hours per person × weeks (honest student estimates). | — |
| **Planning (Gantt chart)** | 6–8 weeks bars: overlap allowed; milestone “teacher validation,” “MVP freeze,” “PDF slides due.” | Gantt image (Sheets, Notion, or Mermaid export). |
| **Team** | Names, emails. | — |
| **Roles** | PM, NLP lead, UI, quant, writer, presenter. | — |
| **Costs (human and tech)** | Human: hours × rough hourly rate or “student time.” Tech: OpenAI API (optional), HF free tier, laptops, $0 hosting if local demo. | Simple table. |

## Figures to generate once

1. **Architecture** — boxes: Reviews CSV, Market CSV, TextBlob, RoBERTa, Daily merge, OLS+HC3, rolling/holdout, Text mining (flagged), Disagreement ranker, Brief, Chat (grounded). **UI:** Landing → Run report → tabs **Overview | Market & stats | Reviews | Themes | Brief | Q&A**.  
2. **UI screenshots** — (a) **Landing** with company name + **Run report**; (b) dashboard: **Market & stats** (regression) + **Reviews** (disagreement) or **Q&A** (chat).  
3. **Gantt** — export as PNG for PDF.  
4. **Optional** — regression table from long offline study (Duolingo etc.) as “validation appendix.”

## Rubric crosswalk (reminder)

- **NLP 35%:** Dual sentiment + disagreement + **flagged-corpus** ngrams/topics + informal-text motivation.  
- **Innovation 35%:** Grounded LLM **brief + chat**; product framing.  
- **Presentation 20%:** Live or recorded UI walkthrough ([DEMO_VIDEO_SCRIPT.md](DEMO_VIDEO_SCRIPT.md)).  
- **Topic 10%:** Clear corporate / alternative-data relevance.
