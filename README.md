# NLP App Reviews × Market Data — Group Project

Course group work: an **NLP-meets-finance** pipeline for organizations that have **large volumes of user text** (e.g. app store reviews) and **market or internal financial series**, and need a **repeatable** way to **score sentiment**, **quantify association with returns or risk**, and (in the full design) **turn results into a short executive brief** with an LLM.

**Remote:** [github.com/samerroz/nlp_app_reviews](https://github.com/samerroz/nlp_app_reviews)

---

## Who would use this, and what are they trying to find out?

**Concrete personas**

| Role | Company type | What they want to learn |
|------|----------------|-------------------------|
| **Investor relations / Corp Dev** at a **consumer tech** or **subscription** company | Public or pre-IPO app business | Whether **waves of user opinion** line up with **stock moves** or volatility — not to “day trade,” but to **anticipate narrative risk**, **earnings prep**, and **messaging**. |
| **Product leadership + FP&A** | Same | Whether **product incidents or praise** in reviews show up in **retention-sensitive metrics** they already track; bridges **qualitative noise** to **quantified patterns**. |
| **Data / analytics team** | Any firm with review exports + market or revenue data | A **documented pipeline**: ingest → score → aggregate → merge → stats → (optional) LLM summary — so results are **auditable** and **reproducible**, not one-off spreadsheets. |

**Example question they might ask**

> “When daily average review sentiment drops sharply after a release, do we tend to see **worse next-day returns** or **higher realized volatility** more often than chance? If so, how strong is the link, and what did users actually say on those days?”

That is **correlation / predictive association** with careful wording (not claiming causality). The **LLM** layer answers the last part: **summarize themes** from retrieved reviews **given** numbers your pipeline already computed.

---

## Bring your own data (recommended story for the business case)

- **Production-style framing:** The customer **exports** what they are allowed to use: review dumps (CSV/JSON), plus **their** returns, revenue, or risk series. **No scraping** is required for the core product — scraping store pages is **fragile, often against ToS**, and hard to defend in a “company project” pitch.
- **Demo / PoC:** This repo includes **tiny synthetic CSVs** under `sample_data/` so anyone can run the pipeline **without** publishing real user data or large files on GitHub.
- **Your full historical analysis** (millions of reviews, CRSP-style market data) stays **local** in the ignored `Data/` folder or your own drive; the report can say “validated on proprietary/large sample; public repo ships samples only.”

---

## “Cost management” on the whiteboard (simple way to say it)

Treat it as **who consumes the output**: **finance and operations** care about **cost of capital, spend efficiency, and downside risk**. The pipeline does not need to “optimize costs” automatically; it **surfaces** when **sentiment deterioration** coincides with **market stress**, so leaders can **align PR, product fixes, and budget** — one sentence in the slide deck is enough.

---

## Repository layout

```
├── README.md                 # This file
├── requirements.txt          # Minimal deps for the sample demo
├── .gitignore                # Excludes large local data and legacy project copy
├── docs/                     # Course brief + architecture photo
├── sample_data/              # Small synthetic CSVs (safe to commit)
└── src/
    └── demo_pipeline.py      # End-to-end toy run on samples
```

**Not in Git (see `.gitignore`):**

- `Big Data in Finance/` — full previous-semester tree; keep on disk for reference, do not push.
- `Data/` — large review + market CSVs (~200MB+); team members download or share separately.

---

## Quick start (sample demo)

```bash
cd "NLP Project"   # or your clone path
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/demo_pipeline.py
```

---

## Full group pipeline (target architecture)

1. **Ingest** — Review export + daily financial series (returns, optionally volume).
2. **NLP** — Per-review polarity (e.g. TextBlob **−1…+1**) or stronger models for the course rubric; aggregate **daily** (or weekly) per entity.
3. **Quant** — Merge, lags, correlation / regression with **robust** errors where appropriate; pre-register hypotheses to avoid p-hacking stories.
4. **LLM** — Input = **your tables + a few retrieved review snippets**; output = **short report** for a human (optional RAG). **Numbers come from step 3**, not from the model guessing.

---

## Course document checklist

Align the written deliverable with the instructor PDF in `docs/` (title, introduction, context, challenges, objectives, solution, architecture, MVP, Gantt, roles, costs, etc.).

---

## Team

Group project — replace with names and roles as you finalize.
