# NLP App Reviews × Market Data — Group Project

Course group work: an **NLP-meets-finance** pipeline for organizations that have **large volumes of user text** (e.g. app store reviews) and **market or internal financial series**, and need a **repeatable** way to **score sentiment**, **quantify association with returns or risk**, and (in the full design) **turn results into a short executive brief** with an LLM.

**Remote:** [github.com/samerroz/nlp_app_reviews](https://github.com/samerroz/nlp_app_reviews)

**Fictional client story (for slides + demo narration):** [docs/CASE_STUDY.md](docs/CASE_STUDY.md) — **LinguaLoop Ltd.**, qualification, problem, insights, and “what they did next.”

**Course document mapping:** [docs/COURSE_DOCUMENT_OUTLINE.md](docs/COURSE_DOCUMENT_OUTLINE.md) — table of contents vs instructor PDF.

**Demo video script:** [docs/DEMO_VIDEO_SCRIPT.md](docs/DEMO_VIDEO_SCRIPT.md) — 3–5 minute screen recording shot list.

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
├── requirements.txt          # Core: pandas, numpy, textblob
├── requirements-ui.txt       # Streamlit + Plotly (product demo)
├── requirements-ml.txt       # Optional: torch + transformers (RoBERTa sentiment)
├── requirements-llm.txt      # Optional: OpenAI API for brief polish only
├── .gitignore                # Excludes large local data and legacy project copy
├── app/
│   └── streamlit_app.py      # ReviewSignal UI (upload → dashboard → brief)
├── docs/                     # Course brief, whiteboard, case study, doc outline, video script
├── sample_data/              # Synthetic CSVs: micro tutorial + extended demo (~90 days)
├── scripts/
│   └── generate_realistic_sample.py   # Regenerate reviews_demo.csv / market_demo.csv
└── src/
    ├── pipeline.py           # Importable engine (used by CLI + Streamlit)
    ├── demo_pipeline.py      # CLI: sample CSVs → print brief
    ├── insights.py           # Rule-based actionable insight bullets
    ├── report_pdf.py         # PDF export (fpdf2)
    ├── sentiment_models.py   # TextBlob + optional RoBERTa → [-1, 1]
    └── llm_brief.py          # Template brief; optional API polish (facts-only)
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

**Stronger NLP path (recommended for grading):** install transformer sentiment (first run downloads ~500MB model weights):

```bash
pip install -r requirements-ml.txt
# Optional: keep HF cache inside the repo (ignored by git)
export HF_HOME="$(pwd)/.hf_cache"
python src/demo_pipeline.py
```

**Optional LLM polish** of the same facts (never adds numbers): `pip install -r requirements-llm.txt`, set `OPENAI_API_KEY`, re-run the script.

---

## Product demo (Streamlit)

Install UI dependencies, then start the app from the **repository root** (the folder that contains `app/` and `sample_data/`):

```bash
pip install -r requirements.txt -r requirements-ui.txt
# optional: pip install -r requirements-ml.txt
streamlit run app/streamlit_app.py
```

In the sidebar: **Extended demo** (~90 trading days, ~1.1k synthetic reviews) → **Run analysis**. You get data health, dual-sentiment chart, lag-1 correlations, **actionable insight bullets**, disagreement queue, Markdown brief, and **Download PDF report** (plus CSV of the queue). Use **Micro tutorial** only for a quick UI smoke test.

Do **not** paste lines starting with `#` into the shell as commands (zsh will error); either omit comments or run each command separately.

**Why RoBERTa here (not random):** App reviews are **short and informal**; lexicon polarity misses **negation and sarcasm** often. We use `cardiffnlp/twitter-roberta-base-sentiment-latest` (social-style fine-tune), map 3-way probs to **[-1, 1]**, and feed **disagreement** (|TextBlob − RoBERTa|) into the **brief** so leadership **reads the right reviews first**. That ties **NLP depth** directly to the **LLM + finance** story.

---

## Full group pipeline (target architecture)

1. **Ingest** — Review export + daily financial series (returns, optionally volume).
2. **NLP** — **Dual** per-review sentiment: TextBlob **−1…+1** plus **RoBERTa** social sentiment (same scale); **daily** averages; **disagreement** flags reviews for human + LLM context.
3. **Quant** — Merge, lags, correlation / regression with **robust** errors where appropriate; pre-register hypotheses to avoid p-hacking stories.
4. **LLM** — Input = **your tables + a few retrieved review snippets**; output = **short report** for a human (optional RAG). **Numbers come from step 3**, not from the model guessing.

---

## Course fit: is this “NLP / text mining” enough?

The course weights **Application of NLP techniques (35%)** heavily. This repo’s **default** path is already **lexicon + transformer** (TextBlob + RoBERTa on social-style text) with **disagreement-aware** retrieval for the brief — that is a **credible** NLP story tied to **informal review language**, not a bolt-on.

**What already helps your grade**

- Clear **text→structure** pipeline (documents → scores → time series).
- **Transformer sentiment** with a **motivated** choice of checkpoint (short, noisy, user-generated text).
- **Disagreement** as a **semantic risk** signal (when to trust lexicon vs when to read the text).
- **LLM layer** that only **explains** precomputed stats and **cites retrieved review spans** (grounding).

**Further upgrades if you want “Excellent” depth**

- Add **one** of: **topic modeling** (NMF/LDA/BERTopic) or **aspect** buckets (bugs / billing / UX) on **flagged** days only.
- Optional: **named-entity** or **keyword** tags for product areas.

**Bottom line:** Name the **dual sentiment + disagreement → brief** path explicitly in **Architecture** and **Solution approach**; optional topics/aspects become your stretch goal.

---

## PoC demo: pretend we are **Company XYZ**

**Fictional client:** **XYZ Analytics** (or “DemoCo”) — a consumer app company with a CSV export of reviews and daily returns. They want a **repeatable internal report**, not a public website for end users.

**What the demo proves (in order)**

1. **Ingest** — They drop (or you pre-load) `reviews.csv` + `market.csv` matching the formats in `sample_data/` (or your real `Data/` locally for the live presentation).
2. **NLP** — Table or chart: **per-review scores** and **daily average sentiment** (−1…1) over time; optional second panel if you add topics/aspects.
3. **Quant** — One screen with **lagged correlation / regression summary** and a **plain-language** interpretation (“association, not causation”).
4. **Grounded brief** — Button **“Generate executive summary”**: LLM output that **only restates your numbers** and **quotes 3–5 short review excerpts** you pass in (no free-form inventing of statistics).

**What it should look like (recommended formats)**

| Format | Pros | Cons |
|--------|------|------|
| **Streamlit** (or **Gradio**) mini-app | Feels like a **product**; easy “upload → run → plots → summary”; great for **15′ presentation** | Small amount of UI work |
| **Jupyter notebook** walkthrough | Fast to build; very acceptable as **PoC** in many courses | Less “company demo” polish unless you hide cells and narrate cleanly |
| **CLI + exported PDF/HTML** | Minimal dependencies | Weaker theater for presentation |

**Recommendation:** Aim for **Streamlit** if one person can own it for a week; otherwise **notebook** with a **one-slide mock** of the “final product” UI. You do **not** need a mobile app or a deployed public website — a **local** or **Streamlit Cloud** demo is enough.

**What you say in 30 seconds**

> “XYZ uploads its **own** exports. Our pipeline **mines** the text into **daily sentiment**, **joins** market data, **quantifies** the link, and the **LLM writes a cited brief** for leadership — the same pattern we validated on our multi-company research sample.”

---

## Course document checklist

Use [docs/COURSE_DOCUMENT_OUTLINE.md](docs/COURSE_DOCUMENT_OUTLINE.md) so every bullet from [docs/group-work-information.pdf](docs/group-work-information.pdf) has a paragraph and, where noted, a figure (architecture, UI screenshot, Gantt, costs table).

---

## Team

Group project — replace with names and roles as you finalize.
