# ReviewSignal — NLP app reviews × market data (group project)

Course group work: **ReviewSignal**, an **NLP-meets-finance** proof of concept for teams that have **large volumes of user text** (e.g. app store reviews) and **daily market returns**, and need a **repeatable** way to **measure mood over time**, **check honest alignment with returns** (association, not prediction), **prioritize which reviews humans read first** (especially when scorers disagree), and **ship an evidence pack** — insights, **executive brief**, **PDF/CSV**, and **grounded Q&A**.

**Remote:** [github.com/samerroz/nlp_app_reviews](https://github.com/samerroz/nlp_app_reviews)

**Presentation deck (plain text, copy to Word/Slides):** [docs/PRESENTATION_OUTLINE.md](docs/PRESENTATION_OUTLINE.md) — title through objectives, solution, demo flow, limitations, FAQ.

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

That is **correlation / association** with careful wording (not claiming causality). The **LLM** layer **explains and phrases** what the pipeline already computed — it does **not** invent statistics; **grounded chat** uses the same idea for Q&A.

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
├── requirements.txt          # Core: pandas, numpy, textblob, statsmodels, scikit-learn
├── requirements-ui.txt       # Streamlit + Plotly (product demo)
├── requirements-ml.txt       # Optional: torch + transformers (RoBERTa sentiment)
├── requirements-llm.txt      # Optional: OpenAI API for brief polish only
├── .gitignore                # Excludes large local data and legacy project copy
├── app/
│   └── streamlit_app.py      # ReviewSignal UI (landing → Run report → six-tab dashboard)
├── docs/                     # Course brief, whiteboard, case study, doc outline, video script
├── sample_data/              # Synthetic CSVs: micro tutorial + extended demo (~90 days)
├── scripts/
│   └── generate_realistic_sample.py   # Synthetic CSVs; `--days`, `--reviews-min/max`, `--out-*`
└── src/
    ├── pipeline.py           # Importable engine (used by CLI + Streamlit)
    ├── demo_pipeline.py      # CLI: sample CSVs → print brief
    ├── insights.py           # Rule-based actionable insight bullets
    ├── report_pdf.py         # PDF export (fpdf2); page 2 = regression + text mining
    ├── sentiment_models.py   # TextBlob + optional RoBERTa → [-1, 1]
    ├── time_series_eval.py   # Expanding/rolling r; holdout OLS; full-sample HC3 OLS + CI
    ├── text_mining.py        # Bigrams + NMF on disagreement / extreme-return reviews only
    ├── csv_infer.py          # Heuristic column detection for messy exports
    ├── llm_chat.py           # Grounded chat context JSON + optional OpenAI
    └── llm_brief.py          # Template brief; optional API polish (facts-only)
```

**Not in Git (see `.gitignore`):**

- `Big Data in Finance/` — full previous-semester tree; keep on disk for reference, do not push.
- `Data/` — large review + market CSVs (~200MB+); team members download or share separately.
- `.cache/` — optional **large synthetic** demo CSVs from `scripts/generate_realistic_sample.py` (gitignored).

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

**Optional LLM** — same facts only: `pip install -r requirements-llm.txt`, set `OPENAI_API_KEY` for **brief polish** (CLI) or **grounded chat** in Streamlit (stub replies if unset).

---

## Product demo (Streamlit — ReviewSignal)

Install UI dependencies, then start the app from the **repository root** (the folder that contains `app/` and `sample_data/`):

```bash
pip install -r requirements.txt -r requirements-ui.txt
# optional: pip install -r requirements-ml.txt
streamlit run app/streamlit_app.py
```

**Dev note:** `.streamlit/config.toml` sets `fileWatcherType = "none"` so Streamlit does not walk every `transformers` vision submodule (which would otherwise spam `torchvision` import errors in the terminal). Use **Rerun** in the app after code edits, or temporarily set `fileWatcherType = "auto"` if you want live reload.

**Flow:** The app opens on a **landing page** (full-width layout, presentation-style palette). The **LinguaLoop built-in demo** (~90 trading days) **loads automatically**. Enter or keep the **company name**, adjust options if needed, then click **Run report** (no CSV upload required for class). Upload **both** CSVs on the landing page when you want **your own exports**. The app **guesses columns** (override under **Advanced** if needed). **RoBERTa** defaults to **off**; turn it on after `pip install -r requirements-ml.txt`.

**After a successful run** you get a **six-tab dashboard**: **Overview** (data health, downloads, plain-language bullets), **Market & stats** (dual sentiment, Pearson + **OLS (HC3) + CI**, stability, holdout), **Reviews** (disagreement queue), **Themes** (bigrams + NMF on flagged text), **Brief** (markdown + optional API polish + technical notes), **Q&A** (grounded chat). Use **Back to start** to return to the landing page. **PDF** (page 2 = quant + mining) and **CSV** exports live on **Overview**. To stress-test scale locally, run `python scripts/generate_realistic_sample.py --days 500 ...` (writes to `.cache/` if you choose).

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

- **Aspect** buckets (bugs / billing / UX) or **BERTopic** on flagged days — the repo already ships **bigrams + NMF** on the disagreement / tail-return corpus.
- Optional: **named-entity** tags for product areas.

**Bottom line:** Name the **dual sentiment + disagreement → quant + flagged text mining → brief / grounded Q&A** path explicitly in **Architecture** and **Solution approach**. Objectives O1–O4 in [docs/PRESENTATION_OUTLINE.md](docs/PRESENTATION_OUTLINE.md) map directly to tabs and exports.

---

## PoC demo: pretend we are **Company XYZ**

**Fictional client:** **XYZ Analytics** (or “DemoCo”) — a consumer app company with a CSV export of reviews and daily returns. They want a **repeatable internal report**, not a public website for end users.

**What the demo proves (in order)**

1. **Ingest** — They drop (or you pre-load) `reviews.csv` + `market.csv` matching the formats in `sample_data/` (or your real `Data/` locally for the live presentation).
2. **NLP** — Table or chart: **per-review scores** and **daily average sentiment** (−1…1) over time; optional second panel if you add topics/aspects.
3. **Quant** — One screen with **lagged correlation / regression summary** and a **plain-language** interpretation (“association, not causation”).
4. **Grounded brief** — Produced on each **Run report** (optional **polish** toggle if an API key is set): wording may be refined, but **facts come from the pipeline**, not from free invention of statistics. **Q&A** uses the same **grounded** rule set.

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

---

## Evaluation methodology (for the course report)

**No random train/test split** on days (that would leak future information). We use:

1. **Expanding-window correlation:** at each date, Pearson *r* between lag-1 daily sentiment and same-day return using **all history from the start** through that date (after a minimum number of points).
2. **Rolling-window correlation (20 trading days):** same *r* but using **only the last 20 days** ending at each date — highlights **short-run instability**.
3. **Pseudo-forecasting / chronological holdout:** fit `return ~ lag-1_sentiment + const` with **OLS and HC3 robust standard errors** on the **early** segment; hold out the **last ~20%** of days (min 8) **in calendar order**. Report train **p-value** on the slope, and on the holdout **corr(predicted, actual)** and **RMSE**. This is a **sanity check**, not a trading backtest.

4. **Full-sample OLS (HC3)** on the same lagged panel reports **slope, 95% Wald CI, p-value, R²** for finance-style language alongside Pearson *r*.

5. **Text mining** (bigrams + small **NMF**) runs only on a **flagged** subset: high **disagreement** reviews plus text on **extreme return** days — not global LDA on all reviews.

6. **Grounded LLM chat** in Streamlit embeds a **frozen JSON** snapshot of the run; the model is instructed **not** to invent statistics (optional `OPENAI_API_KEY`).

**Large local CSVs:** Place files such as `sample_data/local_reviews.csv` and `sample_data/local_market.csv` on your machine — paths are **gitignored**; use **Upload** on the **landing page** for class demos without committing proprietary data. Optional **large synthetic** pairs live under **`.cache/`** (also gitignored) via `scripts/generate_realistic_sample.py`.
