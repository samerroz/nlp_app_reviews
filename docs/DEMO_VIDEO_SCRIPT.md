# Demo video script (3–5 minutes)

**Goal:** Show the **product** (Streamlit **ReviewSignal**), not the terminal. Use **built-in sample data** unless you have permission to blur a real export.

**Recording:** 1080p, hide file paths and API keys. Narrate in **complete sentences**; cut dead air in edit.

**Current UI:** **Landing page** first → **Run report** → **six-tab** dashboard (**Back to start** returns to landing). There is no data-entry sidebar; the sidebar stays collapsed.

---

## Shot list

| Time (approx) | Visual | Say (guide, not word-for-word) |
|---------------|--------|--------------------------------|
| 0:00–0:25 | Title slide or app on **landing** | “We built **ReviewSignal** for teams that have **app review exports** and **daily returns** but no repeatable way to score text, check overlap with the market, and brief leadership.” |
| 0:25–0:55 | **Landing**: company name, demo status, uploads | “Users **bring their own CSVs** in production — we are not scraping stores in the core story. For the video the **LinguaLoop demo is already loaded** (~90 trading days). I set the **company name** and click **Run report**. Uploads on this page replace the demo when you have **both** files.” |
| 0:55–1:15 | **Advanced** expander on landing | “The app **guesses column names** for messy exports; you can override here if a field is wrong.” |
| 1:15–1:35 | Options: RoBERTa, queue size, polish | “**RoBERTa** adds a second sentiment opinion for **short informal** text; you can turn it off if ML is not installed. **Polish** optionally rephrases the brief — same facts.” |
| 1:35–1:55 | **Run report** + spinner | “Run scores every review with **TextBlob** and optionally **RoBERTa**, averages by **day**, merges with **returns**, and uses a **one-day lag** so we are not cheating with same-day noise.” |
| 1:55–2:20 | **Overview** tab | “**Overview** is data health: counts, **review span**, **overlap** with market days, **PDF** and **CSV** downloads, and **plain-language bullets** — what this run means for non-technical readers.” |
| 2:20–2:45 | **Market & stats** — dual sentiment chart | “Two lines: lexicon vs neural. When they diverge, **simple scores lie** — that is why **Reviews** has a **disagreement queue**.” |
| 2:45–3:15 | **Pearson + full-sample OLS (HC3)** | “We show **Pearson *r*** and **OLS with robust standard errors** — **slope, 95% confidence interval, p-value** — so finance stakeholders see **effect size and noise**, not one headline correlation.” |
| 3:15–3:35 | **Correlation stability** + **holdout** | “**No random train/test** on days — we use **expanding and rolling** correlation paths and a **chronological holdout** as **pseudo-forecasting** sanity checks.” |
| 3:35–3:50 | **Reviews** tab — disagreement table | “Rows ranked by **absolute gap** between scorers — **read these first**.” |
| 3:50–4:05 | **Themes** tab | “**Bigrams and small NMF** run only on **flagged** text — high **disagreement** and **extreme return** days.” |
| 4:05–4:25 | **Brief** tab | “Narrative brief, optional polished wording, and **technical notes** for analysts.” |
| 4:25–4:45 | **Q&A** tab | “**Chat** uses a **frozen JSON** snapshot of this run — **do not invent numbers**. Without an API key you see a **stub**; with a key, ad-hoc **grounded** answers.” |
| 4:45–5:00 | **Overview** — download PDF (optional) | “**PDF** adds a **second page** with regression and text mining for a **board pack**.” |
| 5:00–5:15 | **Back to start** or closing | “**LinguaLoop**-style story in our case study doc. Limitations: toy data, **no** trading costs, **association ≠ causation**.” |

---

## Checklist before you hit record

- [ ] `pip install -r requirements.txt -r requirements-ui.txt` and optionally `-r requirements-ml.txt` and `-r requirements-llm.txt`  
- [ ] `streamlit run app/streamlit_app.py` — dry run once (**landing** → **Run report** → skim all six tabs)  
- [ ] Browser zoom ~100%, window wide enough for charts and metrics  
- [ ] `OPENAI_API_KEY` unset **or** skip chat / keep answers generic if you do not want to show API content  

---

## For the 15′ class deck

- Embed **30–60 s** of this video **or** live-demo the same clicks.  
- Keep **one** slide on **why two sentiment models** (informal text).  
- Keep **one** slide on **time-honest evaluation** (rolling / holdout, no random split).  
- Keep **one** slide on **limitations** (sample size, causation).  
- Full slide text and FAQ: [PRESENTATION_OUTLINE.md](PRESENTATION_OUTLINE.md).
