# Demo video script (3–5 minutes)

**Goal:** Show the **product** (Streamlit), not the terminal. Use **built-in sample data** unless you have permission to blur a real export.

**Recording:** 1080p, hide file paths and API keys. Narrate in **complete sentences**; cut dead air in edit.

---

## Shot list

| Time (approx) | Visual | Say (guide, not word-for-word) |
|---------------|--------|--------------------------------|
| 0:00–0:25 | Title slide or app idle | “We built ReviewSignal for teams that have **app review exports** and **daily returns** but no repeatable way to score text, check overlap with the market, and brief leadership.” |
| 0:25–0:50 | Browser: Streamlit sidebar | “Users **bring their own CSVs** — we are not scraping stores in the core story. For the video I’ll load the **Extended demo** (~90 trading days) so the chart looks like a real pilot.” Click **Extended demo**. |
| 0:50–1:10 | Sidebar: company name, toggles | “Company name feeds the **executive brief**. RoBERTa adds a second sentiment opinion for **short informal** text; you can turn it off if ML is not installed.” |
| 1:10–1:35 | Click **Run analysis**; spinner | “Run scores every review with **TextBlob** and optionally **RoBERTa**, averages by **day**, merges with **returns**, and applies a **one-day lag** so we are not cheating with same-day noise.” |
| 1:35–2:15 | **Data health** metrics | “Here is **overlap**: how many reviews, how many **trading days** line up. If tickers or dates do not match, the app errors clearly.” |
| 2:15–2:55 | **Dual sentiment** chart | “Two lines: lexicon vs neural. When they diverge, **simple scores lie** — that is why we surface a **disagreement queue** next.” |
| 2:55–3:25 | **Pearson metrics** + caption | “These are **associations**, not proof of causation; our **tiny sample** is for the workflow demo. Our **course write-up** cites a **longer offline study** where one name showed significant results.” |
| 3:25–3:55 | **Disagreement** table | “These rows are ranked by **absolute gap** between scorers — **read these first** for sarcasm and negation.” |
| 3:40–4:00 | **Actionable insights** | “These bullets interpret **depth**, **model disagreement**, and **drawdown context** — they turn the dashboard into **next steps**, not just charts.” |
| 3:55–4:25 | **Executive brief** + **Download PDF** | “The brief **does not invent statistics**. **PDF** packages the same facts for email or the board pack. Optional API polish only **rephrases** under a strict prompt.” |
| 4:25–4:45 | Closing: LinguaLoop one-liner | “**LinguaLoop** used this style of output to prioritize a **stability patch** and prep **IR** — story in our case study doc.” |
| 4:45–5:00 | Limitations | “Limitations: **no transaction costs** in the toy run, **no** claim that every stock behaves like Duolingo, **association ≠ causation**.” |

---

## Checklist before you hit record

- [ ] `pip install -r requirements.txt -r requirements-ui.txt` and optionally `-r requirements-ml.txt`  
- [ ] `streamlit run app/streamlit_app.py` — dry run once  
- [ ] Browser zoom ~100%, window wide enough for metrics row  
- [ ] `OPENAI_API_KEY` unset **or** expander closed if you do not want to show API content  

---

## For the 15′ class deck

- Embed **30–60 s** of this video **or** live-demo the same clicks.  
- Keep **one** slide on **why two sentiment models** (informal text).  
- Keep **one** slide on **limitations** (sample size, causation).
