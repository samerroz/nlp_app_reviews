# ReviewSignal — presentation outline (copy-friendly)

Plain-language deck text aligned with the current PoC (landing page → **Run report** → six-tab dashboard). Use as speaker notes or paste into Word/Google Slides.

---

## 1. Title (slide)

**ReviewSignal — From thousands of app reviews to clear signals: sentiment, market alignment, and what to read first**

Optional subtitle: NLP + finance **monitoring** for consumer tech, IR, and product teams.

---

## 2. Introduction (what is this, in one breath?)

Companies get **massive** streams of app store reviews and **separate** stock market data. Leadership asks: *“Is user anger real, material, or just noise?”* and *“What should we actually fix?”*

**ReviewSignal** is a small internal tool (proof of concept) that:

- turns reviews into **measurable daily sentiment**;
- lines that up with **daily returns** (honestly: **association**, not magic prediction);
- flags **which few reviews** humans should read when automatic scores **disagree**;
- outputs a **short brief**, **PDF/CSV exports**, and **grounded Q&A** for product, IR, and execs.

---

## 3. Context, need, and scope

**Context**

- Reviews are short, emotional, informal (slang, sarcasm, “not bad”).
- Markets react to news, earnings, product incidents — sometimes **at the same time** as review waves.

**Need**

- One **repeatable workflow** (same steps every week), not ad-hoc reading or gut feel.
- An **evidence pack** for IR prep, release retros, and prioritization.

**Scope (what we are / aren’t claiming)**

- **In scope:** monitoring, triage, association with returns, **cited** review snippets, exports for meetings.
- **Out of scope (for this course PoC):** guaranteed alpha, legal trading advice, scraping stores as the main data plan (we assume **exports / licensed data**).

---

## 4. Challenges (why this is hard)

- **Language is messy** — the same words can mean opposite things in context.
- **One angry review** shouldn’t drive the whole story → we **aggregate by day**.
- **Correlation ≠ causation** — bad reviews can coincide with bad returns for many reasons (e.g. release bug + broad market sell-off).
- **Different automatic scorers** can disagree; we need to know **when not to trust a single number**.

---

## 5. Objectives (crystal clear — what the “company” wants)

Match these to your **solution** slides when you walk the product.

| # | Objective | Plain English |
|---|-----------|----------------|
| **O1** | Measure user mood over time | Turn text into **daily sentiment curves** leadership can track. |
| **O2** | Check alignment with the market | See if **yesterday’s** average mood relates to **today’s** return (lagged check, honest wording). |
| **O3** | Prioritize human reading | Surface a **short list** of reviews most likely **misleading for simple scores** — read those first. |
| **O4** | Actionable output | Produce **insights + executive brief + PDF/CSV** for product / IR / leadership — not just charts for data scientists. |

**One-line “so what” for the company:**  
Spend less time drowning in text; spend more time on the **few days** and **few reviews** that matter for **narrative risk** and **product fixes**.

---

## 6. Solution overview — what we built

**ReviewSignal** is a single **Streamlit** app with two phases:

1. **Landing (start screen)** — Full-width, presentation-style layout: **company name**, status for built-in demo or **your CSVs**, file uploaders, options (RoBERTa on/off, queue size, optional brief polish), **Advanced** column overrides, and a primary **Run report** button.
2. **Results dashboard** — After a successful run: a top bar with **Back to start** (new run or change data) and **six tabs** so each audience finds its lane without clutter.

**The six tabs**

| Tab | Purpose |
|-----|---------|
| **Overview** | Data health (counts, date span, overlap), downloads (**PDF**, disagreement **CSV**), RoBERTa status, and **plain-language “what this means”** bullets. |
| **Market & stats** | Dual sentiment over time, **Pearson** and **OLS (HC3)** with confidence intervals, **correlation stability** (expanding / rolling), and **chronological holdout** checks. |
| **Reviews** | **Disagreement queue** only — reviews where **TextBlob** and **RoBERTa** disagree the most; read these first. |
| **Themes** | **Text mining** (bigrams + NMF) on a **flagged** corpus (high disagreement and extreme-return days), not on all text. |
| **Brief** | Executive **markdown** brief, optional API-polished wording (same facts), and **technical notes** for analysts. |
| **Q&A** | **Grounded chat**: answers use a **frozen snapshot** of this run — the model must not invent statistics (optional API; rule-based fallback without a key). |

**Reminder line for the deck:** Not trading advice — **monitoring, triage, and narrative prep**.

---

## 7. How the solution maps to the objectives

| Objective | Where it shows up in the product |
|-----------|----------------------------------|
| **O1** — Mood over time | **Market & stats** — dual daily sentiment chart. |
| **O2** — Alignment with the market | **Market & stats** — lag-1 association, regression, stability, holdout (honest language). |
| **O3** — Prioritize reading | **Reviews** tab — disagreement queue; **Themes** — what people said on risky days. |
| **O4** — Actionable output | **Overview** — metrics, downloads, plain-language bullets; **Brief** + **PDF**; **Q&A** for ad-hoc questions. |

---

## 8. How it works (pipeline — one diagram in your head)

1. **Ingest** — Review export + daily returns (CSV); column names inferred, overridable.
2. **NLP** — Per-review **TextBlob** and optional **RoBERTa** (social-style text); both mapped to **−1…+1**; **daily averages**; **disagreement** = where they diverge.
3. **Quant** — Merge on calendar; **lag-1** sentiment vs return; **Pearson**, **OLS with HC3**, **expanding/rolling r**, **chronological holdout** (no random split on time).
4. **Text mining** — Only on **flagged** reviews (disagreement + extremes), not whole corpus.
5. **Outputs** — Rule-based + optional LLM **brief**; **PDF**; **grounded chat** JSON context.

---

## 9. Why two sentiment models? (slide you will reuse)

- **TextBlob** = fast lexicon baseline.
- **RoBERTa** (social/review-style) = better on **short, informal, sarcastic** text.
- When they **disagree**, simple scores are often wrong → **human should read that review**.

**One line:** *Disagreement = “this text broke the simple model — a human should look here.”*

---

## 10. Why would IR / product care about “correlation” at all?

- **IR / execs:** When the stock moves, people ask *“what happened with users?”* A **time-aligned signal** helps you answer with **evidence**, not guesses — **prepared answers**, not crystal-ball trading.
- **Product / strategy:** If bad sentiment **clusters with** wobbly returns around releases, treat it as **incident + narrative risk**. If sentiment is noisy but the market **doesn’t** react, **fix the app** without declaring a capital-markets crisis.

**So:** not “so we can day-trade,” but so **comms, IR, and product** share **one evidence-based picture**.

---

## 11. Evaluation we actually show (trust, not theater)

- **No random train/test split** on days (that would leak the future).
- **Expanding-window** and **rolling-window** correlation paths — show **stability**, not one lucky number.
- **Chronological holdout** — train on the past, score the **most recent** window; report holdout correlation and RMSE as a **sanity check**, not a trading backtest.
- **Full-sample OLS (HC3)** — slope, **95% CI**, p-value, R² for finance-style readers.

---

## 12. Limitations and honesty (keep this slide)

- **PoC / course scale** — public repo uses **small synthetic** data; real use is **your** exports, local or internal.
- **Association ≠ causation** — many confounders (news, sector moves, earnings).
- **Optional costs** — RoBERTa = heavier install; OpenAI = optional polish and chat; demo works without both.
- **No store scraping** in the core story — **BYOD exports** only.

---

## 13. Demo flow (what you click — aligns with video script)

1. Open app → **landing** shows demo loaded (or upload two CSVs).
2. Set **company name** → **Run report**.
3. Land on **Overview** → point at overlap and downloads.
4. **Market & stats** → dual chart, regression row, stability, holdout.
5. **Reviews** → disagreement table.
6. **Themes** → bigrams / NMF (if RoBERTa/flagged corpus populated).
7. **Brief** → narrative + optional polish + technical notes.
8. **Q&A** → one question; show **grounded** answer (or stub without API).
9. **Back to start** → new run.

---

## 14. Future work (optional slide)

- Richer **aspect** labels (bugs vs billing vs UX) on flagged text.
- Larger windows and **proprietary validation** (offline research story).
- Scheduled runs, auth, and deployment **inside** the company — still not a public trading product.

---

## 15. Closing (slide)

**ReviewSignal** turns **noisy reviews** and **market lines** into a **repeatable weekly story**: daily mood, honest alignment checks, **which sentences to read**, and **exports** leadership can reuse.

**Tagline repeat:** Less drowning in text; more time on what **moves narrative and product**.

---

## FAQ (end of deck or appendix)

**Why correlate reviews with the stock?**  
For **preparedness and alignment** across IR, product, and comms — not for day trading. See section 10.

**What “real insights” do we get?**  
Daily mood trend; **worst market days** paired with **what users said**; **disagreement-ranked** quotes where simple NLP fails (sarcasm, negation). Topics in the app are **exploratory** on a **flagged** subset, not full roadmap prediction.

**Why is TextBlob vs RoBERTa disagreement meaningful?**  
When both say “bad,” you already know it’s negative. When they **disagree**, the sentence is often **hard for a dictionary** — that’s where **reading the text** pays off and where **daily averages** could be skewed if you only used lexicon scores.

**Is this investment advice?**  
No. Internal **monitoring and documentation** only.

**What data do we need?**  
Two CSVs you’re allowed to use: **reviews** (date, text, ticker or equivalent) and **market** (date, return, aligned ticker). Built-in demo for class; your files for real pilots.

---

_End of outline. Update this file when the product or deck changes._
