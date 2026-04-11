# Case study (fictional): **LinguaLoop Ltd.**

Use this narrative in the **presentation** and when walking through the PoC: run the demo, show the brief, then tell **what LinguaLoop did next**.

---

## Company

**LinguaLoop Ltd.** — a **B2C subscription** mobile app that sells **bite-sized language lessons** for commuters (freemium + premium). They are **listed** on a major exchange; leadership cares about **short-term volatility** around releases and **earnings narratives**.

---

## What they sell

- Daily micro-lessons, streaks, and pronunciation drills  
- Premium removes ads and unlocks full courses  
- Heavy **App Store / Play Store** review volume after each release

---

## How they qualified to use our product

| Criterion | LinguaLoop |
|-----------|------------|
| **Data they control** | They export **their own** review dumps (CSV) under their **vendor agreements** and internal retention policy — no scraping of store pages required. |
| **Financial series** | Their finance team provides **daily total shareholder return** (or equivalent) aligned to trading calendars. |
| **Overlap** | At least **several months** of overlapping **daily** review aggregates and market data (PoC used longer; class repo uses a **tiny synthetic window** only to run everywhere). |
| **Use case** | **Monitoring + triage**, not a black-box trading mandate — legal/compliance comfortable with **internal** analytics. |

---

## Why they want it

- **Product and PR** see review **spikes** after releases but cannot say whether **markets** “care” or whether noise is just **always-on complaining**.  
- **IR** gets analyst questions tying **app ratings** to **the stock**; they need **repeatable** evidence and **cited examples**, not anecdotes.  
- **Leadership** wants one **weekly brief**: sentiment trend, link to returns (with caveats), and **which reviews to read first**.

---

## Problem → how our product helps

**Problem:** After a **bad 3.2.x release**, average star ratings wobbled and **support tickets** spiked. The CEO asked whether the dip was **material to investors** or **normal app-store noise**. Teams spent days **manually reading** thousands of reviews and still disagreed.

**How we help:**

1. **NLP (dual sentiment)**  
   - **TextBlob** = fast lexicon baseline.  
   - **RoBERTa fine-tuned on social-style text** = better on **short, informal, sarcastic** reviews — the dominant style in app comments.  
   - Where the two **disagree**, we **flag** rows for humans (often **negation / sarcasm**). That is **text mining with a purpose**, not an extra chart for decoration.

2. **Quant**  
   - Build **daily** sentiment series, **merge** with returns, use **lags** (e.g. yesterday’s sentiment vs today’s return) and report **correlation / regression** with honest **association ≠ causation** language.

3. **LLM layer (grounded)**  
   - The model does **not** invent statistics. It receives **our tables + a few retrieved quotes** (especially **high-disagreement** reviews) and produces a **one-page executive brief** for IR/product.

---

## Actionable insights the brief is meant to unlock

- **Product / engineering:** Prioritize **themes** in **flagged** reviews (e.g. crashes, billing, offline mode) when those days **overlap** weak return windows — **backlog evidence**, not vibes.  
- **IR / comms:** When **RoBERTa is much more negative** than TextBlob, **prepare Q&A** and disclosure language that acknowledges **user frustration** without overreacting to **lexicon noise**.  
- **Risk / leadership:** Use **disagreement rate** as a **data-quality dial**: if it rises after a launch, schedule a **human review sprint** instead of trusting a single score.

---

## Story after the demo (“they went and did XYZ”)

**What the PoC showed (toy numbers):** The template brief listed **lag-1 correlations**, the **worst return day** in the window, and (with ML installed) **which reviews** had the largest **TextBlob vs RoBERTa** gap.

**What LinguaLoop did next (narrative you tell in class):**

1. They **froze feature work** on a cosmetic redesign and shipped a **stability patch** targeting the **themes** that appeared in **high-disagreement negative** reviews during the drawdown week.  
2. **IR** added two **standard Q&A lines** sourced from **verbatim quotes** the system surfaced (not from generic ChatGPT).  
3. They adopted a **weekly** run of the pipeline on **their** exports; the LLM brief became the **first page** of the **release retrospective** deck.

**Outcome (story):** Post-patch, **volatility around minor releases** dropped enough that **support volume** and **leadership fire drills** fell — the business value was **coordination and speed**, not “beating the market.”

---

## Mapping to your course

- **NLP / semantic analysis:** Dual scoring + **disagreement as semantic risk signal** for informal text.  
- **Text mining:** Retrieve and rank **reviews worth reading** instead of reading all.  
- **Innovation:** **Grounded** LLM output tied to **measured** disagreement and **finance** merge.

---

**Demo data:** The public **Extended demo** uses ticker **LLGL** and ~90 trading days of synthetic reviews and returns so the UI resembles a pilot — not LinguaLoop’s real books.

_Fictional company for pedagogy; not affiliated with any real issuer._
