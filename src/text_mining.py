"""
Lightweight text mining on **selected** reviews only (disagreement + extreme-return days).

Bigrams: CountVectorizer; topics: TF-IDF + NMF. Keeps NLP depth without global LDA on all text.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _disagreement_subset(reviews_scored: pd.DataFrame, max_reviews: int) -> pd.DataFrame:
    rev = reviews_scored.copy()
    if not rev["sentiment_transformer"].notna().any():
        return rev.iloc[0:0]
    rev["disagreement"] = (rev["sentiment_textblob"] - rev["sentiment_transformer"]).abs()
    return rev.sort_values("disagreement", ascending=False).head(max_reviews)


def _extreme_return_subset(
    reviews_scored: pd.DataFrame,
    reg: pd.DataFrame,
    *,
    quantile: float = 0.1,
    max_per_tail: int = 200,
) -> pd.DataFrame:
    """Reviews whose (date, ticker) falls on very low or very high return days."""
    if reg.empty or "ret" not in reg.columns:
        return reviews_scored.iloc[0:0]
    r = reg.dropna(subset=["date", "ticker", "ret"]).copy()
    lo = r["ret"].quantile(quantile)
    hi = r["ret"].quantile(1.0 - quantile)
    flag = r[(r["ret"] <= lo) | (r["ret"] >= hi)][["date", "ticker"]].drop_duplicates()
    flag["date"] = pd.to_datetime(flag["date"]).dt.normalize()
    rev = reviews_scored.copy()
    rev["date"] = pd.to_datetime(rev["date"], errors="coerce").dt.normalize()
    m = rev.merge(flag, on=["date", "ticker"], how="inner")
    if len(m) > max_per_tail * 2:
        m = m.sample(n=max_per_tail * 2, random_state=42)
    return m


def select_mining_corpus(
    reviews_scored: pd.DataFrame,
    reg: pd.DataFrame,
    *,
    top_disagreement: int = 80,
    extreme_quantile: float = 0.1,
) -> tuple[pd.DataFrame, str]:
    parts: list[pd.DataFrame] = []
    dsub = _disagreement_subset(reviews_scored, top_disagreement)
    if len(dsub):
        parts.append(dsub)
    esub = _extreme_return_subset(reviews_scored, reg, quantile=extreme_quantile)
    if len(esub):
        parts.append(esub)
    if not parts:
        return reviews_scored.iloc[0:0], "no corpus"
    cat = pd.concat(parts, ignore_index=True)
    cat = cat.drop_duplicates(subset=["review_text", "date", "ticker"], keep="first")
    bits: list[str] = []
    if len(dsub):
        bits.append(f"top **{len(dsub)}** by |TextBlob − RoBERTa|")
    if len(esub):
        bits.append(f"reviews on **{extreme_quantile:.0%}** / **{1 - extreme_quantile:.0%}** return tails")
    if not bits:
        return cat, "empty"
    desc = " plus ".join(bits)
    if not len(dsub) and len(esub):
        desc += " (RoBERTa off → no disagreement slice)"
    return cat, desc


def top_bigrams(texts: list[str], *, top_n: int = 15, min_df: int = 1) -> list[dict[str, Any]]:
    from sklearn.feature_extraction.text import CountVectorizer

    if len(texts) < 3:
        return []
    vec = CountVectorizer(
        ngram_range=(2, 2),
        stop_words="english",
        min_df=min_df,
        max_features=5000,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )
    try:
        X = vec.fit_transform(texts)
    except ValueError:
        return []
    if X.shape[1] == 0:
        return []
    counts = np.asarray(X.sum(axis=0)).ravel()
    names = vec.get_feature_names_out()
    order = np.argsort(-counts)[:top_n]
    return [{"phrase": str(names[i]), "count": int(counts[i])} for i in order if counts[i] > 0]


def nmf_topics(
    texts: list[str],
    *,
    n_topics: int = 5,
    n_top_words: int = 8,
    random_state: int = 42,
) -> list[dict[str, Any]]:
    from sklearn.decomposition import NMF
    from sklearn.feature_extraction.text import TfidfVectorizer

    n_docs = len(texts)
    if n_docs < max(8, n_topics * 2):
        return []
    k = min(n_topics, n_docs // 2)
    if k < 2:
        return []
    vec = TfidfVectorizer(
        max_features=2000,
        stop_words="english",
        min_df=1,
        max_df=0.95,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]+\b",
    )
    try:
        X = vec.fit_transform(texts)
    except ValueError:
        return []
    if X.shape[1] < k:
        return []
    nmf = NMF(n_components=k, random_state=random_state, init="nndsvda", max_iter=400)
    try:
        W = nmf.fit_transform(X)
    except Exception:
        return []
    feats = vec.get_feature_names_out()
    topics: list[dict[str, Any]] = []
    for tid in range(k):
        row = nmf.components_[tid]
        top_idx = np.argsort(-row)[:n_top_words]
        words = [str(feats[j]) for j in top_idx if row[j] > 0]
        topics.append({"id": tid, "top_words": ", ".join(words)})
    return topics


def run_text_mining(
    reviews_scored: pd.DataFrame,
    reg: pd.DataFrame,
    *,
    top_disagreement: int = 80,
    bigram_top_n: int = 15,
    nmf_k: int = 5,
) -> dict[str, Any] | None:
    sub, source_desc = select_mining_corpus(
        reviews_scored, reg, top_disagreement=top_disagreement
    )
    if sub.empty:
        return None
    texts = sub["review_text"].astype(str).tolist()
    texts = [t for t in texts if len(t.strip()) > 5]
    if len(texts) < 3:
        return {
            "n_docs": len(texts),
            "bigrams": [],
            "nmf_topics": [],
            "source_description": source_desc,
        }
    bigrams = top_bigrams(texts, top_n=bigram_top_n)
    topics = nmf_topics(texts, n_topics=nmf_k)
    return {
        "n_docs": len(texts),
        "bigrams": bigrams,
        "nmf_topics": topics,
        "source_description": source_desc,
    }
