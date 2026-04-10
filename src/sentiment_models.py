"""
Dual sentiment: lexicon (TextBlob) vs transformer (RoBERTa on social-style text).

The transformer head is *problem-relevant*: app reviews are short, informal, and
often sarcastic or negated — lexicon methods miss that; a RoBERTa model trained
on Twitter-like text is a standard upgrade path and gives the LLM layer richer
signals (second opinion + disagreement flags for "read these reviews manually").
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from textblob import TextBlob

# Fine-tuned on noisy social text; 3-way NEG / NEU / POS — maps to [-1, 1] via expectation.
DEFAULT_TRANSFORMER_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"


def polarity_textblob(text: str) -> float:
    return float(TextBlob(str(text)).sentiment.polarity)


def _label_to_val(label: str) -> float:
    low = label.lower()
    if "neg" in low:
        return -1.0
    if "pos" in low:
        return 1.0
    return 0.0


def transformer_expected_scores(
    texts: Sequence[str],
    model_name: str = DEFAULT_TRANSFORMER_MODEL,
    batch_size: int = 16,
    max_length: int = 128,
) -> np.ndarray:
    """Return one scalar in [-1, 1] per text (probability-weighted pos/neu/neg)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()

    id2label = model.config.id2label
    idx_to_val = {int(k): _label_to_val(v) for k, v in id2label.items()}
    n_classes = len(idx_to_val)

    out: list[float] = []
    texts_list = [str(t) if t is not None else "" for t in texts]

    with torch.no_grad():
        for i in range(0, len(texts_list), batch_size):
            batch = texts_list[i : i + batch_size]
            enc = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            for row in probs:
                s = sum(float(row[j]) * idx_to_val.get(j, 0.0) for j in range(min(len(row), n_classes)))
                out.append(float(np.clip(s, -1.0, 1.0)))

    return np.array(out, dtype=np.float64)


def try_transformer_scores(texts: Sequence[str]) -> tuple[np.ndarray | None, str | None]:
    """
    Returns (scores, error_message). If imports or runtime fail, scores is None.
    """
    try:
        scores = transformer_expected_scores(texts)
        return scores, None
    except Exception as e:  # pragma: no cover - env specific
        return None, f"{type(e).__name__}: {e}"
