"""
Heuristic column detection for review and market CSVs (PoC ETL).
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


DATE_HINTS = frozenset(
    {
        "date",
        "time",
        "created",
        "created_at",
        "timestamp",
        "review_date",
        "posted",
        "dt",
    }
)
TEXT_HINTS = frozenset(
    {
        "review_text",
        "text",
        "body",
        "content",
        "comment",
        "review",
        "message",
    }
)
TICKER_HINTS = frozenset({"ticker", "symbol", "sym", "stock", "tick"})
RET_HINTS = frozenset({"ret", "return", "daily_return", "log_ret", "pct_return"})


def _norm(c: str) -> str:
    return str(c).strip().lower().replace(" ", "_")


def _parse_datetime_series(s: pd.Series) -> pd.Series:
    """Avoid noisy per-element inference warnings on mixed date strings."""
    try:
        return pd.to_datetime(s, errors="coerce", format="mixed")
    except (TypeError, ValueError):
        return pd.to_datetime(s, errors="coerce")


def _score_date_col(df: pd.DataFrame, col: str) -> float:
    s = df[col].head(200)
    parsed = _parse_datetime_series(s)
    rate = parsed.notna().mean()
    score = rate * 3.0
    lc = _norm(col)
    if lc in DATE_HINTS or any(h in lc for h in ("date", "time", "created")):
        score += 2.0
    return score


def _score_text_col(df: pd.DataFrame, col: str) -> float:
    if df[col].dtype != object and not str(df[col].dtype).startswith("string"):
        return 0.0
    s = df[col].dropna().astype(str).head(150)
    if s.empty:
        return 0.0
    mean_len = s.str.len().mean()
    score = min(mean_len / 30.0, 4.0)
    lc = _norm(col)
    if lc in TEXT_HINTS or any(h in lc for h in ("text", "body", "review", "comment")):
        score += 3.0
    return score


def _score_ticker_col(df: pd.DataFrame, col: str) -> float:
    s = df[col].dropna().astype(str).head(100)
    if s.empty:
        return 0.0
    uniq = s.nunique()
    lens = s.str.len()
    score = 1.0
    if 1 <= uniq <= min(50, len(s)):
        score += 2.0
    if lens.median() <= 6:
        score += 1.0
    lc = _norm(col)
    if lc in TICKER_HINTS or "symbol" in lc or "ticker" in lc:
        score += 3.0
    return score


def _score_ret_col(df: pd.DataFrame, col: str) -> float:
    num = pd.to_numeric(df[col].head(200), errors="coerce")
    rate = num.notna().mean()
    if rate < 0.5:
        return 0.0
    score = rate * 2.0
    v = num.dropna()
    if len(v) and v.abs().median() < 0.5:
        score += 1.0
    lc = _norm(col)
    if lc in RET_HINTS or "return" in lc:
        score += 3.0
    return score


@dataclass
class ReviewColumnGuess:
    date_col: str
    text_col: str
    ticker_col: str
    confidence: float  # 0–1 rough


@dataclass
class MarketColumnGuess:
    date_col: str
    ticker_col: str
    ret_col: str
    confidence: float


def infer_review_columns(df: pd.DataFrame) -> ReviewColumnGuess:
    cols = list(df.columns)
    if not cols:
        return ReviewColumnGuess("date", "review_text", "ticker", 0.0)

    best_d = max(cols, key=lambda c: _score_date_col(df, c))
    best_t = max(cols, key=lambda c: _score_text_col(df, c))
    best_k = max(cols, key=lambda c: _score_ticker_col(df, c))

    # Resolve collisions: prefer distinct columns
    used = {best_d, best_t, best_k}
    if len(used) < 3:
        scores_t = [(c, _score_text_col(df, c)) for c in cols if c != best_d]
        scores_t.sort(key=lambda x: -x[1])
        for c, _ in scores_t:
            if c not in {best_d, best_k}:
                best_t = c
                break
        scores_k = [(c, _score_ticker_col(df, c)) for c in cols if c not in (best_d, best_t)]
        scores_k.sort(key=lambda x: -x[1])
        if scores_k:
            best_k = scores_k[0][0]

    s_d = _score_date_col(df, best_d)
    s_tx = _score_text_col(df, best_t)
    s_k = _score_ticker_col(df, best_k)
    conf = min(1.0, (s_d + s_tx + s_k) / 18.0)
    return ReviewColumnGuess(date_col=best_d, text_col=best_t, ticker_col=best_k, confidence=conf)


def infer_market_columns(df: pd.DataFrame) -> MarketColumnGuess:
    df2 = df.copy()
    orig_map: dict[str, str] = {}
    for c in df.columns:
        n = _norm(c)
        if n not in orig_map:
            orig_map[n] = c
    df2.columns = [ _norm(c) for c in df2.columns]
    cols = list(df2.columns)
    if not cols:
        return MarketColumnGuess("date", "ticker", "ret", 0.0)

    def orig(col: str) -> str:
        return orig_map.get(col, col)

    best_d = max(cols, key=lambda c: _score_date_col(df2, c))
    best_k = max(cols, key=lambda c: _score_ticker_col(df2, c))
    ret_candidates = [c for c in cols if c not in (best_d, best_k)]
    if not ret_candidates:
        ret_candidates = cols
    best_r = max(ret_candidates, key=lambda c: _score_ret_col(df2, c))

    s_d = _score_date_col(df2, best_d)
    s_k = _score_ticker_col(df2, best_k)
    s_r = _score_ret_col(df2, best_r)
    conf = min(1.0, (s_d + s_k + s_r) / 15.0)
    return MarketColumnGuess(
        date_col=orig(best_d),
        ticker_col=orig(best_k),
        ret_col=orig(best_r),
        confidence=conf,
    )
