"""
Time-series-safe evaluation: expanding / rolling correlation and chronological holdout OLS.

No random train/test split — avoids leakage in sequential financial data.
Document in reports as **pseudo-forecasting**: train on past, score association on the final holdout window.
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd


def _std_degenerate(s: pd.Series) -> bool:
    """True if sample std is zero / tiny (OLS and corr unstable; HC3 SE often NaN)."""
    v = float(s.std(ddof=1))
    return (not math.isfinite(v)) or v < 1e-9


def _single_ticker_panel(reg: pd.DataFrame) -> pd.DataFrame:
    """Use one ticker for stability paths if multiple present (typical PoC = one name)."""
    if reg.empty:
        return reg
    tickers = reg["ticker"].dropna().unique()
    if len(tickers) <= 1:
        return reg.sort_values("date").reset_index(drop=True)
    t0 = sorted(tickers)[0]
    return reg[reg["ticker"] == t0].sort_values("date").reset_index(drop=True)


def expanding_window_correlation(
    reg: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    min_periods: int = 15,
) -> pd.Series:
    """Pearson corr(x,y) using all data from row 0..i (inclusive), per row after sort by date."""
    d = _single_ticker_panel(reg).dropna(subset=[x_col, y_col])
    vals: list[float] = []
    for i in range(len(d)):
        if i + 1 < min_periods:
            vals.append(float("nan"))
        else:
            sl = d.iloc[: i + 1]
            vals.append(float(sl[x_col].corr(sl[y_col])))
    return pd.Series(vals, index=d.index, name=f"expanding_corr_{x_col}")


def rolling_window_correlation(
    reg: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    window: int = 20,
    min_periods: int = 15,
) -> pd.Series:
    """Pearson corr(x,y) over the last `window` rows ending at i."""
    d = _single_ticker_panel(reg).dropna(subset=[x_col, y_col])
    vals: list[float] = []
    for i in range(len(d)):
        lo = max(0, i - window + 1)
        chunk = d.iloc[lo : i + 1]
        if len(chunk) < min_periods:
            vals.append(float("nan"))
        else:
            vals.append(float(chunk[x_col].corr(chunk[y_col])))
    return pd.Series(vals, index=d.index, name=f"rolling_corr_{x_col}_w{window}")


def holdout_ols_hc3(
    reg: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    holdout_frac: float = 0.2,
    min_train: int = 25,
    min_holdout: int = 8,
) -> dict[str, Any] | None:
    """
    Fit OLS y ~ x + const on the **chronological train** (all but last h days).
    Evaluate on **holdout**: correlation(predicted, actual) and RMSE.
    Uses HC3 robust covariance on the **training** fit only.
    """
    import statsmodels.api as sm

    d = _single_ticker_panel(reg).dropna(subset=[x_col, y_col])
    if len(d) < min_train + min_holdout:
        return None

    h = max(min_holdout, int(np.ceil(len(d) * holdout_frac)))
    h = min(h, len(d) - min_train)
    if h < 1:
        return None

    train = d.iloc[:-h].copy()
    test = d.iloc[-h:].copy()
    if _std_degenerate(train[x_col]) or _std_degenerate(train[y_col]):
        return None
    X_tr = sm.add_constant(train[x_col], has_constant="add")
    y_tr = train[y_col].astype(float)
    model = sm.OLS(y_tr, X_tr).fit(cov_type="HC3")
    try:
        se_tr = float(model.bse[x_col])
    except Exception:
        se_tr = float("nan")
    if not math.isfinite(se_tr):
        return None

    X_te = sm.add_constant(test[x_col], has_constant="add")
    y_te = test[y_col].astype(float)
    pred = model.predict(X_te)

    resid = y_te.values - pred.values
    rmse = float(np.sqrt(np.mean(resid**2)))
    corr_pa: float | None = None
    if len(test) >= 3 and np.std(pred) > 1e-12 and np.std(y_te) > 1e-12:
        corr_pa = float(np.corrcoef(pred.values, y_te.values)[0, 1])

    params = model.params
    pvalues = model.pvalues
    coef_x = float(params.get(x_col, float("nan")))
    p_x = float(pvalues.get(x_col, float("nan")))
    if not math.isfinite(coef_x) or not math.isfinite(p_x):
        return None
    r2_tr = float(model.rsquared)
    if not math.isfinite(r2_tr):
        return None
    if corr_pa is not None and not math.isfinite(corr_pa):
        corr_pa = None

    return {
        "x_col": x_col,
        "n_train": int(len(train)),
        "n_holdout": int(len(test)),
        "holdout_last_date": str(test["date"].iloc[-1].date()) if "date" in test.columns else None,
        "coef_lag_sentiment": coef_x,
        "p_value_lag_sentiment": p_x,
        "r2_train": r2_tr,
        "corr_pred_vs_actual_holdout": corr_pa,
        "rmse_holdout": rmse,
    }


def build_stability_frame(
    reg: pd.DataFrame,
    *,
    rolling_window: int = 20,
    expanding_min: int = 15,
    rolling_min: int = 15,
) -> pd.DataFrame:
    """Dates + expanding/rolling corr for TextBlob lag1; RoBERTa if column usable."""
    base = _single_ticker_panel(reg).copy()
    base = base.dropna(subset=["avg_textblob_lag1", "ret"])

    exp_tb = expanding_window_correlation(base, "avg_textblob_lag1", "ret", min_periods=expanding_min)
    roll_tb = rolling_window_correlation(
        base, "avg_textblob_lag1", "ret", window=rolling_window, min_periods=rolling_min
    )
    out = base[["date", "ticker", "ret", "avg_textblob_lag1"]].copy()
    out["expanding_corr_textblob"] = exp_tb.values
    out["rolling_corr_textblob"] = roll_tb.values

    if base["avg_transformer_lag1"].notna().sum() >= expanding_min:
        exp_tr = expanding_window_correlation(base, "avg_transformer_lag1", "ret", min_periods=expanding_min)
        roll_tr = rolling_window_correlation(
            base, "avg_transformer_lag1", "ret", window=rolling_window, min_periods=rolling_min
        )
        out["expanding_corr_roberta"] = exp_tr.values
        out["rolling_corr_roberta"] = roll_tr.values
    else:
        out["expanding_corr_roberta"] = np.nan
        out["rolling_corr_roberta"] = np.nan

    return out


def full_sample_ols_hc3(
    reg: pd.DataFrame,
    x_col: str,
    y_col: str,
    *,
    min_obs: int = 8,
) -> dict[str, Any] | None:
    """Full-window OLS y ~ x + const with HC3; coef + 95% CI (Wald on robust cov)."""
    import statsmodels.api as sm

    d = _single_ticker_panel(reg).dropna(subset=[x_col, y_col])
    if len(d) < min_obs:
        return None
    if _std_degenerate(d[x_col]) or _std_degenerate(d[y_col]):
        return None
    X = sm.add_constant(d[x_col], has_constant="add")
    y = d[y_col].astype(float)
    model = sm.OLS(y, X).fit(cov_type="HC3")
    try:
        se_x = float(model.bse[x_col])
    except Exception:
        se_x = float("nan")
    if not math.isfinite(se_x):
        return None
    ci = model.conf_int(alpha=0.05)
    coef = float(model.params.get(x_col, float("nan")))
    lo = float(ci.loc[x_col, 0]) if x_col in ci.index else float("nan")
    hi = float(ci.loc[x_col, 1]) if x_col in ci.index else float("nan")
    pval = float(model.pvalues.get(x_col, float("nan")))
    r2 = float(model.rsquared)
    if not all(math.isfinite(x) for x in (coef, lo, hi, pval, r2)):
        return None
    return {
        "x_col": x_col,
        "n": int(len(d)),
        "coef": coef,
        "ci_low": lo,
        "ci_high": hi,
        "p_value": pval,
        "r2": r2,
    }
