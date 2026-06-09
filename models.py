"""
models.py
---------
AI prediction layer for the Agentic AlgoTrading System dashboard.

Implements a practical stand-in for the doc's Layer 3 "Signal Engine"
(BiLSTM-Transformer in the full system) using gradient boosting, with the
doc's honesty constraints built in:

  - Walk-forward, time-ordered split with an EMBARGO gap (no look-ahead;
    purged CV per Lopez de Prado).
  - Out-of-sample metrics reported front and center: rank IC (Spearman)
    and directional hit rate. Per the doc: 52-55% hit rate is genuinely
    excellent; monthly OOS R^2 of ~0.3-0.7% is a triumph (Gu-Kelly-Xiu).
  - Money comes from conviction-weighted sizing, not frequency — so the
    output is a conviction score wired into fractional Kelly, with the
    implied win probability conservatively capped.

Also includes an optional LLM Bull/Bear analyst (Layer 4 adversarial
debate) via the Anthropic API, with the doc's caveat encoded: two prompts
to one model are NOT two independent analysts — treat it as a de-biasing
aid, never as independent confirmation.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import requests
from scipy.stats import spearmanr
from sklearn.ensemble import GradientBoostingRegressor

# ----------------------------------------------------------------------
# Feature matrix (per ticker) — only price-derived features the doc
# acknowledges: lagged returns, momentum, reversal, vol, ATR%, volume.
# ----------------------------------------------------------------------

def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False).mean()
    dn = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False).mean()
    rs = up / dn.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def build_feature_matrix(df: pd.DataFrame, k_forward: int = 5) -> pd.DataFrame:
    """
    Per-ticker feature/target table. Target = forward k-bar log return.
    All features use information available at bar t only.
    """
    c = df["close"]
    logc = np.log(c)
    ret1 = logc.diff()

    h, l = df["high"], df["low"]
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    atr14 = tr.ewm(alpha=1 / 14, adjust=False).mean()

    vol_z = (df["volume"] - df["volume"].rolling(21).mean()) / df["volume"].rolling(21).std()

    feats = pd.DataFrame({
        "ret_1": ret1,
        "ret_5": logc.diff(5),
        "ret_10": logc.diff(10),
        "mom_21": logc.diff(21),
        "mom_63": logc.diff(63),
        "rev_5": -logc.diff(5),                       # short-term reversal
        "vol_21": ret1.rolling(21).std(),             # vol clustering
        "vol_ratio": ret1.rolling(21).std() / ret1.rolling(63).std(),
        "atr_pct": atr14 / c,
        "rsi_14": _rsi(c) / 100.0,
        "vol_z": vol_z,
        "dist_high_63": c / c.rolling(63).max() - 1,  # distance from 63-bar high
    }, index=df.index)

    feats["target"] = logc.shift(-k_forward) - logc   # forward k-bar return
    return feats


FEATURE_COLS = ["ret_1", "ret_5", "ret_10", "mom_21", "mom_63", "rev_5",
                "vol_21", "vol_ratio", "atr_pct", "rsi_14", "vol_z",
                "dist_high_63"]


# ----------------------------------------------------------------------
# Walk-forward training across the panel
# ----------------------------------------------------------------------

def train_and_predict(panel: dict[str, pd.DataFrame],
                      k_forward: int = 5,
                      train_frac: float = 0.7,
                      seed: int = 7) -> dict:
    """
    Pool features across tickers, train on the first `train_frac` of TIME
    (not rows), leave an embargo gap of k_forward bars, evaluate on the
    rest, then refit on all labeled data and predict the LATEST bar of
    each ticker.

    Returns {"predictions": DataFrame, "metrics": dict, "importances": Series}
    """
    frames = []
    latest_rows = {}
    for tkr, df in panel.items():
        if df is None or len(df) < 80:
            continue
        fm = build_feature_matrix(df, k_forward)
        fm["ticker"] = tkr
        latest = fm[FEATURE_COLS].iloc[[-1]].copy()
        if latest.notna().all(axis=1).iloc[0]:
            latest_rows[tkr] = latest
        frames.append(fm.dropna(subset=FEATURE_COLS + ["target"]))

    if not frames:
        return {"predictions": pd.DataFrame(), "metrics": {}, "importances": pd.Series(dtype=float)}

    data = pd.concat(frames).sort_index()
    times = data.index.unique().sort_values()
    if len(times) < 60:
        return {"predictions": pd.DataFrame(), "metrics": {}, "importances": pd.Series(dtype=float)}

    # --- time-ordered split with embargo (no look-ahead) ---
    cut_idx = int(len(times) * train_frac)
    train_end = times[cut_idx]
    embargo_start_idx = min(cut_idx + k_forward, len(times) - 1)
    test_start = times[embargo_start_idx]

    train = data[data.index <= train_end]
    test = data[data.index > test_start]

    model = GradientBoostingRegressor(
        n_estimators=300, max_depth=3, learning_rate=0.03,
        subsample=0.8, random_state=seed,
    )
    model.fit(train[FEATURE_COLS], train["target"])

    # --- honest out-of-sample metrics ---
    metrics = {}
    if len(test) >= 50:
        pred_test = model.predict(test[FEATURE_COLS])
        ic, _ = spearmanr(pred_test, test["target"])
        hit = float(np.mean(np.sign(pred_test) == np.sign(test["target"])))
        ss_res = float(np.sum((test["target"] - pred_test) ** 2))
        ss_tot = float(np.sum((test["target"] - test["target"].mean()) ** 2))
        metrics = {
            "oos_rank_ic": float(ic),
            "oos_hit_rate": hit,
            "oos_r2": 1 - ss_res / ss_tot if ss_tot > 0 else np.nan,
            "n_test": int(len(test)),
            "train_end": str(train_end)[:16],
        }

    # --- refit on all labeled data, predict latest bar per ticker ---
    model.fit(data[FEATURE_COLS], data["target"])
    rows = []
    for tkr, lr in latest_rows.items():
        pred = float(model.predict(lr)[0])
        rows.append({"ticker": tkr, "pred_fwd_return": pred})

    preds = pd.DataFrame(rows)
    if not preds.empty:
        # Conviction = cross-sectional rank in [-1, 1]; the doc's mandate:
        # money comes from conviction-weighted sizing, not hit frequency.
        r = preds["pred_fwd_return"].rank(pct=True)
        preds["conviction"] = (2 * r - 1).round(3)
        preds["direction"] = np.where(preds["pred_fwd_return"] > 0, "LONG", "SHORT")
        preds = preds.sort_values("pred_fwd_return", ascending=False).reset_index(drop=True)

    importances = pd.Series(model.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
    return {"predictions": preds, "metrics": metrics, "importances": importances}


def implied_win_prob(conviction: float, oos_hit_rate: float | None) -> float:
    """
    Map conviction -> Kelly win probability, CONSERVATIVELY.
    Anchored at the model's measured OOS hit rate (or 0.50 if unknown),
    scaled by |conviction|, hard-capped at 0.56 per the doc's calibration
    ("52-56% is genuinely excellent"). Overconfident inputs to Kelly are
    the canonical path to overbetting at the worst moment.
    """
    base = oos_hit_rate if oos_hit_rate and 0.4 < oos_hit_rate < 0.7 else 0.50
    edge = max(base - 0.50, 0.0) * abs(conviction)
    return float(min(0.50 + edge, 0.56))


# ----------------------------------------------------------------------
# Optional: LLM Bull/Bear analyst (Layer 4 adversarial debate)
# ----------------------------------------------------------------------

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"


def llm_bull_bear(ticker: str, context: dict, anthropic_key: str,
                  model: str = "claude-sonnet-4-5") -> dict:
    """
    Ask Claude for a structured bull case, bear case, and calibration note
    grounded ONLY in the quantitative context provided (no fabricated
    catalysts — the doc's hallucination failure mode).

    context: dict of the name's computed metrics (momentum, vol, prediction,
    regime, etc.). Returns {"bull": str, "bear": str, "calibration": str}.
    """
    prompt = (
        "You are the adversarial debate layer of a systematic trading "
        "dashboard. Using ONLY the quantitative context below (do not "
        "invent news, catalysts, or fundamentals you were not given), "
        f"write for {ticker}:\n"
        "1) the strongest bull interpretation of these numbers,\n"
        "2) the strongest bear interpretation,\n"
        "3) a one-sentence calibration note on which side the evidence "
        "favors and how uncertain that judgment is.\n\n"
        f"Context: {json.dumps(context, default=str)}\n\n"
        "Respond ONLY with JSON: {\"bull\": \"...\", \"bear\": \"...\", "
        "\"calibration\": \"...\"} — no preamble, no markdown fences."
    )
    resp = requests.post(
        ANTHROPIC_URL,
        headers={
            "x-api-key": anthropic_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json={"model": model, "max_tokens": 700,
              "messages": [{"role": "user", "content": prompt}]},
        timeout=60,
    )
    resp.raise_for_status()
    text = "".join(blk.get("text", "") for blk in resp.json().get("content", [])
                   if blk.get("type") == "text")
    text = text.replace("```json", "").replace("```", "").strip()
    return json.loads(text)
