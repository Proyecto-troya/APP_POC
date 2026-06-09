"""
features.py
-----------
Signal & risk computations for the Agentic AlgoTrading System dashboard.

Implements ONLY the signals the Financial Behavior doc says survive
rigorous testing (Layer 2, Technical Feature Matrix):

  1. Time-series momentum / trend   (Moskowitz-Ooi-Pedersen 2012)
  2. Short-term reversal            (Lehmann 1990; Nagel 2012)
  3. Volatility clustering / ATR    (risk-sizing, NOT direction)

Plus the system-level mechanics:
  - Vol/trend 4-quadrant regime dial (HMM-lite proxy; Layer 2 classifier)
  - Cross-sectional z-score composite, sector-neutral (Layer 3 screener)
  - ATR stop framework (Layer 6)
  - Fractional Kelly sizing with edge-uncertainty shrinkage (Layer 5)
  - Effective breadth N / (1 + (N-1)·rho) (Part II, emergent behavior #6)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

ANNUALIZATION = {"4H": 252 * 2, "1D": 252, "1W": 52}  # approx bars/year


# ----------------------------------------------------------------------
# 1) The three surviving technical signals
# ----------------------------------------------------------------------

def ts_momentum(close: pd.Series, lookback: int) -> float:
    """Time-series momentum: trailing total return over `lookback` bars."""
    if len(close) <= lookback:
        return np.nan
    return float(close.iloc[-1] / close.iloc[-lookback - 1] - 1.0)


def short_term_reversal(close: pd.Series, lookback: int = 5) -> float:
    """
    Short-term reversal signal: NEGATIVE of the recent return.
    Positive value => recent sell-off => liquidity-provision long candidate.
    The doc frames this as a fee for risk-bearing, not a forecast.
    """
    if len(close) <= lookback:
        return np.nan
    return float(-(close.iloc[-1] / close.iloc[-lookback - 1] - 1.0))


def realized_vol(close: pd.Series, window: int, horizon: str) -> float:
    """Annualized realized volatility of log returns."""
    rets = np.log(close).diff().dropna().tail(window)
    if len(rets) < max(5, window // 2):
        return np.nan
    return float(rets.std(ddof=1) * np.sqrt(ANNUALIZATION.get(horizon, 252)))


def atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    """Average True Range (Wilder). Requires high/low/close columns."""
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([h - l, (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / window, adjust=False).mean()


# ----------------------------------------------------------------------
# 2) Regime dial — vol/trend quadrant (HMM-lite proxy)
# ----------------------------------------------------------------------

def classify_regime(close: pd.Series, horizon: str = "1D") -> dict:
    """
    4-state regime proxy from trend sign x volatility level.
    The doc's mandate: use this for RISK SCALING (Moreira-Muir 2017),
    never for directional forecasts (Asness 2016 skepticism on timing).

    Returns dict with regime label and a gross-exposure multiplier.
    """
    if len(close) < 60:
        return {"regime": "Insufficient data", "exposure_mult": 0.5,
                "trend": np.nan, "vol_ratio": np.nan}

    trend = ts_momentum(close, min(63, len(close) - 2))      # ~3M trend
    vol_short = realized_vol(close, 21, horizon)
    vol_long = realized_vol(close, min(126, len(close) - 2), horizon)
    vol_ratio = vol_short / vol_long if vol_long and vol_long > 0 else np.nan

    high_vol = bool(vol_ratio and vol_ratio > 1.2)
    up_trend = bool(trend and trend > 0)

    if up_trend and not high_vol:
        regime, mult = "Calm Uptrend (risk-on)", 1.0
    elif up_trend and high_vol:
        regime, mult = "Volatile Uptrend (reduce)", 0.7
    elif not up_trend and not high_vol:
        regime, mult = "Calm Downtrend (defensive)", 0.6
    else:
        regime, mult = "Volatile Downtrend (de-risk)", 0.35

    return {"regime": regime, "exposure_mult": mult,
            "trend": trend, "vol_ratio": vol_ratio}


# ----------------------------------------------------------------------
# 3) Cross-sectional screener — the system's durable, capacity-rich core
# ----------------------------------------------------------------------

def cross_sectional_scores(panel: dict[str, pd.DataFrame],
                           sectors: dict[str, str],
                           horizon: str = "1D",
                           mom_lb: int = 126,
                           rev_lb: int = 5) -> pd.DataFrame:
    """
    Build per-name signals, z-score them cross-sectionally, then
    SECTOR-NEUTRALIZE (demean within sector) so the residual ranking is
    genuine idiosyncratic selection — the only uncrowded alpha (per doc).

    panel: {ticker: OHLCV DataFrame}; sectors: {ticker: sector label}
    """
    rows = []
    for tkr, df in panel.items():
        if df is None or df.empty or len(df) < 30:
            continue
        close = df["close"]
        a = atr(df).iloc[-1] if len(df) >= 15 else np.nan
        rows.append({
            "ticker": tkr,
            "sector": sectors.get(tkr, "Unknown"),
            "last": float(close.iloc[-1]),
            "momentum": ts_momentum(close, min(mom_lb, len(close) - 2)),
            "reversal": short_term_reversal(close, rev_lb),
            "vol_ann": realized_vol(close, 21, horizon),
            "atr": float(a) if pd.notna(a) else np.nan,
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    def _z(s: pd.Series) -> pd.Series:
        sd = s.std(ddof=0)
        return (s - s.mean()) / sd if sd and sd > 0 else s * 0.0

    out["z_momentum"] = _z(out["momentum"])
    out["z_reversal"] = _z(out["reversal"])
    out["z_lowvol"] = _z(-out["vol_ann"])  # low-vol preference (defensive quality proxy)

    # Sector-neutralize: demean each z within its sector
    for col in ["z_momentum", "z_reversal", "z_lowvol"]:
        out[col + "_sn"] = out[col] - out.groupby("sector")[col].transform("mean")

    # Composite: momentum-dominant core, reversal satellite, low-vol tilt.
    # Weights mirror the doc's "durable core / fragile satellite" thesis.
    out["composite"] = (0.55 * out["z_momentum_sn"]
                        + 0.25 * out["z_reversal_sn"]
                        + 0.20 * out["z_lowvol_sn"])
    out["rank"] = out["composite"].rank(ascending=False).astype(int)
    return out.sort_values("composite", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------------
# 4) Risk & sizing — ATR stops, fractional Kelly, effective breadth
# ----------------------------------------------------------------------

def atr_stop_levels(last_price: float, atr_value: float,
                    multiple: float = 2.5) -> dict:
    """ATR-scaled stop/target. The doc: stops reshape the distribution
    (cut left tail), they do NOT add return (Kaminski-Lo 2014)."""
    return {
        "stop_long": last_price - multiple * atr_value,
        "stop_short": last_price + multiple * atr_value,
        "atr_pct": atr_value / last_price if last_price else np.nan,
    }


def fractional_kelly(win_prob: float, win_loss_ratio: float,
                     fraction: float = 0.35,
                     edge_uncertainty: float = 0.5) -> dict:
    """
    Kelly f* = p - (1-p)/b, then shrink twice:
      - fraction (25-50% per the doc; overbetting is brutally asymmetric)
      - edge_uncertainty in [0,1]: extra shrinkage as estimation error rises
        ("the binding constraint is not your edge but your uncertainty
         about your edge").
    """
    p, b = float(win_prob), float(win_loss_ratio)
    full = p - (1 - p) / b if b > 0 else 0.0
    full = max(full, 0.0)
    sized = full * fraction * (1.0 - edge_uncertainty)
    return {"full_kelly": full, "applied_fraction": sized}


def effective_breadth(n: int, avg_corr: float) -> float:
    """Effective independent bets = N / (1 + (N-1)·rho). Part II, #6."""
    if n <= 0:
        return 0.0
    rho = min(max(avg_corr, 0.0), 0.9999)
    return n / (1 + (n - 1) * rho)


def avg_pairwise_correlation(panel: dict[str, pd.DataFrame],
                             window: int = 63) -> tuple[float, pd.DataFrame]:
    """Average pairwise return correlation across the universe (recent window)."""
    rets = {}
    for tkr, df in panel.items():
        if df is None or df.empty:
            continue
        r = np.log(df["close"]).diff().dropna().tail(window)
        if len(r) >= 20:
            rets[tkr] = r
    if len(rets) < 2:
        return np.nan, pd.DataFrame()
    mat = pd.DataFrame(rets).dropna(how="any")
    corr = mat.corr()
    n = corr.shape[0]
    off_diag = (corr.values.sum() - n) / (n * (n - 1))
    return float(off_diag), corr
