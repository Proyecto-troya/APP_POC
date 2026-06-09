"""
app.py — Agentic AlgoTrading System · Financial Behavior Dashboard
==================================================================
Streamlit front-end that operationalizes the "Financial Behavior &
Economic Logic" document with live Polygon.io data.

Run:
    pip install -r requirements.txt
    export POLYGON_API_KEY="your_key"        # or paste it in the sidebar
    streamlit run app.py

Honesty banner (from the doc): this dashboard surfaces the signals that
survive testing and the risk math that keeps a book solvent. It is an
analytical tool, not a trade recommendation engine.
"""

from __future__ import annotations

import os

import numpy as np
import pandas as pd
import streamlit as st

import features as F
import models as M
import polygon_client as pc

st.set_page_config(page_title="Agentic AlgoTrading — Behavior Dashboard",
                   page_icon="📈", layout="wide")

# ----------------------------------------------------------------------
# Cached Polygon fetchers (cache = defense against free-tier rate limits)
# ----------------------------------------------------------------------

@st.cache_data(ttl=900, show_spinner=False)
def cached_aggs(ticker: str, horizon: str, api_key: str) -> pd.DataFrame:
    return pc.get_aggregates(ticker, horizon, api_key)


@st.cache_data(ttl=86400, show_spinner=False)
def cached_details(ticker: str, api_key: str) -> dict:
    try:
        return pc.get_ticker_details(ticker, api_key)
    except pc.PolygonError:
        return {"ticker": ticker, "sector": "Unknown", "name": ticker}


@st.cache_data(ttl=300, show_spinner=False)
def cached_snapshot(ticker: str, api_key: str) -> dict:
    return pc.get_snapshot(ticker, api_key)


@st.cache_data(ttl=900, show_spinner=False)
def cached_news(ticker: str, api_key: str) -> pd.DataFrame:
    return pc.get_news(ticker, api_key)


# ----------------------------------------------------------------------
# Sidebar — configuration & the data trigger
# ----------------------------------------------------------------------

st.sidebar.title("⚙️ Configuration")

api_key = st.sidebar.text_input(
    "Polygon.io API key",
    value=os.environ.get("POLYGON_API_KEY", ""),
    type="password",
    help="Get one at polygon.io. Free tier ≈ 5 requests/min — keep the universe small.",
)

horizon = st.sidebar.selectbox(
    "Sleeve horizon", ["1D", "4H", "1W"], index=0,
    help="4H is most contaminated by microstructure noise (per the doc); "
         "1D/1W are the more reliable sleeves.")

default_universe = "AAPL, MSFT, NVDA, AMZN, GOOGL, META, JPM, XOM, UNH, CAT"
universe_text = st.sidebar.text_area(
    "Universe (comma-separated tickers)", value=default_universe, height=90,
    help="Breadth drives the screener's IR (Grinold), but the free tier "
         "rate-limits — 8–12 names is a practical ceiling per refresh.")

tickers = [t.strip().upper() for t in universe_text.split(",") if t.strip()]

mom_lb = st.sidebar.slider("Momentum lookback (bars)", 20, 252, 126, step=2)
rev_lb = st.sidebar.slider("Reversal lookback (bars)", 2, 21, 5)

fetch = st.sidebar.button("🔄 Fetch data from Polygon", type="primary",
                          use_container_width=True)

st.sidebar.caption(
    "Data is cached 15 min to respect rate limits. Polygon OHLCV is the "
    "**substrate**, not an alpha source — the edge, if any, is in the "
    "cross-sectional processing.")

# ----------------------------------------------------------------------
# Header
# ----------------------------------------------------------------------

st.title("📈 Agentic AlgoTrading System — Financial Behavior Dashboard")
st.caption(
    "Operationalizes the *Financial Behavior & Economic Logic* doc: "
    "the three surviving technical signals, the sector-neutral screener "
    "(durable core), the regime risk dial, ATR stops, fractional Kelly, "
    "and effective breadth. **Analytical tool — not investment advice.**")

if not api_key:
    st.info("👈 Enter your Polygon.io API key in the sidebar, then click "
            "**Fetch data from Polygon**.")
    st.stop()

# ----------------------------------------------------------------------
# Data trigger — pull the panel from Polygon
# ----------------------------------------------------------------------

if fetch or "panel" not in st.session_state:
    if not fetch and "panel" not in st.session_state:
        st.info("Click **Fetch data from Polygon** in the sidebar to load the universe.")
        st.stop()

    panel: dict[str, pd.DataFrame] = {}
    sectors: dict[str, str] = {}
    errors: list[str] = []

    progress = st.progress(0.0, text="Fetching aggregates from Polygon…")
    for i, tkr in enumerate(tickers):
        try:
            df = cached_aggs(tkr, horizon, api_key)
            if df.empty:
                errors.append(f"{tkr}: no bars returned")
            else:
                panel[tkr] = df
                sectors[tkr] = cached_details(tkr, api_key).get("sector", "Unknown")
        except pc.PolygonError as e:
            errors.append(f"{tkr}: {e}")
        progress.progress((i + 1) / len(tickers),
                          text=f"Fetched {tkr} ({i + 1}/{len(tickers)})")
    progress.empty()

    st.session_state["panel"] = panel
    st.session_state["sectors"] = sectors
    st.session_state["horizon_loaded"] = horizon
    if errors:
        st.warning("Some fetches failed:\n\n" + "\n".join(f"- {e}" for e in errors))

panel = st.session_state.get("panel", {})
sectors = st.session_state.get("sectors", {})
loaded_horizon = st.session_state.get("horizon_loaded", horizon)

if not panel:
    st.error("No data loaded. Check your API key, tickers, or rate limits, "
             "then click **Fetch data from Polygon** again.")
    st.stop()

st.success(f"Loaded {len(panel)} tickers at the **{loaded_horizon}** horizon.")

# ----------------------------------------------------------------------
# Tabs
# ----------------------------------------------------------------------

tab_screen, tab_ai, tab_name, tab_regime, tab_risk, tab_breadth = st.tabs([
    "🏆 Screener (durable core)",
    "🤖 AI Predictions",
    "🔍 Single-name signals",
    "🌡️ Regime dial",
    "🛡️ Risk & sizing",
    "🧮 Effective breadth",
])

# ------------------------- Screener ----------------------------------
with tab_screen:
    st.subheader("Sector-neutral cross-sectional ranking")
    st.markdown(
        "Per the doc, this is the system's **most durable, capacity-rich** "
        "alpha layer (Grinold: IR ≈ IC × √breadth). Z-scores are demeaned "
        "within sector so the residual ranking is idiosyncratic selection, "
        "not an accidental sector bet.")

    scores = F.cross_sectional_scores(panel, sectors, loaded_horizon,
                                      mom_lb=mom_lb, rev_lb=rev_lb)
    if scores.empty:
        st.warning("Not enough data to rank.")
    else:
        show = scores[["rank", "ticker", "sector", "last", "momentum",
                       "reversal", "vol_ann", "composite"]].copy()
        show["momentum"] = (show["momentum"] * 100).round(2)
        show["reversal"] = (show["reversal"] * 100).round(2)
        show["vol_ann"] = (show["vol_ann"] * 100).round(1)
        show["composite"] = show["composite"].round(3)
        show.columns = ["Rank", "Ticker", "Sector", "Last $", "Momentum %",
                        "Reversal %", "Vol % (ann)", "Composite z"]
        st.dataframe(show, use_container_width=True, hide_index=True)

        st.bar_chart(scores.set_index("ticker")["composite"])
        st.caption(
            "⚠️ Failure mode (per doc): breadth is illusory if names are "
            "correlated — check the **Effective breadth** tab before "
            "trusting this ranking's diversification.")

# ------------------------ AI Predictions ------------------------------
with tab_ai:
    st.subheader("Signal Engine — ML conditional return forecaster")
    st.markdown(
        "Gradient-boosted stand-in for the system's BiLSTM-Transformer "
        "Signal Engine. Trained **walk-forward with an embargo gap** (no "
        "look-ahead, purged per López de Prado). Per the doc: expect "
        "**52–55% hit rates and tiny R² — that is success, not failure** "
        "(Gu-Kelly-Xiu 2020); the money comes from conviction-weighted "
        "sizing, not frequency.")

    k_fwd = st.slider("Prediction horizon (bars ahead)", 1, 21, 5,
                      help="Target = forward k-bar return at the loaded sleeve horizon.")
    if st.button("🧠 Train model & predict", type="primary"):
        with st.spinner("Walk-forward training across the universe…"):
            st.session_state["ai_result"] = M.train_and_predict(panel, k_forward=k_fwd)

    res = st.session_state.get("ai_result")
    if res and not res["predictions"].empty:
        met = res["metrics"]
        if met:
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("OOS rank IC", f"{met['oos_rank_ic']:.3f}",
                      help="Spearman correlation of predictions vs realized "
                           "forward returns, out of sample. 0.02–0.05 is "
                           "respectable; via IR ≈ IC·√breadth it compounds.")
            c2.metric("OOS hit rate", f"{met['oos_hit_rate'] * 100:.1f}%",
                      help="Directional accuracy out of sample. 52–56% is "
                           "genuinely excellent per the doc.")
            c3.metric("OOS R²", f"{met['oos_r2'] * 100:.2f}%",
                      help="Gu-Kelly-Xiu: ~0.3–0.7% monthly OOS R² is a triumph.")
            c4.metric("Test observations", f"{met['n_test']:,}")
            if met["oos_rank_ic"] <= 0:
                st.error("⚠️ Non-positive out-of-sample IC: the model shows "
                         "NO measured edge on this universe/horizon. Per the "
                         "doc's discipline, do not size positions from these "
                         "predictions.")
        else:
            st.warning("Test window too small for honest OOS metrics — "
                       "widen the universe or use a longer-history horizon.")

        preds = res["predictions"].copy()
        preds["implied_win_prob"] = preds["conviction"].apply(
            lambda c: M.implied_win_prob(c, met.get("oos_hit_rate") if met else None))
        show = preds.copy()
        show["pred_fwd_return"] = (show["pred_fwd_return"] * 100).round(2)
        show["implied_win_prob"] = (show["implied_win_prob"] * 100).round(1)
        show.columns = ["Ticker", "Pred fwd return %", "Conviction [-1,1]",
                        "Direction", "Implied win prob % (capped 56)"]
        st.dataframe(show, use_container_width=True, hide_index=True)
        st.caption("Feed the implied win probability into the **Risk & sizing** "
                   "tab's Kelly calculator — it is conservatively capped so an "
                   "overconfident model cannot induce overbetting.")

        with st.expander("Feature importances (what the model leans on)"):
            st.bar_chart(res["importances"])
            st.caption("⚠️ Failure mode: a model trained in one regime learns "
                       "'buy dips' and is destroyed by the first sustained "
                       "bear. Retrain after regime shifts; distrust stability.")

        with st.expander("🤖 Optional: Claude Bull/Bear analyst (Layer 4 debate)"):
            st.markdown(
                "Adversarial debate grounded ONLY in the computed numbers — "
                "no fabricated catalysts. **Caveat from the doc:** two prompts "
                "to one model are not two independent analysts; treat this as "
                "a de-biasing aid, never independent confirmation.")
            anth_key = st.text_input("Anthropic API key", type="password")
            sel_ai = st.selectbox("Name to debate", preds["ticker"].tolist())
            if st.button("Run Bull/Bear debate") and anth_key:
                row = preds[preds["ticker"] == sel_ai].iloc[0]
                dfx = panel[sel_ai]
                ctx = {
                    "horizon": loaded_horizon,
                    "pred_fwd_return_pct": round(row["pred_fwd_return"] * 100, 2),
                    "conviction": row["conviction"],
                    "momentum_pct": round(F.ts_momentum(dfx["close"],
                                          min(mom_lb, len(dfx) - 2)) * 100, 2),
                    "realized_vol_ann_pct": round(F.realized_vol(
                        dfx["close"], 21, loaded_horizon) * 100, 1),
                    "regime": F.classify_regime(dfx["close"], loaded_horizon)["regime"],
                    "oos_model_hit_rate": met.get("oos_hit_rate") if met else None,
                }
                try:
                    with st.spinner("Running adversarial debate…"):
                        debate = M.llm_bull_bear(sel_ai, ctx, anth_key)
                    st.markdown(f"**🐂 Bull case:** {debate['bull']}")
                    st.markdown(f"**🐻 Bear case:** {debate['bear']}")
                    st.info(f"**Calibration:** {debate['calibration']}")
                except Exception as e:
                    st.error(f"Debate failed: {e}")
    else:
        st.info("Click **Train model & predict** to fit the forecaster on the "
                "loaded universe.")

# ------------------------ Single name --------------------------------
with tab_name:
    sel = st.selectbox("Ticker", sorted(panel.keys()))
    df = panel[sel]
    close = df["close"]

    c1, c2, c3, c4 = st.columns(4)
    mom = F.ts_momentum(close, min(mom_lb, len(close) - 2))
    rev = F.short_term_reversal(close, rev_lb)
    vol = F.realized_vol(close, 21, loaded_horizon)
    atr_now = F.atr(df).iloc[-1]

    c1.metric("Last", f"${close.iloc[-1]:,.2f}")
    c2.metric(f"TS Momentum ({mom_lb} bars)", f"{mom * 100:+.2f}%")
    c3.metric(f"ST Reversal ({rev_lb} bars)", f"{rev * 100:+.2f}%",
              help="Positive = recent sell-off = liquidity-provision long "
                   "candidate. A fee for risk-bearing, not a forecast.")
    c4.metric("Realized vol (21-bar, ann.)", f"{vol * 100:.1f}%")

    st.line_chart(close.rename("close"))

    with st.expander("ATR (volatility clustering — sizing input, not direction)"):
        st.line_chart(F.atr(df).rename("ATR(14)"))

    with st.expander("Recent headlines (context only — interpretation edge, not speed edge)"):
        try:
            news = cached_news(sel, api_key)
            if news.empty:
                st.write("No recent headlines.")
            else:
                for _, row in news.iterrows():
                    st.markdown(f"- **{row['published']}** · "
                                f"[{row['title']}]({row['url']}) — {row['publisher']}")
        except pc.PolygonError as e:
            st.warning(f"News fetch failed: {e}")

# -------------------------- Regime -----------------------------------
with tab_regime:
    st.subheader("Vol/trend regime dial (risk scaling, not return timing)")
    st.markdown(
        "Proxy for the 4-state HMM. Doc mandate: treat the output as a "
        "**'how much risk is the regime willing to pay for' dial** "
        "(Moreira-Muir 2017) — never as a directional forecast. "
        "Expect it to lag turning points; that lag is the known cost.")

    bench = st.selectbox("Reference name for regime",
                         sorted(panel.keys()),
                         help="Ideally a broad ETF like SPY — add it to the universe.")
    reg = F.classify_regime(panel[bench]["close"], loaded_horizon)

    c1, c2, c3 = st.columns(3)
    c1.metric("Regime", reg["regime"])
    c2.metric("Suggested gross-exposure multiplier", f"{reg['exposure_mult']:.0%}")
    if pd.notna(reg.get("vol_ratio", np.nan)):
        c3.metric("Vol ratio (21-bar / 126-bar)", f"{reg['vol_ratio']:.2f}",
                  help=">1.2 flags an elevated-vol state")

    st.caption("⚠️ Failure mode: regimes are only confidently labeled "
               "ex-post. The classifier will be most wrong at turning "
               "points — exactly the most expensive moments.")

# ------------------------ Risk & sizing -------------------------------
with tab_risk:
    st.subheader("ATR stops & fractional Kelly")
    sel_r = st.selectbox("Position ticker", sorted(panel.keys()), key="risk_tkr")
    dfr = panel[sel_r]
    last = float(dfr["close"].iloc[-1])
    atr_v = float(F.atr(dfr).iloc[-1])

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**ATR stop framework** — stops *reshape the "
                    "distribution* (cut the left tail); they do **not** add "
                    "return (Kaminski-Lo 2014).")
        mult = st.slider("ATR multiple", 1.0, 5.0, 2.5, 0.25)
        stops = F.atr_stop_levels(last, atr_v, mult)
        st.metric("Last price", f"${last:,.2f}")
        st.metric("Long stop", f"${stops['stop_long']:,.2f}")
        st.metric("Short stop", f"${stops['stop_short']:,.2f}")
        st.caption(f"ATR = ${atr_v:,.2f} ({stops['atr_pct'] * 100:.2f}% of price). "
                   "⚠️ Clustered stops at obvious ATR levels get hunted; "
                   "gaps fill below the stop level.")

    with col_b:
        st.markdown("**Fractional Kelly** — the binding constraint is not "
                    "your edge but your *uncertainty about your edge*. "
                    "0.5× Kelly keeps ~75% of optimal growth at half the variance.")
        p = st.slider("Estimated win probability", 0.40, 0.70, 0.54, 0.01,
                      help="The doc: 52–56% hit rate is genuinely excellent.")
        b = st.slider("Win/loss payoff ratio", 0.5, 3.0, 1.3, 0.05)
        frac = st.slider("Kelly fraction", 0.10, 0.50, 0.35, 0.05)
        unc = st.slider("Edge-uncertainty shrinkage", 0.0, 0.9, 0.5, 0.05,
                        help="Extra shrinkage as estimation error rises / "
                             "regime confidence falls.")
        k = F.fractional_kelly(p, b, frac, unc)
        st.metric("Full Kelly f*", f"{k['full_kelly'] * 100:.1f}% of capital")
        st.metric("Applied size (after both shrinkages)",
                  f"{k['applied_fraction'] * 100:.2f}% of capital")
        st.caption("⚠️ Kelly assumes independent bets — correlated positions "
                   "secretly compound far above the intended fraction.")

# ------------------------- Breadth ------------------------------------
with tab_breadth:
    st.subheader("Effective breadth — the diversification illusion check")
    st.markdown(
        "Effective independent bets = **N / (1 + (N−1)·ρ)**. "
        "At ρ=0.3 across 1,000 names, effective breadth is ~3, not 1,000. "
        "Size the book for the **stressed** correlation, not the average one.")

    rho, corr = F.avg_pairwise_correlation(panel)
    n = len(panel)
    if np.isnan(rho):
        st.warning("Need ≥2 names with overlapping return history.")
    else:
        eb = F.effective_breadth(n, rho)
        eb_stress = F.effective_breadth(n, min(rho + 0.35, 0.95))
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Nominal N", n)
        c2.metric("Avg pairwise ρ (63 bars)", f"{rho:.2f}")
        c3.metric("Effective breadth (today)", f"{eb:.1f}")
        c4.metric("Effective breadth (stress, ρ+0.35)", f"{eb_stress:.1f}")
        st.dataframe(corr.round(2), use_container_width=True)
        st.caption("⚠️ Diversification is abundant when you don't need it "
                   "and scarce when you do (Part II, #6).")

st.divider()
st.caption("Built from the *Financial Behavior & Economic Logic* companion "
           "doc · Polygon.io data is the substrate, not the alpha · "
           "Treat every backtested number with the contempt it deserves.")
