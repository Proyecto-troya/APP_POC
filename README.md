# Agentic AlgoTrading System — Financial Behavior Dashboard

Streamlit app operationalizing the *Financial Behavior & Economic Logic* (v1.0)
companion document with live Polygon.io market data.

## What it implements (mapped to the doc)

| Tab | Doc concept |
|---|---|
| Screener | Layer 3 — sector-neutral cross-sectional ranking (Grinold breadth; the durable, capacity-rich core) |
| Single-name signals | Layer 2 — the three surviving technicals: TS momentum, short-term reversal, vol clustering (ATR) |
| Regime dial | Layer 2 — 4-state vol/trend regime proxy used for *risk scaling*, never timing (Moreira-Muir) |
| Risk & sizing | Layers 5–6 — ATR stop framework, fractional Kelly with edge-uncertainty shrinkage |
| Effective breadth | Part II #6 — N / (1 + (N−1)ρ), incl. stressed-correlation view |

## Setup

```bash
pip install -r requirements.txt
export POLYGON_API_KEY="your_key"   # or paste it in the sidebar
streamlit run app.py
```

## Polygon usage notes

- All data comes from REST endpoints: `/v2/aggs` (4H / 1D / 1W bars),
  `/v3/reference/tickers`, `/v2/snapshot`, `/v2/reference/news`.
- Free tier ≈ 5 requests/min. The app caches aggregates for 15 minutes and
  retries 429s with backoff — keep the universe to ~8–12 names per refresh.
- 4H bars are filtered to 08:00–20:00 ET because (per the doc) the 4H sleeve
  is the most contaminated by pre/post-market microstructure noise.

## Disclaimer

Analytical tooling derived from the design document. Not investment advice.
