# DriftFire

**Quantitative equity signal research — EMA consolidation + volume breakout strategy**

Indiana University | Python | April 2026

---

## What This Is

DriftFire is a systematic backtest of a discretionary swing trading strategy.
The research question: does a combination of market regime filtering, sector 
rotation ranking, and institutional accumulation signals produce statistically 
significant forward returns in US equities from 2018–2024?

---

## Strategy Logic

The signal fires only when all of the following are true simultaneously:

1. **Market regime** — QQQ is above its 8, 21, and 50 EMAs (bull mode only)
2. **Sector rotation** — stock's sector ranks top 3 on 1-week, 1-month, and 3-month rolling returns across 11 SPDR sector ETFs
3. **Stock trend** — EMA8 > EMA21 > EMA50 on the individual stock
4. **Volume consolidation** — volume declining for 5 consecutive days near the 8 EMA (institutional accumulation pattern)
5. **Breakout confirmation** — today's volume exceeds 1.5x the 20-day average (Setup A) or 3x average (Setup B VCP base breakout)
6. **Exit** — daily close below 8 EMA, or 20-day maximum hold

### Two Entry Setups Tested

**Setup A — EMA Pullback:** Stock in established uptrend consolidates on declining volume near the 8 EMA, then breaks out on volume spike.

**Setup B — VCP Base Breakout:** Stock forms a tight volatility contraction pattern (Minervini VCP) after a downtrend, then explodes on massive volume with a strong close near the high of the candle.

---

## Results

- Walk-forward validation: 5/5 out-of-sample windows positive
- Average OOS Sharpe: 0.96
- CAGR: 7.8%
- All results include 10 basis points round-trip transaction costs

---

## Tech Stack

- **Data:** yfinance, finviz, Cap IQ via Indiana University (GICS classifications)
- **Processing:** pandas, numpy
- **Optimization:** Optuna (Bayesian hyperparameter search)
- **Statistics:** scipy, statsmodels
- **Visualization:** matplotlib, seaborn
- **Storage:** parquet via pyarrow
- **Environment:** Python 3.9.6, Jupyter, VSCode

---

## Project Structure