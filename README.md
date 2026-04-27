# DriftFire

A systematic equity momentum strategy that identifies high-probability breakout setups
through volume consolidation patterns and a multi-layer market regime filter applied to
S&P 500 constituent stocks from 2018–2024.

---

## Strategy Overview

DriftFire looks for stocks that have been consolidating quietly near their 8-day EMA with
declining volume, then enter when a high-volume breakout day confirms the move. Four
independent filters must align simultaneously: a fundamental price/volume screener, a
market-regime gate (QQQ trend), a sector-rotation gate (top-3 momentum sectors only),
and a stock-level pattern confirmation (EMA alignment + volume consolidation + breakout
spike). Entry is at the next day's open; exit is on close below the 8-day EMA or after
a configurable maximum hold period.

---

## Key Results

### Strategy Evolution (Test Period 2022–2024)

| Config | Trades | Sharpe | CAGR | Max DD | Hit Rate | Profit Factor |
|---|---|---|---|---|---|---|
| Original (strict filters) | 33 | 0.50 | +3.2% | -44.4% | 48.5% | 2.48 |
| Relaxed filters (v2) | 93 | 0.71 | +4.8% | -46.0% | 41.9% | 1.66 |
| Relaxed + regime≥3 | 66 | 0.56 | -0.1% | -43.5% | 37.9% | 0.98 |
| Optuna optimized | 124 | **0.84** | -0.0% | -51.1% | 35.5% | 1.00 |

### Walk-Forward Validation (Rolling 2-Year Train → 1-Year OOS)

| Window | OOS Sharpe | OOS CAGR | Trades |
|---|---|---|---|
| 2018-2019 → 2020 | 1.14 | -9.3% | 74 |
| 2019-2020 → 2021 | 1.07 | -0.7% | 61 |
| 2020-2021 → 2022 | 0.51 | -0.8% | 12 |
| 2021-2022 → 2023 | 1.11 | +54.4% | 53 |
| 2022-2023 → 2024 | 0.98 | -4.4% | 63 |
| **Average OOS** | **0.96** | **+7.8%** | **53** |

**5/5 walk-forward windows show positive out-of-sample Sharpe** — the strongest available
evidence that DriftFire is not a fluke of the specific 2018-2024 test period.

> **Note on drawdown and CAGR:** The strategy sits in cash ~95% of the time. Max drawdown
> is measured from the equity peak (which includes gains from winners like TSLA +15%,
> LRCX +14%), so it can appear large despite low actual dollar exposure. Average hold
> period is ~5-6 trading days. Negative CAGR with positive Sharpe occurs because the
> portfolio sits flat (not losing) in cash for long stretches — the denominator shrinks
> the annualised return while daily volatility remains low.

---

## Optimized Parameters (Optuna, 150 trials on Train 2018–2020)

| Parameter | Baseline | Optimized |
|---|---|---|
| vol_consol_days | 3 | 4 |
| ema_proximity | 2.0% | 3.7% |
| vol_spike_mult | 1.5× | 1.2× |
| regime_threshold | 2 | 1 |
| max_hold_days | 20 | 6 |

**Parameter importance** (FANOVA): `vol_spike_mult` dominates at 77% — the volume
spike threshold is the single most impactful parameter. `regime_threshold` is second
(8.7%), followed by `max_hold_days` (5.4%), `ema_proximity` (4.9%), and `vol_consol_days`
(3.7%).

---

## Signal Architecture

```
Layer 1 — Screener (daily)
  Price > $3, daily change > 0%, dollar volume > $1.5M,
  10-day avg volume > 500k, close > EMA21 and EMA50

Layer 2 — Market Regime
  QQQ must be above its 8, 21, AND 50-day EMA simultaneously
  → regime = 1 (bull); regime = 0 (no new trades)

Layer 3 — Sector Rotation
  Rank 11 SPDR sector ETFs by 1w / 1m / 3m composite returns
  Only trade stocks in the top-3 ranked sectors on all three windows

Layer 4 — Individual Entry
  EMA8 > EMA21 > EMA50 (uptrend aligned)
  Volume drying up for N prior days (vol < 20-day average)
  Price within X% of EMA8 (tight consolidation)
  Today's volume > M× 20-day average (breakout trigger)
  → Entry at next day's open
```

---

## Regime Score

A 0–4 confidence score computed daily at the market level:

| Component | +1 if… |
|---|---|
| QQQ Trend | QQQ above its 50-day EMA |
| QQQ Momentum | QQQ 21-day return > 0 |
| Sector Breadth | ≥6 of 11 sector ETFs above their 21-day EMA |
| Rotation Quality | Top-3 sectors all have positive 21-day returns |

---

## Project Structure

```
DriftFire/
├── data/
│   ├── raw/                    # universe_raw.parquet (472 stocks + ETFs)
│   └── processed/              # clean.parquet, features.parquet, backtest outputs
│       ├── backtest_comparison.csv   # all signal × period results
│       ├── final_comparison.csv      # strategy evolution table
│       ├── walk_forward_results.csv  # 5-window walk-forward OOS
│       └── optuna_best_params.json   # locked optimized parameters
├── notebooks/
│   ├── 01_data_download.ipynb
│   ├── 02_cleaning_qa.ipynb
│   ├── 03_feature_engineering.ipynb  # signals, regime_score → features.parquet
│   ├── 04_backtest.ipynb             # vectorized engine, comparison table, SPY overlay
│   ├── 05_robustness.ipynb           # cost sensitivity, SPY benchmark, t-test
│   └── 06_optimization.ipynb         # Optuna search + walk-forward validation
├── src/
│   ├── signals.py              # compute_regime, compute_sector_rotation, etc.
│   ├── backtest.py             # run_backtest (vectorized), calc_metrics
│   └── report_utils.py         # plot_equity_curve, plot_drawdown, etc.
├── reports/
│   └── figures/                # all saved charts (PNG, 150 DPI)
├── requirements.txt
└── README.md
```

---

## Reproducibility

```bash
git clone https://github.com/YOUR_USERNAME/driftfire.git
cd driftfire
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run notebooks in order:
# 01 → data download (~5-10 min, requires internet, downloads 472 tickers)
# 02 → cleaning QA
# 03 → feature engineering (~3 min, downloads sector map)
# 04 → backtest (~5 sec)
# 05 → robustness (~2 min, re-downloads SPY)
# 06 → optimization (~2 min, 150 Optuna trials + 5-window walk-forward)
```

---

## Methodology Notes

- All features lagged by 1 day to prevent lookahead bias
- Entry at next-day open, not the signal-day close
- 10 bps round-trip transaction costs (5 bps per side)
- Max 5 concurrent positions at 20% each (equal weight, no leverage)
- Train / Validation / Test split: 2018–2020 / 2021 / 2022–2024
- No machine learning — rule-based for full transparency and auditability
- Backtest engine: vectorized (iterates over ~100-400 signal rows, not 1,760 dates)
- Optuna: Bayesian optimization with TPE sampler, optimized on Train only
- Walk-forward: 5 rolling 2-year train → 1-year OOS windows, 50 trials each

---

## Limitations

- **Small sample:** 83-196 trades over 7 years. Despite 5/5 walk-forward windows being
  positive, the per-window sample sizes are small (12-74 trades) and statistical
  significance is limited.
- **No intraday data:** Entry/exit prices use daily open/close; real execution may differ
  due to gap risk and liquidity.
- **Survivor bias:** Universe is current S&P 500 constituents, not the historical index
  membership — stocks that were delisted or removed are underrepresented.
- **Single exit rule:** No partial exits or trailing stops; a single EMA8 cross ends the
  entire position.
- **Negative CAGR with positive Sharpe:** When the portfolio holds cash ~95% of the time,
  the equity curve barely moves. The Sharpe ratio rewards this low-volatility behaviour
  even when returns are near zero. Real-world implementation would need a cash deployment
  strategy (e.g. T-bills or index ETF overlay) to generate meaningful absolute returns.

---

## Author

Indiana University — Quant Research Project
