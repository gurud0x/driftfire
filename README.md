# DriftFire

**Quantitative equity signal research — EMA consolidation + volume breakout strategy**

Indiana University | Python | April 2026

---

## What This Is

DriftFire is a reproducible analytics project that documents a rule-based
EMA + volume-breakout signal and its historical evaluation. The goal is to
present a clear, auditable data pipeline, feature engineering, QA checks, and
validation results suitable for a data analytics / corporate technology
internship portfolio.

The research question: does a combination of market regime filtering,
sector rotation ranking, and volume contraction / expansion signals produce
measurable forward-return behavior in U.S. equities (2018–2024)?

---

## Strategy Logic (high level)

The implemented strategy is rule-based and intentionally simple; key gates
are:

1. Market regime: QQQ must be above EMA8, EMA21 and EMA50 on the prior day.
2. Sector rotation: trade stocks whose sector ETF ranks among the top-3 by
   composite momentum across short/medium/long windows.
3. Trend alignment: per-stock EMA8 > EMA21 > EMA50 (yesterday's values).
4. Volume pattern: multi-day contraction near the trend followed by a
   breakout day with volume expansion measured by a `vol_ratio` metric.
5. Entries occur at next-day open; exits occur on close below EMA8 or after
   a configured maximum holding period.

This repository documents the implementation and evaluation; strategy
parameters and thresholds are configurable in the notebooks.

---

## Results

- These results are research outputs, not live-trading claims.
- Performance is sensitive to data quality, execution assumptions,
  transaction costs, and parameter choices.

---

## Project Structure

- `notebooks/` — ordered Jupyter notebooks that perform data download,
  cleaning, feature engineering, backtesting, robustness checks and
  optimization. Notebooks are intended to be executed in prefix order.
- `src/` — small Python package containing reusable functions: `signals.py`,
  `backtest.py`, and `report_utils.py` used by the notebooks and quick tests.
- `data/processed/` — local processed datasets referenced by notebooks
  (`clean.parquet`, `features.parquet`) — **excluded from GitHub**.
- `outputs/` — curated public artifacts (charts, tables and a short report).
- `docs/` — supplemental methodology, limitations, and conservative future
  work notes for reviewers.
- `requirements.txt` — python dependencies for a reproducible dev environment.

## Installation

This repository is prepared as a reproducible analytics project for Python
3.9.6. Recommended steps (macOS / Linux):

```bash
python3.9 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Use the exact Python minor version if you want bit-for-bit reproducibility.

## Notebook Execution Order

Run notebooks in the following sequence to reproduce analysis artifacts locally.
Stop and inspect intermediate outputs after each step; heavy notebooks may take
significant time and require local data downloads.

1. `notebooks/01_data_download.ipynb` — stage and download raw data, then
   create the first cleaned dataset.
2. `notebooks/02_cleaning_qa.ipynb` — validate the cleaned dataset and create
   `data/processed/clean.parquet`.
3. `notebooks/03_feature_engineering.ipynb` — create features and signals and
   produce `data/processed/features.parquet`.
4. `notebooks/04_backtest.ipynb` — core backtests and aggregated metric tables.
5. `notebooks/05_robustness.ipynb` and `notebooks/06_optimization.ipynb` —
   robustness analyses and parameter tuning/Optuna experiments.

Note: only execute `03_feature_engineering.ipynb` after
`data/processed/clean.parquet` exists locally.

## Data Notes

- Raw and processed data files are intentionally excluded from the GitHub
  repository for size and privacy reasons. The canonical cleaned dataset is
  `data/processed/clean.parquet` and the engineered features file is
  `data/processed/features.parquet`.
- To reproduce results locally, run `notebooks/01_data_download.ipynb` to
  acquire raw data, then `02_cleaning_qa.ipynb` to create `clean.parquet`.
- All downstream notebooks expect these local files; the `outputs/` folder
  contains curated artifacts so reviewers can inspect results without
  re-running the pipeline.

## Output Artifacts

Public, curated artifacts live in `outputs/` for fast review:

- `outputs/tables/` contains cleaned CSV summaries used in the report
  (examples: `backtest_comparison.csv`, `final_comparison.csv`,
  `walk_forward_results.csv`, `sector_map.csv`).
- `outputs/charts/` contains publication-ready figures (PNG/SVG/PDF).
- `outputs/report/` contains a short static summary (markdown/HTML/PDF).

These curated artifacts are small and safe to include in the public repo so
that hiring reviewers can evaluate the core results without re-running heavy
computations.

## Methodology Docs

Detailed methods are summarized in `docs/methodology.md`. Key points:

- Emphasis on data pipeline hygiene: explicit download, cleaning, and QA
  steps with checks for NA values, symbol continuity, and sector mapping
  consistency.
- Feature engineering includes multiple EMAs per ticker, per-ticker
  normalization of volume, rolling average volume, and multi-day
  contraction/expansion tests that produce `vol_ratio` style signals.
- Sector rotation ranking is computed on SPDR sector ETFs and used as a
  market-level filter.
- Backtests use next-day open entries, close-based exits, explicit
  transaction-cost assumptions, and walk-forward validation to reduce
  overfitting risk.

## Limitations

- This repository demonstrates research outputs, not live-trading signals.
  Do not interpret backtest returns as real-world performance guarantees.
- Execution assumptions (next-open fills, exit-at-close) are simplifying
  and do not model intraday slippage or partial fills.
- Historical data may contain survivorship or selection biases; readers
  should review `docs/limitations.md` for details.

## Future Work (conservative)

- Improve data-source reliability and reproducible download checks.
- Expand robustness testing with additional walk-forward windows and
  parameter-sensitivity analyses.
- Produce cleaner, one-page stakeholder reports (static HTML or PDF) and a
  basic dashboard for interactive exploration of results.
- Compare the strategy to simple baseline portfolios in a formal model
  comparison framework.

----
