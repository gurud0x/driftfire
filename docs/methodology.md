# Methodology

This document provides a concise description of the data and methods used
in DriftFire v1.

Data
- Source: public data via `yfinance` and curated GICS sector mapping.
- Raw data are downloaded and stored locally per `notebooks/01_data_download.ipynb`.
- The canonical cleaned dataset is `data/processed/clean.parquet`.

Feature engineering
- EMAs (8, 21, 50) are computed per-ticker.
- Volume-based features include multi-day decline tests and a 20-day
  average-based `vol_ratio` spike measure.
- Sector momentum is computed on SPDR sector ETFs and used to rank sectors
  on short/medium/long horizons.

Signal construction
- Market regime filter (QQQ EMA alignment) is applied before per-stock
  entry logic.
- Entry requires prior-day EMA alignment, a multi-day low-volume consolidation
  pattern, and a same-day volume breakout above a threshold.
- Exits occur on close below EMA8 or after a maximum holding period.

Backtesting
- Trades are simulated using next-day open entries and close-based exits.
- Position-sizing is fixed per configuration and transaction costs are
  applied round-trip (configurable in notebooks).
- Walk-forward experiments and parameter optimization are performed and
  summarized in `outputs/tables/`.

Reproducibility
- Notebooks are annotated with execution order. To reproduce results,
  create a Python 3.9.6 environment, run the notebooks in order, and
  ensure `data/processed/` contains the required downloads.
