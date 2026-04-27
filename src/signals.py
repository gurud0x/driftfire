"""Signal computation functions for DriftFire strategy."""

import pandas as pd
import numpy as np
from typing import Optional

GICS_TO_ETF = {
    'Information Technology': 'XLK', 'Technology': 'XLK',
    'Communication Services': 'XLC',
    'Energy': 'XLE',
    'Financials': 'XLF', 'Financial Services': 'XLF',
    'Health Care': 'XLV', 'Healthcare': 'XLV',
    'Industrials': 'XLI',
    'Consumer Staples': 'XLP', 'Consumer Defensive': 'XLP',
    'Consumer Discretionary': 'XLY', 'Consumer Cyclical': 'XLY',
    'Real Estate': 'XLRE',
    'Materials': 'XLB', 'Basic Materials': 'XLB',
    'Utilities': 'XLU',
}

SECTOR_ETFS = ['XLK', 'XLC', 'XLE', 'XLF', 'XLV', 'XLI', 'XLP', 'XLY', 'XLRE', 'XLB', 'XLU']


def _check_columns(df: pd.DataFrame, required: list[str], func_name: str) -> None:
    """Raise ValueError if any required columns are missing.

    Args:
        df: DataFrame to validate.
        required: List of column names that must be present.
        func_name: Name of the calling function (for error messages).

    Raises:
        ValueError: If one or more required columns are absent.
    """
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"[{func_name}] Missing required columns: {missing}. "
            f"Available columns: {list(df.columns)}"
        )


def compute_ema(series: pd.Series, span: int) -> pd.Series:
    """Compute exponential moving average per ticker group.

    Expects *series* to have a two-level MultiIndex (date, ticker) **or** to be
    a plain Series that has already been sliced to a single ticker.  When the
    index contains a 'ticker' level the EMA is computed independently for each
    ticker so that no look-ahead leakage occurs across symbols.

    Args:
        series: Price series.  Either a plain Series or one with a MultiIndex
            whose second level is named 'ticker'.
        span: EMA span (e.g. 8, 21, 50).

    Returns:
        EMA series with the same index as *series*.

    Raises:
        ValueError: If *span* is not a positive integer.
    """
    if span <= 0:
        raise ValueError(f"span must be a positive integer, got {span}.")

    if isinstance(series.index, pd.MultiIndex) and 'ticker' in series.index.names:
        ticker_level = series.index.names.index('ticker')
        return (
            series.groupby(level=ticker_level)
            .transform(lambda s: s.ewm(span=span, adjust=False).mean())
        )

    # Plain Series — single ticker or already grouped externally.
    return series.ewm(span=span, adjust=False).mean()


def compute_regime(df: pd.DataFrame) -> pd.Series:
    """Compute market regime from QQQ price data.

    Extracts QQQ rows, computes three EMAs (8, 21, 50), then returns a
    date-indexed boolean Series indicating whether QQQ closed above all three
    EMAs on the *previous* day (shift(1) so the signal is tradeable today).

    Args:
        df: Full feature DataFrame containing at least the columns
            ``['date', 'ticker', 'close']``.  QQQ rows are identified by
            ``ticker == 'QQQ'``.

    Returns:
        Boolean Series indexed by date.  ``True`` means regime is bullish
        (QQQ > ema8 AND ema21 AND ema50 as of yesterday's close).

    Raises:
        ValueError: If required columns are missing or QQQ has no rows.
    """
    required = ['date', 'ticker', 'close']
    _check_columns(df, required, 'compute_regime')

    qqq = df[df['ticker'] == 'QQQ'].copy()
    if qqq.empty:
        raise ValueError("compute_regime: no rows found for ticker='QQQ'.")

    qqq = qqq.sort_values('date').set_index('date')
    close = qqq['close']

    ema8 = close.ewm(span=8, adjust=False).mean()
    ema21 = close.ewm(span=21, adjust=False).mean()
    ema50 = close.ewm(span=50, adjust=False).mean()

    above_all = (close > ema8) & (close > ema21) & (close > ema50)

    # Shift by 1: today's regime signal is based on yesterday's close.
    regime = above_all.shift(1).fillna(False).rename('regime')
    return regime


def compute_sector_rotation(
    df: pd.DataFrame,
    top_n: int = 3,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compute sector rotation signals from sector ETF price data.

    Extracts rows for SECTOR_ETFS, computes 5-, 21-, and 63-day rolling
    returns, ranks sectors by an equal-weighted composite of those returns,
    then shifts by 1 day so the signal is known before the trading open.

    Args:
        df: Full feature DataFrame with at least ``['date', 'ticker', 'close']``.
        top_n: Number of top-ranked sectors to flag as ``True``.

    Returns:
        A tuple ``(leading_df, rank1_df)`` where:

        * **leading_df** – DataFrame indexed by date, one column per sector
          ETF, ``True`` if that sector is in the top *top_n* by composite
          momentum (shifted by 1 day).
        * **rank1_df** – DataFrame indexed by date, one column per sector ETF,
          containing the integer rank (1 = best) of each sector (shifted by 1
          day).

    Raises:
        ValueError: If required columns are missing or no sector ETF rows exist.
    """
    required = ['date', 'ticker', 'close']
    _check_columns(df, required, 'compute_sector_rotation')

    sector_df = df[df['ticker'].isin(SECTOR_ETFS)].copy()
    if sector_df.empty:
        raise ValueError(
            "compute_sector_rotation: no rows found for any SECTOR_ETFS ticker."
        )

    # Build a date x ticker pivot of closing prices.
    price_pivot = sector_df.pivot_table(
        index='date', columns='ticker', values='close', aggfunc='last'
    )
    price_pivot = price_pivot.sort_index()

    # Rolling returns (shift so they use data available at close).
    ret5 = price_pivot.pct_change(5)
    ret21 = price_pivot.pct_change(21)
    ret63 = price_pivot.pct_change(63)

    # Equal-weight composite score.
    composite = (ret5 + ret21 + ret63) / 3.0

    # Rank: rank 1 = highest composite score (best momentum).
    ranked = composite.rank(axis=1, ascending=False, method='min')

    # Shift by 1 trading day so today's signal is based on yesterday's data.
    ranked_shifted = ranked.shift(1)
    composite_shifted = composite.shift(1)

    leading_df = ranked_shifted <= top_n
    rank1_df = ranked_shifted

    # Only keep columns that actually exist.
    available_etfs = [e for e in SECTOR_ETFS if e in leading_df.columns]
    leading_df = leading_df[available_etfs]
    rank1_df = rank1_df[available_etfs]

    return leading_df, rank1_df


def compute_entry_signal(stocks_df: pd.DataFrame) -> pd.Series:
    """Compute Layer-4 entry signals for individual stocks.

    Applies all DriftFire entry filters.  Directional / EMA-alignment and
    consolidation filters use *shifted* (yesterday's) values so they are
    tradeable at today's open.  Volume spike is evaluated on today's bar.

    Required columns in *stocks_df* (grouped / filtered to stocks only,
    i.e. QQQ and sector ETFs excluded):

    * ``ema8``, ``ema21``, ``ema50`` – precomputed EMAs
    * ``vol_ratio``  – current volume / 20-day average volume
    * ``pct_from_ema8`` – ``(close - ema8) / ema8``
    * ``screener_pass`` – bool, stock passes fundamental/price screen
    * ``regime`` – bool, market is in regime (precomputed & merged)
    * ``sector_top3`` – bool, stock's sector ETF is in top-3 rotation
    * ``close`` – closing price
    * ``volume`` – raw daily volume
    * ``ticker`` – symbol (used for groupby)
    * ``date`` – trading date

    Args:
        stocks_df: DataFrame for stock tickers only (QQQ & sector ETFs removed).
            Must be sorted by (ticker, date) or at least have those columns.

    Returns:
        Boolean Series with the same index as *stocks_df*.  ``True`` means all
        Layer-4 entry conditions are met and this bar generates an entry signal.

    Raises:
        ValueError: If any required column is missing.
    """
    required = [
        'ema8', 'ema21', 'ema50', 'vol_ratio', 'pct_from_ema8',
        'screener_pass', 'regime', 'sector_top3', 'close', 'volume',
        'ticker', 'date',
    ]
    _check_columns(stocks_df, required, 'compute_entry_signal')

    df = stocks_df.copy()

    # --- Shifted (yesterday's) conditions -------------------------------------------
    # Group by ticker so shifts don't bleed across symbols.
    grp = df.groupby('ticker', sort=False)

    ema_aligned_prev = grp.apply(
        lambda g: (
            g['close'].shift(1) > g['ema8'].shift(1)
        ) & (
            g['ema8'].shift(1) > g['ema21'].shift(1)
        ) & (
            g['ema21'].shift(1) > g['ema50'].shift(1)
        )
    )
    # apply with groupby returns a MultiIndex (ticker, original_index); flatten.
    if isinstance(ema_aligned_prev.index, pd.MultiIndex):
        ema_aligned_prev = ema_aligned_prev.droplevel(0)
    ema_aligned_prev = ema_aligned_prev.reindex(df.index).fillna(False)

    # Tight consolidation: yesterday's |pct_from_ema8| <= 3 %
    tight_consol_prev = (
        grp['pct_from_ema8']
        .shift(1)
        .abs()
        .reindex(df.index)
        <= 0.03
    )

    # Volume drying-up: prior 3 bars all had vol_ratio < 0.8
    vol_dryup = (
        grp['vol_ratio']
        .apply(lambda s: (s.shift(1) < 0.8) & (s.shift(2) < 0.8) & (s.shift(3) < 0.8))
    )
    if isinstance(vol_dryup.index, pd.MultiIndex):
        vol_dryup = vol_dryup.droplevel(0)
    vol_dryup = vol_dryup.reindex(df.index).fillna(False)

    # --- Today's conditions ---------------------------------------------------------
    # Volume spike: today's vol_ratio >= 1.5
    vol_spike = df['vol_ratio'] >= 1.5

    # Market-level gates (already shifted when merged onto stocks_df).
    regime_ok = df['regime'].astype(bool)
    sector_ok = df['sector_top3'].astype(bool)
    screener_ok = df['screener_pass'].astype(bool)

    entry_signal = (
        ema_aligned_prev
        & tight_consol_prev
        & vol_dryup
        & vol_spike
        & regime_ok
        & sector_ok
        & screener_ok
    )

    return entry_signal.rename('entry_signal')


def compute_regime_score(
    df: pd.DataFrame,
    qqq_df: pd.DataFrame,
    top3_all: pd.DataFrame,
) -> pd.Series:
    """Compute the 4-component DriftFire regime score (market-level, 0–4).

    Components:

    1. **QQQ trend** – QQQ close > EMA50 (1 point).
    2. **QQQ momentum** – QQQ 21-day return > 0 (1 point).
    3. **Sector breadth** – average number of sector ETFs above their EMA21
       on a given day >= 6 (1 point).
    4. **Rotation quality** – top-3 sector ETFs (by composite momentum) all
       have positive 21-day returns (1 point).

    All components are shifted by 1 day before summing so the score is
    tradeable on the open of the *current* day.

    Args:
        df: Full feature DataFrame with at least
            ``['date', 'ticker', 'close']``.
        qqq_df: QQQ-only DataFrame (``ticker == 'QQQ'``) with at least
            ``['date', 'close']``.  Can be a subset of *df*.
        top3_all: DataFrame returned as *leading_df* from
            :func:`compute_sector_rotation` — date-indexed, sector ETF
            columns, boolean True/False for top-3.

    Returns:
        Integer Series indexed by date with values 0–4.  Already shifted so
        it can be used directly as a gate (score >= threshold).

    Raises:
        ValueError: If required columns are missing.
    """
    _check_columns(df, ['date', 'ticker', 'close'], 'compute_regime_score')
    _check_columns(qqq_df, ['date', 'close'], 'compute_regime_score (qqq_df)')

    # ---- Component 1 & 2: QQQ trend and momentum ----------------------------------
    qqq = qqq_df.sort_values('date').set_index('date')['close']
    ema50_qqq = qqq.ewm(span=50, adjust=False).mean()
    comp1 = (qqq > ema50_qqq).astype(int)
    comp2 = (qqq.pct_change(21) > 0).astype(int)

    # ---- Component 3: sector breadth -----------------------------------------------
    sector_df = df[df['ticker'].isin(SECTOR_ETFS)].copy()
    if not sector_df.empty:
        price_pivot = sector_df.pivot_table(
            index='date', columns='ticker', values='close', aggfunc='last'
        ).sort_index()
        ema21_sectors = price_pivot.ewm(span=21, adjust=False).mean()
        above_ema21 = (price_pivot > ema21_sectors).sum(axis=1)
        comp3 = (above_ema21 >= 6).astype(int)
    else:
        comp3 = pd.Series(0, index=qqq.index, dtype=int)

    # ---- Component 4: rotation quality ---------------------------------------------
    if not top3_all.empty:
        # Identify which sectors are top-3 on each day.
        sector_prices = sector_df.pivot_table(
            index='date', columns='ticker', values='close', aggfunc='last'
        ).sort_index() if not sector_df.empty else pd.DataFrame()

        if not sector_prices.empty:
            ret21_sectors = sector_prices.pct_change(21)
            # top3_all is already shifted; positive-return check is on raw ret21.
            # We re-align indexes.
            common_idx = top3_all.index.intersection(ret21_sectors.index)
            top3_aligned = top3_all.reindex(common_idx)
            ret21_aligned = ret21_sectors.reindex(common_idx)

            # For days where top3 ETFs all have positive 21d return.
            positive_returns = ret21_aligned > 0
            # Count how many top-3 slots have positive returns.
            top3_positive = (top3_aligned & positive_returns).sum(axis=1)
            top3_count = top3_aligned.sum(axis=1).replace(0, np.nan)
            comp4 = (top3_positive >= top3_count).fillna(False).astype(int)
            comp4 = comp4.reindex(qqq.index, fill_value=0)
        else:
            comp4 = pd.Series(0, index=qqq.index, dtype=int)
    else:
        comp4 = pd.Series(0, index=qqq.index, dtype=int)

    # ---- Combine and shift ---------------------------------------------------------
    # Align all components to a common date index.
    all_idx = qqq.index
    score_raw = (
        comp1.reindex(all_idx, fill_value=0)
        + comp2.reindex(all_idx, fill_value=0)
        + comp3.reindex(all_idx, fill_value=0)
        + comp4.reindex(all_idx, fill_value=0)
    )

    # Shift by 1: today's score is based on yesterday's data.
    score = score_raw.shift(1).fillna(0).astype(int).rename('regime_score')
    return score


def compute_exit_trigger(df: pd.DataFrame) -> pd.Series:
    """Compute per-bar exit trigger: close crosses below EMA-8.

    Args:
        df: DataFrame containing at least ``['close', 'ema8']``.

    Returns:
        Boolean Series with the same index as *df*.  ``True`` on bars where
        ``close < ema8``, indicating an intraday or end-of-day exit should be
        considered.

    Raises:
        ValueError: If required columns are missing.
    """
    required = ['close', 'ema8']
    _check_columns(df, required, 'compute_exit_trigger')

    trigger = df['close'] < df['ema8']
    return trigger.rename('exit_trigger')
