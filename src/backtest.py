"""Vectorized backtest engine for DriftFire strategy."""

import time
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def run_backtest(
    df: pd.DataFrame,
    signal_col: str = "entry_regime3",
    initial_capital: float = 100_000,
    cost_bps: int = 10,
    max_hold_days: int = 20,
    max_positions: int = 5,
    position_size_pct: float = 0.20,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    spy_returns: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run a vectorized event-driven backtest of the DriftFire strategy.

    Approach:
        1. Iterate only over signal rows (~86-109 iterations) to compute each
           trade's hypothetical entry/exit and return — avoiding a full
           date-by-date loop over 1,760 dates.
        2. Apply position limits using a numpy integer array that tracks the
           number of open positions on each trading day.
        3. Build the equity curve via a cumulative cash-flow array and a
           pivoted price table for mark-to-market.

    All signals must already be precomputed (e.g. ``entry_regime3``) and
    stored in *df*.  Entry is at ``next_open`` the day following the signal;
    exit is at the ``close`` on the first day where ``exit_trigger`` is True
    **or** ``max_hold_days`` have elapsed, whichever comes first.

    Args:
        df: Feature DataFrame sorted by (ticker, date).  Must contain at
            minimum: ``date``, ``ticker``, ``close``, ``next_open``,
            ``exit_trigger``, ``vol_ratio``, and the column named by
            *signal_col*.
        signal_col: Name of the boolean entry-signal column to use.
        initial_capital: Starting portfolio cash in dollars.
        cost_bps: Round-trip transaction cost in basis points (split evenly
            on entry and exit).
        max_hold_days: Maximum number of trading days to hold a position
            before a forced exit.
        max_positions: Maximum number of concurrent open positions.
        position_size_pct: Fraction of portfolio to allocate to each position
            (e.g. 0.20 = 20%).
        start_date: Optional ISO date string; only include dates >= this.
        end_date: Optional ISO date string; only include dates <= this.

    Returns:
        A tuple ``(trades_df, equity_df)`` where:

        * **trades_df** – one row per completed trade with columns:
          ``ticker``, ``signal_date``, ``entry_date``, ``exit_date``,
          ``entry_price``, ``exit_price``, ``shares``, ``gross_pnl``,
          ``net_pnl``, ``return_pct``, ``days_held``, ``exit_reason``,
          ``period``.
        * **equity_df** – daily portfolio value with columns ``date``,
          ``equity``.

    Raises:
        ValueError: If *signal_col* is not present in *df*, or if required
            columns are missing.
    """
    # ── Validation ────────────────────────────────────────────────────────────
    if signal_col not in df.columns:
        avail = [c for c in df.columns if "entry" in c or "signal" in c]
        raise ValueError(
            f"Signal column '{signal_col}' not found. Available: {avail}"
        )

    required = ["date", "ticker", "close", "next_open", "exit_trigger", "vol_ratio"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"run_backtest: missing required columns: {missing}")

    n_signals = int(df[signal_col].sum())
    if n_signals == 0:
        warnings.warn(f"No signals found in '{signal_col}'. Returning empty results.")
        return pd.DataFrame(), pd.DataFrame()

    # ── Filter date range ──────────────────────────────────────────────────────
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    if start_date:
        df = df[df["date"] >= pd.Timestamp(start_date)]
    if end_date:
        df = df[df["date"] <= pd.Timestamp(end_date)]

    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    # ── Date index mapping ────────────────────────────────────────────────────
    # Use pd.Timestamp keys so Timestamp comparisons in the signal loop work.
    all_dates_ts = sorted(df["date"].unique())   # list of numpy datetime64
    all_dates = np.array(all_dates_ts)           # numpy array for equity indexing
    n_days = len(all_dates)
    # Normalise to pd.Timestamp for consistent dict lookups.
    date_to_idx: dict = {pd.Timestamp(d): i for i, d in enumerate(all_dates)}

    # ── Period labeler ────────────────────────────────────────────────────────
    def _period(d: pd.Timestamp) -> str:
        if d <= pd.Timestamp("2020-12-31"):
            return "train"
        elif d <= pd.Timestamp("2021-12-31"):
            return "val"
        return "test"

    # ── Per-ticker forward-exit lookup (numpy arrays) ─────────────────────────
    # For each ticker, store arrays of dates, closes, exit_triggers.
    ticker_arrays: dict = {}
    for ticker, grp in df.groupby("ticker", sort=False):
        g = grp.sort_values("date")
        ticker_arrays[ticker] = {
            "dates": g["date"].values,
            "closes": g["close"].values.astype(float),
            "exit_triggers": g["exit_trigger"].values.astype(bool),
        }

    cost_per_side = cost_bps / 2 / 10_000

    # ── Step 1: Compute all hypothetical trade outcomes ───────────────────────
    # Only loop over signal rows (fast — ~86-109 iterations).
    signal_rows = df[df[signal_col] == True].sort_values(["date", "vol_ratio"],
                                                          ascending=[True, False])

    all_trades = []
    for row in signal_rows.itertuples(index=False):
        signal_date = row.date
        ticker = row.ticker
        next_op = getattr(row, "next_open", np.nan)

        if pd.isna(next_op) or next_op <= 0:
            continue
        if ticker not in ticker_arrays:
            continue

        ta = ticker_arrays[ticker]
        # Find the index of the next trading day for this ticker.
        future_mask = ta["dates"] > signal_date
        if not future_mask.any():
            continue

        future_offset = int(np.argmax(future_mask))   # first future day index
        entry_ticker_idx = future_offset

        # Scan forward up to max_hold_days for an exit.
        hold_window = ta["exit_triggers"][entry_ticker_idx : entry_ticker_idx + max_hold_days]
        hold_closes = ta["closes"][entry_ticker_idx : entry_ticker_idx + max_hold_days]
        hold_dates  = ta["dates"][entry_ticker_idx : entry_ticker_idx + max_hold_days]

        exit_reason = "max_hold"
        exit_local_idx = len(hold_window) - 1  # default: last day in window

        triggers = np.where(hold_window)[0]
        if len(triggers) > 0:
            exit_local_idx = int(triggers[0])
            exit_reason = "ema8_stop"

        if exit_local_idx < 0 or exit_local_idx >= len(hold_closes):
            continue

        entry_price = float(next_op)
        exit_price  = float(hold_closes[exit_local_idx])
        exit_date   = pd.Timestamp(hold_dates[exit_local_idx])
        signal_ts   = pd.Timestamp(signal_date)
        days_held   = exit_local_idx + 1

        # Dates in the global all_dates array (keys are pd.Timestamp)
        if signal_ts not in date_to_idx or exit_date not in date_to_idx:
            continue

        signal_global_idx = date_to_idx[signal_ts]
        entry_global_idx  = signal_global_idx + 1   # entry is next trading day
        exit_global_idx   = date_to_idx[exit_date]

        all_trades.append({
            "ticker":           ticker,
            "signal_date":      signal_ts,
            "entry_date":       pd.Timestamp(all_dates[min(entry_global_idx, n_days - 1)]),
            "exit_date":        exit_date,
            "entry_price":      entry_price,
            "exit_price":       exit_price,
            "vol_ratio":        getattr(row, "vol_ratio", 0.0),
            "days_held":        days_held,
            "exit_reason":      exit_reason,
            "signal_global_idx": signal_global_idx,
            "entry_global_idx":  min(entry_global_idx, n_days - 1),
            "exit_global_idx":  exit_global_idx,
            "period":           _period(pd.Timestamp(signal_date)),
        })

    if not all_trades:
        warnings.warn(f"No tradeable signals after exit lookup for '{signal_col}'.")
        return pd.DataFrame(), pd.DataFrame()

    # ── Step 2: Apply position-limit constraint ───────────────────────────────
    # Use a numpy array counting active positions per day.
    active = np.zeros(n_days, dtype=np.int16)
    accepted = []

    for trade in all_trades:   # already sorted by date then vol_ratio desc
        ei = trade["entry_global_idx"]
        xi = trade["exit_global_idx"]
        if xi < ei:
            xi = ei
        # Check if any day in [entry, exit] already has MAX_POSITIONS open.
        window = active[ei : xi + 1]
        if window.max() < max_positions:
            active[ei : xi + 1] += 1
            accepted.append(trade)

    if not accepted:
        warnings.warn("All trades filtered out by position-limit constraint.")
        return pd.DataFrame(), pd.DataFrame()

    # ── Step 3: Compute trade-level financials ────────────────────────────────
    position_value_used = initial_capital * position_size_pct
    trade_records = []
    for t in accepted:
        shares     = position_value_used / t["entry_price"]
        entry_cost = shares * t["entry_price"] * cost_per_side
        exit_cost  = shares * t["exit_price"]  * cost_per_side
        gross_pnl  = (t["exit_price"] - t["entry_price"]) * shares
        net_pnl    = gross_pnl - entry_cost - exit_cost
        ret_pct    = net_pnl / (t["entry_price"] * shares)
        trade_records.append({
            "ticker":       t["ticker"],
            "signal_date":  t["signal_date"],
            "entry_date":   t["entry_date"],
            "exit_date":    t["exit_date"],
            "entry_price":  t["entry_price"],
            "exit_price":   t["exit_price"],
            "shares":       shares,
            "gross_pnl":    gross_pnl,
            "net_pnl":      net_pnl,
            "return_pct":   ret_pct,
            "days_held":    t["days_held"],
            "exit_reason":  t["exit_reason"],
            "period":       t["period"],
            "_ei":          t["entry_global_idx"],
            "_xi":          t["exit_global_idx"],
        })

    trades_df = pd.DataFrame(trade_records)

    # ── Step 4: Build equity curve via cumulative cash-flows ──────────────────
    # Build price pivot for mark-to-market of open positions.
    price_pivot = df.pivot_table(
        index="date", columns="ticker", values="close", aggfunc="last"
    )
    price_pivot = price_pivot.reindex(all_dates)

    cash_delta = np.zeros(n_days, dtype=float)
    mtm        = np.zeros(n_days, dtype=float)

    for t in trade_records:
        shares     = t["shares"]
        ep, xp     = t["entry_price"], t["exit_price"]
        ei, xi     = t["_ei"], t["_xi"]
        entry_cost = shares * ep * cost_per_side
        exit_cost  = shares * xp * cost_per_side

        # Cash out on entry, cash back on exit.
        cash_delta[ei] -= shares * ep + entry_cost
        cash_delta[xi] += shares * xp - exit_cost

        # Mark-to-market while position is open.
        tkr = t["ticker"]
        if tkr in price_pivot.columns:
            prices = price_pivot[tkr].iloc[ei : xi + 1].values.astype(float)
            # Fill NaN prices with last known price to avoid gaps.
            for k in range(1, len(prices)):
                if np.isnan(prices[k]):
                    prices[k] = prices[k - 1]
            mtm[ei : xi + 1] += np.where(np.isnan(prices), 0.0, prices) * shares

    cash_balance = float(initial_capital) + np.cumsum(cash_delta)
    equity_values = cash_balance + mtm

    equity_df = pd.DataFrame({
        "date":   pd.to_datetime(all_dates),
        "equity": equity_values,
    })

    # ── Optional SPY cash overlay ─────────────────────────────────────────────
    # When spy_returns is provided, idle cash earns SPY daily returns instead
    # of sitting at 0%.  Total exposure never exceeds 100% — not leverage.
    run_backtest._spy_returns = spy_returns  # store for overlay block below
    if spy_returns is not None:
        spy_ret = run_backtest._spy_returns.reindex(
            pd.to_datetime(all_dates)
        ).fillna(0.0).values

        # Track how much equity is invested in DriftFire positions each day.
        invested = np.zeros(n_days, dtype=float)
        for t in trade_records:
            ei, xi = t["_ei"], t["_xi"]
            invested[ei : xi + 1] += t["shares"] * t["entry_price"]

        # Cash portion = total equity minus invested value.
        # We compound SPY returns on the uninvested (cash) portion each day.
        eq_vals = equity_values.copy()
        spy_compound = np.ones(n_days, dtype=float)
        for d in range(1, n_days):
            cash_frac = max(0.0, eq_vals[d - 1] - invested[d - 1])
            spy_gain = cash_frac * spy_ret[d]
            eq_vals[d] += spy_gain
            spy_compound[d] = spy_compound[d - 1] * (1 + spy_ret[d])

        equity_df["equity"] = eq_vals
        equity_df["equity_no_overlay"] = equity_values

    # Drop internal columns from trades_df.
    trades_df = trades_df.drop(columns=["_ei", "_xi"], errors="ignore")

    return trades_df, equity_df


def calc_metrics(
    trades_df: pd.DataFrame,
    equity_df: pd.DataFrame,
    label: str = "",
    signal: str = "",
) -> dict:
    """Compute standard performance metrics for a backtest result.

    Args:
        trades_df: Trade log from :func:`run_backtest`.
        equity_df: Daily equity curve from :func:`run_backtest`.
        label: Period label (e.g. ``'train'``, ``'test'``, ``'full'``).
        signal: Signal name (e.g. ``'regime>=3'``).

    Returns:
        Dictionary with keys: ``signal``, ``period``, ``n_trades``,
        ``sharpe``, ``cagr_%``, ``max_dd_%``, ``hit_rate_%``,
        ``avg_return_%``, ``profit_factor``, ``avg_days``.
    """
    empty = {
        "signal": signal, "period": label, "n_trades": 0,
        "sharpe": np.nan, "cagr_%": np.nan, "max_dd_%": np.nan,
        "hit_rate_%": np.nan, "avg_return_%": np.nan,
        "profit_factor": np.nan, "avg_days": np.nan,
    }

    if trades_df is None or len(trades_df) == 0:
        return empty
    if equity_df is None or len(equity_df) == 0:
        return empty

    eq = equity_df.set_index("date")["equity"]
    if len(eq) < 2:
        return empty

    daily_ret = eq.pct_change().dropna()
    n_years   = max((eq.index[-1] - eq.index[0]).days / 365.25, 0.01)
    # Use first value as base (rebase so each period starts at 1.0)
    eq_rebased = eq / eq.iloc[0]
    cagr      = (eq_rebased.iloc[-1]) ** (1 / n_years) - 1
    sharpe    = (
        daily_ret.mean() / daily_ret.std() * np.sqrt(252)
        if daily_ret.std() > 0 else 0.0
    )
    max_dd    = ((eq_rebased - eq_rebased.cummax()) / eq_rebased.cummax()).min()

    rets    = trades_df["return_pct"]
    winners = rets[rets > 0]
    losers  = rets[rets < 0]
    hit_rate = len(winners) / len(rets) if len(rets) > 0 else np.nan
    pf = (
        winners.sum() / losers.abs().sum()
        if len(losers) > 0 and losers.abs().sum() > 0
        else np.nan
    )

    return {
        "signal":        signal,
        "period":        label,
        "n_trades":      len(trades_df),
        "sharpe":        round(float(sharpe), 2),
        "cagr_%":        round(float(cagr) * 100, 1),
        "max_dd_%":      round(float(max_dd) * 100, 1),
        "hit_rate_%":    round(float(hit_rate) * 100, 1) if not np.isnan(hit_rate) else np.nan,
        "avg_return_%":  round(float(rets.mean()) * 100, 2),
        "profit_factor": round(float(pf), 2) if not np.isnan(pf) else np.nan,
        "avg_days":      round(float(trades_df["days_held"].mean()), 1),
    }
