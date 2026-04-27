"""Publication-quality plotting utilities for DriftFire strategy reports."""

from typing import Optional, List, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns


def set_plot_style() -> None:
    """Apply consistent publication-quality matplotlib style.

    Sets rcParams globally: clean axes, grid with low alpha, sans-serif
    font, 150 DPI, no top/right spines.  Call once at the top of a
    notebook or plotting function.
    """
    plt.rcParams.update({
        "figure.figsize":        (12, 6),
        "figure.dpi":            150,
        "font.family":           "sans-serif",
        "font.size":             11,
        "axes.titlesize":        14,
        "axes.labelsize":        12,
        "axes.spines.top":       False,
        "axes.spines.right":     False,
        "axes.grid":             True,
        "grid.alpha":            0.3,
        "grid.linestyle":        "--",
        "lines.linewidth":       1.5,
        "legend.fontsize":       10,
        "legend.framealpha":     0.7,
        "savefig.bbox":          "tight",
        "savefig.dpi":           150,
    })


def plot_equity_curve(
    equity_dict: dict,
    benchmark_series: Optional[pd.Series] = None,
    splits: Optional[List[Tuple]] = None,
    initial_capital: float = 100_000,
    title: str = "DriftFire — Equity Curves",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot equity curves for one or more strategies with optional SPY benchmark.

    Args:
        equity_dict: Mapping of ``{label: equity_df}`` where each equity_df
            has columns ``['date', 'equity']``.
        benchmark_series: Optional pd.Series of benchmark equity values
            (e.g. SPY buy-and-hold), indexed by date.
        splits: Optional list of ``(pd.Timestamp, label, color)`` tuples for
            vertical split lines marking train/val/test boundaries.
        initial_capital: Used to draw a horizontal reference line.
        title: Chart title.
        save_path: If provided, figure is saved to this path at 150 DPI.

    Returns:
        matplotlib Figure object.
    """
    set_plot_style()

    palette = [
        "#2166AC", "#4DAC26", "#D6604D", "#762A83",
        "#1A1A1A", "#5AAE61", "#B2182B",
    ]

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, (label, eq_df) in enumerate(equity_dict.items()):
        eq = eq_df.set_index("date")["equity"] if isinstance(eq_df, pd.DataFrame) else eq_df
        ax.plot(
            eq.index, eq.values,
            label=label,
            color=palette[i % len(palette)],
            lw=2.0 if i == 0 else 1.5,
            alpha=0.92,
        )

    if benchmark_series is not None:
        ax.plot(
            benchmark_series.index, benchmark_series.values,
            label="SPY (buy & hold)",
            color="#B2182B",
            lw=1.5,
            linestyle="--",
            alpha=0.75,
        )

    ax.axhline(initial_capital, color="#888888", lw=0.8, linestyle=":", zorder=0)

    if splits:
        for split_ts, split_label, split_color in splits:
            ax.axvline(split_ts, color=split_color, linestyle="--", lw=1.2, alpha=0.7)
            ymax = ax.get_ylim()[1]
            ax.text(
                split_ts, ymax * 0.98, f"  {split_label}",
                color=split_color, fontsize=9, va="top",
            )

    ax.set_title(title)
    ax.set_ylabel("Portfolio Value ($)")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"${x:,.0f}")
    )
    ax.legend(loc="upper left")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_drawdown(
    equity_df: pd.DataFrame,
    title: str = "Strategy Drawdown",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot an underwater drawdown chart.

    Args:
        equity_df: DataFrame with columns ``['date', 'equity']``.
        title: Chart title.
        save_path: If provided, saved to this path at 150 DPI.

    Returns:
        matplotlib Figure object.
    """
    set_plot_style()

    eq = equity_df.set_index("date")["equity"]
    rolling_max = eq.cummax()
    drawdown = (eq - rolling_max) / rolling_max * 100

    fig, ax = plt.subplots(figsize=(14, 4))
    ax.fill_between(drawdown.index, drawdown.values, 0, color="#D6604D", alpha=0.55)
    ax.plot(drawdown.index, drawdown.values, color="#B2182B", lw=0.9)
    ax.axhline(0, color="#555555", lw=0.7)

    max_dd = drawdown.min()
    max_dd_date = drawdown.idxmin()
    ax.annotate(
        f"Max DD: {max_dd:.1f}%",
        xy=(max_dd_date, max_dd),
        xytext=(max_dd_date, max_dd - 3),
        arrowprops=dict(arrowstyle="->", color="#B2182B", lw=1.0),
        color="#B2182B",
        fontsize=9,
    )

    ax.set_title(title)
    ax.set_ylabel("Drawdown (%)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:.0f}%"))
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_trade_distribution(
    trades_df: pd.DataFrame,
    split_by: str = "period",
    groups: Optional[List[str]] = None,
    title: str = "Trade Return Distributions",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot side-by-side histograms of trade returns for each period/group.

    Args:
        trades_df: Trade log with at least ``['return_pct', split_by]``.
        split_by: Column to split panels by (default ``'period'``).
        groups: Ordered list of group values to plot.  Defaults to
            ``['train', 'test']``.
        title: Overall figure title.
        save_path: If provided, saved to this path at 150 DPI.

    Returns:
        matplotlib Figure object.
    """
    set_plot_style()

    if groups is None:
        groups = ["train", "test"]

    colors = ["#2166AC", "#D6604D", "#4DAC26", "#762A83"]
    n = len(groups)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 4), sharey=False)
    if n == 1:
        axes = [axes]

    for ax, group, color in zip(axes, groups, colors):
        sub = trades_df[trades_df[split_by] == group]["return_pct"] * 100
        if len(sub) > 0:
            ax.hist(sub, bins=max(10, len(sub) // 3), color=color,
                    edgecolor="white", alpha=0.85)
            ax.axvline(sub.mean(), color="black", linestyle="--", lw=1.5,
                       label=f"Mean: {sub.mean():.1f}%")
            ax.axvline(0, color="#888888", linestyle=":", lw=1.0)
            ax.legend(fontsize=9)
        ax.set_title(f"{group.capitalize()} ({len(sub)} trades)")
        ax.set_xlabel("Return (%)")
        ax.set_ylabel("Count")

    fig.suptitle(title, fontsize=13, y=1.01)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_monthly_heatmap(
    equity_df: pd.DataFrame,
    title: str = "Monthly Returns Heatmap",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Calendar heatmap of monthly strategy returns.

    Args:
        equity_df: DataFrame with columns ``['date', 'equity']``.
        title: Chart title.
        save_path: If provided, saved to this path at 150 DPI.

    Returns:
        matplotlib Figure object.
    """
    set_plot_style()

    eq = equity_df.set_index("date")["equity"]
    monthly_eq  = eq.resample("ME").last()
    monthly_ret = monthly_eq.pct_change().dropna() * 100

    hm_data = pd.DataFrame({
        "year":  monthly_ret.index.year,
        "month": monthly_ret.index.month,
        "ret":   monthly_ret.values,
    })
    pivot = hm_data.pivot(index="year", columns="month", values="ret")
    pivot.columns = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]

    fig, ax = plt.subplots(figsize=(14, 5))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        center=0,
        cmap="RdYlGn",
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Monthly Return (%)"},
        annot_kws={"size": 9},
    )
    ax.set_title(title)
    ax.set_ylabel("Year")
    ax.set_xlabel("")
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig


def plot_cost_sensitivity(
    cost_df: pd.DataFrame,
    baseline_bps: int = 10,
    title: str = "Sharpe vs Transaction Cost",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """Plot Sharpe ratio as a function of round-trip transaction cost.

    Args:
        cost_df: DataFrame with columns ``['cost_bps', 'sharpe']``.
        baseline_bps: Highlight this cost level as the baseline.
        title: Chart title.
        save_path: If provided, saved to this path at 150 DPI.

    Returns:
        matplotlib Figure object.
    """
    set_plot_style()

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(
        cost_df["cost_bps"], cost_df["sharpe"],
        marker="o", color="#2166AC", lw=2, markersize=6,
    )
    ax.axhline(0, color="#888888", lw=0.8, linestyle=":")

    baseline_row = cost_df[cost_df["cost_bps"] == baseline_bps]
    if len(baseline_row) > 0:
        bx = int(baseline_row["cost_bps"].iloc[0])
        by = float(baseline_row["sharpe"].iloc[0])
        ax.axvline(bx, color="orange", linestyle="--", lw=1.2, alpha=0.7)
        ax.annotate(
            f"Baseline ({bx} bps)\nSharpe={by:.2f}",
            xy=(bx, by),
            xytext=(bx + 5, by + 0.05),
            arrowprops=dict(arrowstyle="->", color="orange"),
            color="orange",
            fontsize=9,
        )

    ax.set_xlabel("Round-trip Cost (bps)")
    ax.set_ylabel("Sharpe Ratio")
    ax.set_title(title)
    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")

    return fig
