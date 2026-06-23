# Limitations

This section highlights known limitations of the DriftFire v1 research
artifact. Readers should treat this as a research-grade demonstration,
not production infrastructure.

- Data: raw downloads depend on third-party providers and may contain
  survivorship bias or symbol delistings.
- Market assumptions: fills at next-open and exit at close are simplified
  execution assumptions and ignore intraday liquidity or slippage.
- Parameter sensitivity: results can change materially under different
  EMA spans, vol thresholds, and position-sizing rules.
- Evaluation: while walk-forward windows were used, the historical period
  is finite and subject to regime differences in out-of-sample markets.

See notebooks and `docs/methodology.md` for more details on how features
and signals are constructed.
