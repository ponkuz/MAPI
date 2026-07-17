# Event-study execution semantics

The Phase 1 research engine is an event study, not a portfolio simulator.

For a signal at zero-based bar `i`:

- The signal timestamp is the close timestamp of bar `i`; all features may use bar `i` and earlier information.
- Entry occurs at the close of bar `i + signal_delay_bars`. The default delay is one bar.
- Exit occurs at the close of bar `i + signal_delay_bars + horizon_bars`.
- `horizon_bars` counts close-to-close holding intervals after entry.
- The MFE/MAE path includes the entry close, every intermediate close, and the exit close.
- Long gross return is `exit / entry - 1`.
- Short gross return is `-(exit / entry - 1)`.
- Long MFE/MAE are the maximum/minimum path returns from entry.
- Short MFE is the negative of long-path MAE; short MAE is the negative of long-path MFE.
- Transaction cost, spread, and slippage are summed and subtracted once as a round-trip return estimate.

By default, the engine greedily keeps the first eligible signal and rejects later signals whose holding intervals overlap it. `allow_overlapping=True` is available only for descriptive event research and adds an explicit warning.

Legacy fields `sharpe_ratio`, `maximum_drawdown`, and `profit_factor` remain for output compatibility. The same values are exposed as `event_return_sharpe`, `event_sequence_drawdown`, and `event_profit_factor`. They must not be interpreted as capital-aware strategy results.

The reserved `mapi.research.portfolio` namespace contains only a future metrics contract. It does not imply that position sizing, capital, borrow, exposure, or capacity has been implemented.
