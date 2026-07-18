# Event-study execution semantics

The Phase 1 research engine is an event study, not a portfolio simulator.

For a signal at zero-based bar `i`:

- The signal timestamp is the close timestamp of bar `i`; all features may use bar `i` and earlier information.
- Entry occurs at the close of bar `i + signal_delay_bars`. The default delay is one bar.
- Exit occurs at the close of bar `i + signal_delay_bars + horizon_bars`.
- `horizon_bars` counts close-to-close holding intervals after entry.
- Realized return is close-to-close. Intrabar MFE/MAE and breakout use high/low from bars after entry through the exit bar; the entry bar's earlier range is excluded.
- Long gross return is `exit / entry - 1`.
- Short gross return is `-(exit / entry - 1)`.
- Long intrabar MFE/MAE are the maximum future high and minimum future low returns from entry.
- Short MFE is the negative of long-path MAE; short MAE is the negative of long-path MFE.
- Transaction cost, spread, and slippage are summed and subtracted once as a round-trip return estimate.

By default, the engine greedily keeps the first eligible signal and rejects later signals whose holding intervals overlap it. `allow_overlapping=True` is available only for descriptive event research and adds an explicit warning.

`event_return_mean_to_std` and `event_return_mean_to_downside_std` are unannualized descriptive event-return ratios. The mean event return includes a deterministic 95% bootstrap confidence interval. Annualized Sharpe is intentionally reserved for a future timestamp- and capital-aware portfolio simulator.

Canonical descriptive classification names are `gross_directional_accuracy`, `net_profitable_event_rate`, and `large_move_capture_rate`; the large-move universe uses a fixed configured threshold. Legacy `sharpe_ratio`, `sortino_ratio`, `precision`, `recall`, `hit_rate`, `mean_mfe`, `mean_mae`, and `breakout_rate` fields remain aliases and produce a compatibility warning.

Negative controls fit their own score threshold on the chronological fit partition to match the MAPI candidate frequency, then freeze the threshold for test evaluation. `always_long_fixed_horizon` is the repeated event control. `full_period_buy_and_hold` is reported separately as a capital-path benchmark.

The reserved `mapi.research.portfolio` namespace contains only a future metrics contract. It does not imply that position sizing, capital, borrow, exposure, or capacity has been implemented.
