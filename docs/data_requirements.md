# Data requirements

## Adjusted prices

MAPI v0.2 expects split- and dividend-adjusted OHLC prices on one consistent basis. Mixing adjusted close with unadjusted open/high/low creates false gap, volatility, momentum, and residual anomalies. Volume should be split-adjusted when possible.

The validator warns when a close-to-close ratio is at least 1.8 or at most `1 / 1.8`, and OHLCV quality is reduced on that row. Genuine market moves can trigger the warning, while some corporate actions can remain below the threshold, so vendor corporate-action metadata remains preferable.

## Cross-asset alignment

Sector and benchmark series are matched to stock bars with backward-only as-of matching. A match is valid only when:

- its timestamp is not later than the stock timestamp;
- both timestamps share the same UTC date;
- staleness is within `cross_asset_max_staleness_bars` times the target median bar interval;
- source median frequency is no slower than the target frequency times `cross_asset_frequency_tolerance`.

Invalid matches become missing values and their component confidence is zero. This prevents indefinite forward filling, but UTC dates are only a Phase 1 session proxy. Exchange calendars are still required for overnight and multi-session instruments.

## Point-in-time research

Fundamentals, news, estimates, options, and sentiment are inactive placeholders. When connected later, their timestamps must represent original availability, not revised database timestamps. Delisted symbols and historical universe membership remain data-provider responsibilities.
