# MAPI research report: [universe and period]

MAPI is experimental. This report tests a research hypothesis and does not claim profitability.

## Research question

- Hypothesis:
- Universe and survivorship policy:
- Data vendor and point-in-time guarantees:
- Bar frequency and timezone:
- Train / validation / untouched test dates:

## Signal specification

- MAPI version:
- Enabled components and weights:
- Horizon:
- Entry delay and holding period:
- Score and direction thresholds:

## Execution assumptions

- Transaction cost:
- Slippage and spread:
- Liquidity constraints:
- Corporate action treatment:

## Event-study results

Include sample count, return distribution, unannualized event-return ratios, bootstrap confidence intervals, gross directional accuracy, net profitable event rate, large-move capture rate, information coefficient, intrabar MFE/MAE, and intrabar breakout rate. Report each baseline's fitted threshold, fit/test candidate frequency, selected event count, and excluded overlap count. Keep the full-period buy-and-hold benchmark separate from event studies. Show results by year, regime, symbol, sector, score bucket, and direction.

Treat Sharpe, drawdown, and profit factor as descriptive event-sequence statistics. Do not present them as portfolio performance until capital, sizing, exposure, overlapping positions, borrow, and capacity are modeled.

## Controls

Compare against random signals with matching frequency, previous-return direction, pure volume z-score, moving-average crossover, RSI reversal, MACD direction, and buy-and-hold. Report component ablations.

## Robustness checks

- Walk-forward stability:
- Parameter sensitivity:
- Delisted symbols and missing observations:
- Alternative cost assumptions:
- Multiple-testing correction:

## Decision

- Evidence supporting the hypothesis:
- Evidence against the hypothesis:
- Known failure modes:
- Keep, revise, or reject:
- Locked next experiment:
