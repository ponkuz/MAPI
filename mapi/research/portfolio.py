"""Reserved namespace for future capital-aware portfolio simulation.

The Phase 1 event-study engine intentionally does not calculate portfolio PnL,
position sizing, exposure, borrow cost, capacity, or overlapping position state.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PortfolioMetrics:
    total_return: float
    volatility: float
    sharpe_ratio: float
    maximum_drawdown: float
    gross_exposure: float
    net_exposure: float
