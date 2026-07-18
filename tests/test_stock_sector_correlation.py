from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from mapi.components.base import ComponentContext
from mapi.components.stock_sector import StockSectorDivergence
from mapi.config import MapiConfig
from mapi.data.validation import normalize_ohlcv
from mapi.models import HorizonConfig


def _ohlcv_from_returns(returns: np.ndarray, start: float = 100.0) -> pd.DataFrame:
    close = start * np.cumprod(1.0 + returns)
    timestamps = pd.date_range("2024-01-02", periods=len(close), freq="B", tz="UTC")
    return normalize_ohlcv(
        pd.DataFrame(
            {
                "timestamp": timestamps,
                "open": close,
                "high": close * 1.002,
                "low": close * 0.998,
                "close": close,
                "volume": 1_000_000.0,
            }
        )
    )


class StockSectorCorrelationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.horizon = HorizonConfig("test", 1, 20, 8)
        self.config = MapiConfig()

    def test_consistently_low_correlation_is_not_a_continuous_breakdown(self) -> None:
        sector_returns = np.tile([0.01, -0.01, 0.01, -0.01], 50)
        stock_returns = np.tile([0.01, 0.01, -0.01, -0.01], 50)
        stock = _ohlcv_from_returns(stock_returns)
        sector = _ohlcv_from_returns(sector_returns, start=80.0)
        result = StockSectorDivergence().calculate(
            stock,
            ComponentContext(symbol="TEST", sector_frame=sector),
            self.horizon,
            self.config,
        )
        declines = pd.Series(
            [item.get("correlation_decline") for item in result["metrics"]],
            dtype=float,
        ).dropna()
        self.assertLess(float(declines.iloc[-40:].max()), 0.05)

    def test_decline_from_high_historical_correlation_is_a_breakdown(self) -> None:
        rng = np.random.default_rng(204)
        sector_returns = rng.normal(0.0005, 0.01, 220)
        stock_returns = sector_returns.copy()
        stock_returns[-60:] = rng.normal(0.0005, 0.01, 60)
        stock = _ohlcv_from_returns(stock_returns)
        sector = _ohlcv_from_returns(sector_returns, start=80.0)
        result = StockSectorDivergence().calculate(
            stock,
            ComponentContext(symbol="TEST", sector_frame=sector),
            self.horizon,
            self.config,
        )
        declines = pd.Series(
            [item.get("correlation_decline") for item in result["metrics"]],
            dtype=float,
        ).dropna()
        self.assertGreater(float(declines.iloc[-60:].max()), 0.40)


if __name__ == "__main__":
    unittest.main()
