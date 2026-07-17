from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from mapi.components.base import ComponentContext
from mapi.components.market_regime import MarketRegimeDivergence
from mapi.components.stock_sector import StockSectorDivergence
from mapi.data.alignment import align_point_in_time
from mapi.data.validation import normalize_ohlcv
from tests.helpers import make_ohlcv, small_config


class AlignmentTests(unittest.TestCase):
    def test_missing_benchmark_session_is_not_forward_filled(self) -> None:
        prices = normalize_ohlcv(make_ohlcv(40, seed=1))
        benchmark_raw = make_ohlcv(40, seed=2)
        missing_timestamp = benchmark_raw.iloc[25]["timestamp"]
        benchmark = normalize_ohlcv(
            benchmark_raw[benchmark_raw["timestamp"] != missing_timestamp]
        )
        config = small_config()
        horizon = config.horizons["short_term"]

        result = MarketRegimeDivergence().calculate(
            prices,
            ComponentContext(symbol="TEST", benchmark_frame=benchmark),
            horizon,
            config,
        )

        timestamp = pd.to_datetime(missing_timestamp, utc=True)
        self.assertEqual(float(result.loc[timestamp, "confidence"]), 0.0)
        self.assertIn("stale", result.loc[timestamp, "reason"])
        self.assertGreater(float(result.iloc[-1]["confidence"]), 0.0)

    def test_mixed_daily_and_intraday_frequencies_are_rejected(self) -> None:
        timestamps = pd.date_range("2025-01-02 14:30", periods=20, freq="30min", tz="UTC")
        close = np.linspace(100.0, 102.0, len(timestamps))
        intraday = normalize_ohlcv(
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "open": close,
                    "high": close + 0.2,
                    "low": close - 0.2,
                    "close": close,
                    "volume": 1000.0,
                }
            )
        )
        daily = normalize_ohlcv(make_ohlcv(5, seed=8))

        alignment = align_point_in_time(intraday, daily)

        self.assertFalse(alignment.frequency_compatible)
        self.assertFalse(alignment.valid.any())

    def test_matching_is_backward_only(self) -> None:
        target = normalize_ohlcv(make_ohlcv(10, seed=4))
        source = target.copy()
        source["timestamp"] = source["timestamp"] + pd.Timedelta(minutes=5)
        source.index = pd.DatetimeIndex(source["timestamp"])

        alignment = align_point_in_time(target, source)

        self.assertFalse(alignment.valid.any())

    def test_stock_sector_component_rejects_frequency_mismatch(self) -> None:
        prices = normalize_ohlcv(make_ohlcv(50, seed=9))
        sector = normalize_ohlcv(make_ohlcv(10, seed=10).iloc[::5])
        config = small_config()
        result = StockSectorDivergence().calculate(
            prices,
            ComponentContext(symbol="TEST", sector_frame=sector),
            config.horizons["short_term"],
            config,
        )
        self.assertTrue((result["confidence"] == 0.0).all())


if __name__ == "__main__":
    unittest.main()
