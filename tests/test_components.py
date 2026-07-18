from __future__ import annotations

import unittest

import pandas as pd

from mapi.components.base import ComponentContext
from mapi.components.momentum_disagreement import MomentumDisagreement
from mapi.components.price_volume import PriceVolumeDivergence
from mapi.components.stock_sector import StockSectorDivergence
from mapi.data.validation import normalize_ohlcv
from tests.helpers import make_ohlcv, small_config


class ComponentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.prices = normalize_ohlcv(make_ohlcv(90))
        self.config = small_config()
        self.horizon = self.config.horizons["short_term"]

    def test_price_volume_component_respects_output_contract(self) -> None:
        result = PriceVolumeDivergence().calculate(
            self.prices,
            ComponentContext(symbol="TEST"),
            self.horizon,
            self.config,
        )
        self.assertEqual(len(result), len(self.prices))
        for column in (
            "anomaly_strength",
            "confidence",
            "novelty",
            "historical_extremeness",
            "recurrence_rate",
        ):
            self.assertTrue(result[column].between(0.0, 1.0).all())
        self.assertTrue(result["direction"].between(-1.0, 1.0).all())

    def test_missing_sector_returns_zero_confidence(self) -> None:
        result = StockSectorDivergence().calculate(
            self.prices,
            ComponentContext(symbol="TEST", sector_frame=None),
            self.horizon,
            self.config,
        )
        self.assertTrue((result["confidence"] == 0.0).all())
        self.assertTrue(result["reason"].str.contains("unavailable").all())

    def test_momentum_divergence_maps_new_high_bearish_and_new_low_bullish(self) -> None:
        for closes, expected_direction, phrase in (
            ([100.0, 102.0, 104.0, 106.0, 100.0, 102.0, 105.0, 106.5], -1.0, "new high"),
            ([106.0, 104.0, 102.0, 100.0, 106.0, 104.0, 101.0, 99.5], 1.0, "new low"),
        ):
            with self.subTest(phrase=phrase):
                index = pd.date_range("2025-01-02", periods=len(closes), freq="B", tz="UTC")
                close = pd.Series(closes, index=index)
                prices = pd.DataFrame(
                    {
                        "open": close,
                        "high": close + 0.2,
                        "low": close - 0.2,
                        "close": close,
                        "volume": 1000.0,
                    },
                    index=index,
                )
                horizon = type(self.horizon)(
                    "test", return_window=1, rolling_window=7, min_periods=3
                )
                result = MomentumDisagreement().calculate(
                    prices,
                    ComponentContext(symbol="TEST"),
                    horizon,
                    self.config,
                )
                self.assertIn(phrase, result["reason"].iloc[-1].lower())
                self.assertEqual(
                    float(result["forecast_direction"].iloc[-1]),
                    expected_direction * 0.75,
                )
                self.assertEqual(
                    result["direction_semantics"].iloc[-1],
                    "hypothesized_forward_direction",
                )

    def test_high_volume_decline_is_directionally_confirmed_flow(self) -> None:
        raw = make_ohlcv(90, seed=77)
        previous_close = float(raw.loc[len(raw) - 2, "close"])
        raw.loc[len(raw) - 1, "open"] = previous_close
        raw.loc[len(raw) - 1, "close"] = previous_close * 0.90
        raw.loc[len(raw) - 1, "high"] = previous_close * 1.001
        raw.loc[len(raw) - 1, "low"] = previous_close * 0.899
        raw.loc[len(raw) - 1, "volume"] *= 12.0
        prices = normalize_ohlcv(raw)
        result = PriceVolumeDivergence().calculate(
            prices,
            ComponentContext(symbol="TEST"),
            self.horizon,
            self.config,
        )
        metrics = result["metrics"].iloc[-1]
        self.assertLess(metrics["price_z"], 0.0)
        self.assertGreater(metrics["volume_z"], 0.0)
        self.assertLess(metrics["directional_flow_z"], 0.0)
        self.assertEqual(metrics["directional_flow_mismatch"], 0.0)


if __name__ == "__main__":
    unittest.main()
