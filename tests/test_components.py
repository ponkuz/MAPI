from __future__ import annotations

import unittest

from mapi.components.base import ComponentContext
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


if __name__ == "__main__":
    unittest.main()
