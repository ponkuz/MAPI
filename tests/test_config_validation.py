from __future__ import annotations

import math
import unittest

from mapi.config import MapiConfig
from mapi.models import HorizonConfig
from mapi.scoring import calculate_mapi
from mapi.weights import WeightInputs, calculate_dynamic_weight
from tests.helpers import make_ohlcv


class ConfigValidationTests(unittest.TestCase):
    def test_unknown_enabled_component_fails_before_scoring(self) -> None:
        config = MapiConfig()
        config.enabled_components = ["misspelled_component"]
        config.component_weights["misspelled_component"] = 1.0
        with self.assertRaisesRegex(ValueError, "Unknown enabled components"):
            calculate_mapi("TEST", make_ohlcv(30), config=config)

    def test_invalid_component_weights_and_reliability_fail(self) -> None:
        for value in (-1.0, math.nan, math.inf):
            with self.subTest(weight=value):
                config = MapiConfig()
                config.component_weights[config.enabled_components[0]] = value
                with self.assertRaises(ValueError):
                    config.validate()
        config = MapiConfig()
        config.component_weights = {name: 0.0 for name in config.enabled_components}
        with self.assertRaisesRegex(ValueError, "must be positive"):
            config.validate()
        for value in (-0.01, 1.01):
            with self.subTest(reliability=value):
                config = MapiConfig()
                config.component_reliability[config.enabled_components[0]] = value
                with self.assertRaises(ValueError):
                    config.validate()

    def test_invalid_windows_thresholds_redundancy_and_staleness_fail(self) -> None:
        cases = []
        config = MapiConfig()
        config.horizons = {"bad": HorizonConfig("bad", 1, 5, 6)}
        cases.append(config)
        config = MapiConfig()
        config.high_anomaly_threshold = 70.0
        config.strong_anomaly_threshold = 60.0
        cases.append(config)
        config = MapiConfig()
        config.redundancy_min_periods = config.redundancy_window + 1
        cases.append(config)
        config = MapiConfig()
        config.cross_asset_max_staleness_bars = -1.0
        cases.append(config)
        for config in cases:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    config.validate()

    def test_invalid_enums_frequency_tolerance_and_penalty_fail(self) -> None:
        cases = []
        config = MapiConfig()
        config.score_semantics = "predictive_alpha"
        cases.append(config)
        config = MapiConfig()
        config.frequency_mismatch_policy = "ignore"
        cases.append(config)
        config = MapiConfig()
        config.cross_asset_frequency_tolerance = 0.0
        cases.append(config)
        config = MapiConfig()
        config.min_redundancy_penalty = 1.1
        cases.append(config)
        for config in cases:
            with self.subTest(config=config):
                with self.assertRaises(ValueError):
                    config.validate()

    def test_zero_reliability_disables_dynamic_weight(self) -> None:
        decision = calculate_dynamic_weight(
            "price_volume_divergence",
            0.5,
            WeightInputs(
                regime="risk_on_low_volatility",
                freshness=1.0,
                reliability=0.0,
                liquidity=1.0,
                persistence=1.0,
            ),
        )
        self.assertEqual(decision.factors["reliability"], 0.0)
        self.assertEqual(decision.weight, 0.0)


if __name__ == "__main__":
    unittest.main()
