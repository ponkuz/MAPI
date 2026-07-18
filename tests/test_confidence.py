from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from mapi.components.base import ComponentContext
from mapi.config import MapiConfig
from mapi.models import HorizonConfig
from mapi.scoring import calculate_mapi
from tests.helpers import make_ohlcv


class FixedComponent:
    family = "test_family"

    def __init__(
        self,
        name: str,
        confidence: float,
        direction: float = 1.0,
        strengths: np.ndarray | None = None,
    ) -> None:
        self.name = name
        self.confidence = confidence
        self.direction = direction
        self.strengths = strengths

    def calculate(
        self,
        price_frame: pd.DataFrame,
        context: ComponentContext,
        horizon: HorizonConfig,
        config: MapiConfig,
    ) -> pd.DataFrame:
        strength = (
            self.strengths
            if self.strengths is not None
            else np.full(len(price_frame), 0.8)
        )
        return pd.DataFrame(
            {
                "anomaly_strength": strength,
                "direction": self.direction,
                "confidence": self.confidence,
                "novelty": 1.0,
                "reason": self.name,
                "metrics": [{} for _ in range(len(price_frame))],
            },
            index=price_frame.index,
        )


def coverage_config() -> MapiConfig:
    config = MapiConfig()
    names = [f"component_{index}" for index in range(6)]
    config.enabled_components = names
    config.component_weights = {name: 1.0 / 6.0 for name in names}
    config.component_reliability = {name: 1.0 for name in names}
    config.horizons = {
        "test": HorizonConfig("test", return_window=1, rolling_window=16, min_periods=6)
    }
    config.redundancy_window = 12
    config.redundancy_min_periods = 4
    return config


class ConfidenceCoverageTests(unittest.TestCase):
    def _latest(self, available: int) -> pd.Series:
        config = coverage_config()
        components = [
            FixedComponent(name, 1.0 if index < available else 0.0)
            for index, name in enumerate(config.enabled_components)
        ]
        return calculate_mapi(
            "TEST", make_ohlcv(50, seed=20), config=config, components=components
        )["test"].iloc[-1]

    def test_confidence_and_quality_follow_evidence_coverage(self) -> None:
        one = self._latest(1)
        three = self._latest(3)
        all_six = self._latest(6)

        self.assertLess(one["mapi_confidence"], three["mapi_confidence"])
        self.assertLess(three["mapi_confidence"], all_six["mapi_confidence"])
        self.assertLess(one["data_quality_score"], three["data_quality_score"])
        self.assertLess(three["data_quality_score"], all_six["data_quality_score"])
        self.assertAlmostEqual(float(one["evidence_coverage_score"]), 1.0 / 6.0, places=6)
        self.assertAlmostEqual(float(all_six["evidence_coverage_score"]), 1.0, places=6)

    def test_all_components_unavailable_has_zero_denominator(self) -> None:
        latest = self._latest(0)
        self.assertEqual(float(latest["mapi_score"]), 0.0)
        self.assertEqual(float(latest["mapi_confidence"]), 0.0)
        self.assertEqual(float(latest["data_quality_score"]), 0.0)

    def test_direction_cancellation_is_neutral(self) -> None:
        config = coverage_config()
        config.enabled_components = config.enabled_components[:2]
        config.component_weights = {name: 0.5 for name in config.enabled_components}
        components = [
            FixedComponent(config.enabled_components[0], 1.0, direction=1.0),
            FixedComponent(config.enabled_components[1], 1.0, direction=-1.0),
        ]
        latest = calculate_mapi(
            "TEST", make_ohlcv(50, seed=21), config=config, components=components
        )["test"].iloc[-1]
        self.assertGreater(float(latest["mapi_score"]), 0.0)
        self.assertAlmostEqual(float(latest["mapi_direction"]), 0.0, places=12)

    def test_correlated_evidence_receives_lower_coverage(self) -> None:
        config = coverage_config()
        config.enabled_components = config.enabled_components[:2]
        config.component_weights = {name: 0.5 for name in config.enabled_components}
        increasing = np.linspace(0.1, 0.9, 50)
        correlated = [
            FixedComponent(name, 1.0, strengths=increasing.copy())
            for name in config.enabled_components
        ]
        independent = [
            FixedComponent(config.enabled_components[0], 1.0, strengths=increasing),
            FixedComponent(
                config.enabled_components[1],
                1.0,
                strengths=np.sin(np.arange(50)) * 0.35 + 0.5,
            ),
        ]
        correlated_latest = calculate_mapi(
            "TEST", make_ohlcv(50, seed=22), config=config, components=correlated
        )["test"].iloc[-1]
        independent_latest = calculate_mapi(
            "TEST", make_ohlcv(50, seed=22), config=config, components=independent
        )["test"].iloc[-1]
        self.assertLess(
            float(correlated_latest["evidence_coverage_score"]),
            float(independent_latest["evidence_coverage_score"]),
        )

    def test_component_order_does_not_change_aggregate_output(self) -> None:
        config = coverage_config()
        strengths = [
            np.sin(np.arange(50) * (index + 1) / 7.0) * 0.3 + 0.5
            for index in range(6)
        ]
        components = [
            FixedComponent(name, 1.0, strengths=values)
            for name, values in zip(config.enabled_components, strengths)
        ]
        prices = make_ohlcv(50, seed=23)
        forward = calculate_mapi(
            "TEST", prices, config=config, components=components
        )["test"]
        reversed_config = coverage_config()
        reversed_config.enabled_components = list(
            reversed(reversed_config.enabled_components)
        )
        reverse = calculate_mapi(
            "TEST",
            prices,
            config=reversed_config,
            components=list(reversed(components)),
        )["test"]
        for column in ("mapi_score", "mapi_direction", "mapi_confidence"):
            pd.testing.assert_series_equal(
                forward[column],
                reverse[column],
                check_exact=False,
                rtol=1e-12,
                atol=1e-12,
            )

    def test_zero_reliability_does_not_add_coverage_or_confirmation(self) -> None:
        config = coverage_config()
        config.enabled_components = config.enabled_components[:2]
        config.component_weights = {name: 0.5 for name in config.enabled_components}
        config.component_reliability = {
            config.enabled_components[0]: 1.0,
            config.enabled_components[1]: 0.0,
        }
        components = [
            FixedComponent(config.enabled_components[0], 1.0),
            FixedComponent(config.enabled_components[1], 1.0),
        ]
        latest = calculate_mapi(
            "TEST", make_ohlcv(50, seed=24), config=config, components=components
        )["test"].iloc[-1]
        self.assertAlmostEqual(float(latest["evidence_coverage_score"]), 1.0)
        self.assertEqual(int(latest["confirmation_count"]), 1)
        disabled = next(
            component
            for component in latest["signal"].anomaly_components
            if component.name == config.enabled_components[1]
        )
        self.assertEqual(disabled.weight, 0.0)


if __name__ == "__main__":
    unittest.main()
