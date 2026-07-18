from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from mapi.components.base import ComponentContext
from mapi.config import MapiConfig
from mapi.models import HorizonConfig
from mapi.scoring import calculate_latest_mapi, calculate_mapi
from mapi.version import ALGORITHM_REVISION, DATA_CONTRACT_VERSION
from tests.helpers import make_ohlcv, small_config


class _FixedDirectionComponent:
    family = "test"

    def __init__(
        self,
        name: str,
        forecast_direction: float,
        directional_evidence_strength: float,
        semantics: str,
    ) -> None:
        self.name = name
        self.forecast_direction = forecast_direction
        self.directional_evidence_strength = directional_evidence_strength
        self.semantics = semantics

    def calculate(
        self,
        price_frame: pd.DataFrame,
        context: ComponentContext,
        horizon: HorizonConfig,
        config: MapiConfig,
    ) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "anomaly_strength": 1.0,
                "forecast_direction": self.forecast_direction,
                "observed_pressure": self.forecast_direction,
                "directional_evidence_strength": self.directional_evidence_strength,
                "direction_semantics": self.semantics,
                "direction_contract_warning": None,
                "confidence": 1.0,
                "novelty": 1.0,
                "historical_extremeness": 1.0,
                "recurrence_rate": 0.0,
                "reason": self.name,
                "metrics": [{} for _ in range(len(price_frame))],
            },
            index=price_frame.index,
        )


class ScoringTests(unittest.TestCase):
    def test_scores_are_bounded_and_serializable(self) -> None:
        prices = make_ohlcv(120)
        sector = make_ohlcv(120, seed=43, start_price=70.0)
        benchmark = make_ohlcv(120, seed=44, start_price=400.0)
        config = small_config()

        frame = calculate_mapi("TEST", prices, sector, benchmark, config)["short_term"]
        latest = calculate_latest_mapi("TEST", prices, sector, benchmark, config)

        self.assertTrue(frame["mapi_score"].between(0.0, 100.0).all())
        self.assertTrue(frame["mapi_direction"].between(-1.0, 1.0).all())
        self.assertTrue(frame["mapi_confidence"].between(0.0, 1.0).all())
        payload = latest["mapi_short_term"]
        self.assertEqual(payload["symbol"], "TEST")
        self.assertEqual(payload["signal_version"], ALGORITHM_REVISION)
        self.assertEqual(payload["algorithm_revision"], ALGORITHM_REVISION)
        self.assertEqual(payload["data_contract_version"], DATA_CONTRACT_VERSION)
        self.assertEqual(len(payload["config_fingerprint"]), 64)
        self.assertIsInstance(payload["anomaly_components"], list)
        self.assertIn("human_summary", payload)

    def test_output_is_deterministic(self) -> None:
        prices = make_ohlcv(100)
        config = small_config(include_cross_asset=False)

        first = calculate_mapi("TEST", prices, config=config)["short_term"]
        second = calculate_mapi("TEST", prices, config=config)["short_term"]

        pd.testing.assert_series_equal(first["mapi_score"], second["mapi_score"])
        pd.testing.assert_series_equal(first["mapi_direction"], second["mapi_direction"])
        self.assertEqual(first.iloc[-1]["human_summary"], second.iloc[-1]["human_summary"])

    def test_v03_separates_intensity_alert_and_actionability(self) -> None:
        config = small_config(include_cross_asset=False)
        frame = calculate_mapi("TEST", make_ohlcv(120, seed=55), config=config)[
            "short_term"
        ]
        pd.testing.assert_series_equal(
            frame["mapi_score"], frame["mapi_intensity_score"], check_names=False
        )
        pd.testing.assert_series_equal(
            frame["mapi_raw_score"], frame["mapi_intensity_score"], check_names=False
        )
        self.assertTrue(
            (frame["mapi_alert_score"] <= frame["mapi_intensity_score"] + 1e-12).all()
        )
        self.assertTrue(
            (frame["mapi_actionability_score"] <= frame["mapi_alert_score"] + 1e-12).all()
        )

    def test_legacy_public_score_selector_does_not_relabel_algorithm(self) -> None:
        config = small_config(include_cross_asset=False)
        config.signal_version = "mapi_v0.1"
        config.score_semantics = "legacy_actionability"
        frame = calculate_mapi("TEST", make_ohlcv(120, seed=56), config=config)[
            "short_term"
        ]
        pd.testing.assert_series_equal(
            frame["mapi_score"],
            frame["mapi_actionability_score"],
            check_names=False,
        )
        self.assertTrue((frame["signal_version"] == ALGORITHM_REVISION).all())

    def test_recurrence_does_not_invalidate_persistent_intensity(self) -> None:
        class RecurringComponent:
            name = "recurring_component"
            family = "test"

            def calculate(
                self,
                price_frame: pd.DataFrame,
                context: ComponentContext,
                horizon: HorizonConfig,
                config: MapiConfig,
            ) -> pd.DataFrame:
                novelty = np.r_[1.0, np.zeros(len(price_frame) - 1)]
                return pd.DataFrame(
                    {
                        "anomaly_strength": 0.8,
                        "direction": 1.0,
                        "confidence": 1.0,
                        "novelty": novelty,
                        "reason": "persistent",
                        "metrics": [{} for _ in range(len(price_frame))],
                    },
                    index=price_frame.index,
                )

        config = MapiConfig()
        config.enabled_components = ["recurring_component"]
        config.component_weights = {"recurring_component": 1.0}
        config.component_reliability = {"recurring_component": 1.0}
        config.horizons = {
            "test": HorizonConfig("test", 1, 12, 4, expected_frequency="daily")
        }
        config.redundancy_window = 10
        config.redundancy_min_periods = 4
        frame = calculate_mapi(
            "TEST",
            make_ohlcv(30, seed=57),
            config=config,
            components=[RecurringComponent()],
        )["test"]
        latest = frame.iloc[-1]
        self.assertGreater(float(latest["mapi_intensity_score"]), 70.0)
        self.assertEqual(float(latest["mapi_alert_score"]), 0.0)
        self.assertEqual(float(latest["mapi_actionability_score"]), 0.0)
        self.assertEqual(latest["anomaly_state"], "confirmed")
        self.assertGreater(int(latest["anomaly_age_bars"]), 1)
        self.assertTrue(latest["dominant_intensity_anomalies"])
        self.assertEqual(latest["dominant_alert_anomalies"], [])
        self.assertEqual(
            latest["dominant_anomalies"],
            latest["dominant_intensity_anomalies"],
        )
        self.assertIn("no longer novel", latest["human_summary"])

    def test_no_view_component_does_not_dilute_bullish_direction(self) -> None:
        bullish = _FixedDirectionComponent(
            "bullish", 1.0, 1.0, "continuation_hypothesis_test_bullish"
        )
        no_view = _FixedDirectionComponent(
            "no_view", 0.0, 0.0, "direction_neutral_test_no_view"
        )
        frame = self._score_fixed_components([bullish, no_view])
        latest = frame.iloc[-1]
        self.assertAlmostEqual(float(latest["mapi_forecast_direction"]), 1.0)
        self.assertEqual(
            latest["mapi_direction_semantics"],
            "weighted_component_forecast_direction_excluding_no_view",
        )
        serialized = latest["signal"].to_dict()["anomaly_components"]
        self.assertEqual(serialized[0]["directional_evidence_strength"], 1.0)
        self.assertEqual(serialized[1]["directional_evidence_strength"], 0.0)

    def test_opposing_directional_components_cancel(self) -> None:
        bullish = _FixedDirectionComponent(
            "bullish", 1.0, 1.0, "continuation_hypothesis_test_bullish"
        )
        bearish = _FixedDirectionComponent(
            "bearish", -1.0, 1.0, "continuation_hypothesis_test_bearish"
        )
        frame = self._score_fixed_components([bullish, bearish])
        self.assertAlmostEqual(
            float(frame["mapi_forecast_direction"].iloc[-1]), 0.0
        )

    def _score_fixed_components(
        self, components: list[_FixedDirectionComponent]
    ) -> pd.DataFrame:
        config = MapiConfig()
        config.enabled_components = [component.name for component in components]
        config.component_weights = {component.name: 1.0 for component in components}
        config.component_reliability = {
            component.name: 1.0 for component in components
        }
        config.horizons = {
            "test": HorizonConfig(
                "test", 1, 12, 4, expected_frequency="daily"
            )
        }
        config.redundancy_window = 10
        config.redundancy_min_periods = 4
        return calculate_mapi(
            "TEST",
            make_ohlcv(30, seed=58),
            config=config,
            components=components,
        )["test"]


if __name__ == "__main__":
    unittest.main()
