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


if __name__ == "__main__":
    unittest.main()
