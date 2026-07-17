from __future__ import annotations

import unittest

import pandas as pd

from mapi.scoring import calculate_latest_mapi, calculate_mapi
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
        self.assertEqual(payload["signal_version"], "mapi_v0.2")
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

    def test_v02_separates_anomaly_intensity_from_actionability(self) -> None:
        config = small_config(include_cross_asset=False)
        frame = calculate_mapi("TEST", make_ohlcv(120, seed=55), config=config)[
            "short_term"
        ]
        pd.testing.assert_series_equal(
            frame["mapi_score"], frame["mapi_raw_score"], check_names=False
        )
        self.assertTrue(
            (frame["mapi_actionability_score"] <= frame["mapi_raw_score"] + 1e-12).all()
        )

    def test_v01_legacy_score_semantics_remain_available(self) -> None:
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


if __name__ == "__main__":
    unittest.main()
