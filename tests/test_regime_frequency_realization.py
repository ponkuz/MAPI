from __future__ import annotations

import unittest

import pandas as pd

from mapi.config import MapiConfig
from mapi.data.frequency import validate_horizon_frequency
from mapi.data.validation import normalize_ohlcv
from mapi.models import HorizonConfig
from mapi.realization import directional_realization_score
from mapi.regimes import detect_market_regime
from mapi.scoring import calculate_mapi
from mapi.research.backtest import run_event_study
from tests.helpers import make_ohlcv, small_config


class RegimeFrequencyRealizationTests(unittest.TestCase):
    def test_stale_benchmark_is_unknown_without_stock_substitution(self) -> None:
        prices = normalize_ohlcv(make_ohlcv(120, seed=201))
        benchmark = normalize_ohlcv(make_ohlcv(120, seed=202)).iloc[:-3]
        horizon = HorizonConfig("test", 3, 30, 10)
        regimes = detect_market_regime(prices, benchmark, horizon, MapiConfig())
        self.assertEqual(regimes["regime"].iloc[-1], "unknown")
        self.assertEqual(regimes["regime_source"].iloc[-1], "benchmark_stale")
        self.assertEqual(float(regimes["regime_confidence"].iloc[-1]), 0.0)

    def test_stock_fallback_is_explicit_only_without_benchmark(self) -> None:
        prices = normalize_ohlcv(make_ohlcv(120, seed=203))
        horizon = HorizonConfig("test", 3, 30, 10)
        regimes = detect_market_regime(prices, None, horizon, MapiConfig())
        self.assertEqual(regimes["regime_source"].iloc[-1], "stock_fallback")
        self.assertAlmostEqual(float(regimes["regime_confidence"].iloc[-1]), 0.6)

    def test_stale_benchmark_neutralizes_regime_adjustment_only(self) -> None:
        prices = make_ohlcv(120, seed=205)
        sector = make_ohlcv(120, seed=206, start_price=80.0)
        benchmark = make_ohlcv(120, seed=207, start_price=420.0).iloc[:-3]
        frame = calculate_mapi(
            "TEST", prices, sector, benchmark, small_config()
        )["short_term"]
        latest = frame.iloc[-1]
        self.assertEqual(latest["regime_source"], "benchmark_stale")
        self.assertEqual(float(latest["regime_confidence"]), 0.0)
        components = {
            component.name: component
            for component in latest["signal"].anomaly_components
        }
        self.assertEqual(components["market_regime_divergence"].confidence, 0.0)
        for name in (
            "price_volume_divergence",
            "momentum_disagreement",
            "volatility_anomaly",
        ):
            with self.subTest(component=name):
                self.assertGreater(components[name].confidence, 0.0)
                self.assertGreater(components[name].weight, 0.0)
                self.assertEqual(
                    components[name].weight_factors["regime"], 1.0
                )
        self.assertGreater(float(latest["mapi_intensity_score"]), 0.0)

    def test_daily_and_intraday_horizon_mismatches_are_detected(self) -> None:
        daily = pd.date_range("2025-01-01", periods=20, freq="B", tz="UTC")
        intraday = pd.date_range("2025-01-01", periods=20, freq="30min", tz="UTC")
        intraday_horizon = HorizonConfig(
            "intraday", 1, 10, 4, expected_frequency="intraday"
        )
        position_horizon = HorizonConfig(
            "position", 4, 10, 4, expected_frequency="daily"
        )
        self.assertFalse(validate_horizon_frequency(daily, intraday_horizon).compatible)
        self.assertFalse(validate_horizon_frequency(intraday, position_horizon).compatible)

    def test_frequency_mismatch_warns_or_fails_according_to_config(self) -> None:
        config = MapiConfig()
        config.horizons = {
            "intraday": HorizonConfig(
                "intraday", 1, 10, 4, expected_frequency="intraday"
            )
        }
        warned = calculate_mapi("TEST", make_ohlcv(30), config=config)["intraday"]
        self.assertFalse(bool(warned["horizon_frequency_compatible"].iloc[-1]))
        self.assertGreaterEqual(float(warned["mapi_confidence"].iloc[-1]), 0.0)
        self.assertIsNotNone(warned["horizon_warning"].iloc[-1])
        metrics = run_event_study(
            warned,
            make_ohlcv(30),
            horizon_bars=1,
            score_threshold=0.0,
            min_direction=0.0,
            require_frequency_compatible=True,
        )
        self.assertEqual(metrics.sample_count, 0)
        self.assertGreater(metrics.excluded_frequency_mismatch_count, 0)
        config.frequency_mismatch_policy = "error"
        with self.assertRaisesRegex(ValueError, "expects intraday"):
            calculate_mapi("TEST", make_ohlcv(30), config=config)

    def test_weekly_and_monthly_are_not_classified_as_daily(self) -> None:
        horizon = HorizonConfig(
            "daily", 2, 10, 4, expected_frequency="daily"
        )
        weekly = pd.date_range("2025-01-03", periods=12, freq="7D", tz="UTC")
        monthly = pd.date_range("2025-01-31", periods=12, freq="30D", tz="UTC")
        weekly_result = validate_horizon_frequency(weekly, horizon)
        monthly_result = validate_horizon_frequency(monthly, horizon)
        self.assertEqual(weekly_result.observed_frequency, "weekly")
        self.assertEqual(monthly_result.observed_frequency, "monthly")
        self.assertFalse(weekly_result.compatible)
        self.assertFalse(monthly_result.compatible)

    def test_directional_realization_penalizes_only_aligned_continuation(self) -> None:
        self.assertEqual(directional_realization_score(1.0, -0.03, 0.05), 0.0)
        self.assertEqual(directional_realization_score(-1.0, 0.03, 0.05), 0.0)
        self.assertAlmostEqual(directional_realization_score(1.0, 0.03, 0.05), 0.6)
        self.assertAlmostEqual(directional_realization_score(-1.0, -0.03, 0.05), 0.6)
        self.assertEqual(directional_realization_score(0.0, 0.50, 0.05), 0.0)


if __name__ == "__main__":
    unittest.main()
