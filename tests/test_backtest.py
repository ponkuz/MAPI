from __future__ import annotations

import unittest

import pandas as pd

from mapi.data.validation import normalize_ohlcv
from mapi.research.backtest import compare_score_buckets, run_backtest, run_event_study
from mapi.research.labels import forward_path_metrics
from mapi.research.reports import render_markdown_report
from tests.helpers import make_ohlcv


class BacktestTests(unittest.TestCase):
    def setUp(self) -> None:
        raw = make_ohlcv(80, seed=90)
        raw["close"] = 100.0 + pd.Series(range(len(raw)), dtype=float) * 0.25
        raw["open"] = raw["close"] - 0.05
        raw["high"] = raw["close"] + 0.20
        raw["low"] = raw["close"] - 0.20
        self.prices = normalize_ohlcv(raw)

    def test_costs_reduce_mean_return(self) -> None:
        signals = pd.DataFrame(
            {"mapi_score": 100.0, "mapi_direction": 1.0}, index=self.prices.index
        )
        free = run_backtest(
            signals,
            self.prices,
            horizon_bars=3,
            transaction_cost_bps=0.0,
            spread_bps=0.0,
            slippage_bps=0.0,
        )
        costly = run_backtest(
            signals,
            self.prices,
            horizon_bars=3,
            transaction_cost_bps=100.0,
            spread_bps=0.0,
            slippage_bps=25.0,
        )
        self.assertEqual(free.sample_count, costly.sample_count)
        self.assertGreater(free.mean_return, costly.mean_return)

    def test_score_buckets_do_not_leak_signals(self) -> None:
        signals = pd.DataFrame(
            {"mapi_score": 30.0, "mapi_direction": 1.0}, index=self.prices.index
        )
        buckets = compare_score_buckets(
            signals,
            self.prices,
            horizon_bars=2,
            buckets=(0, 20, 40),
            transaction_cost_bps=0.0,
            spread_bps=0.0,
            slippage_bps=0.0,
        ).set_index("bucket")
        self.assertEqual(int(buckets.loc["0-19", "sample_count"]), 0)
        self.assertGreater(int(buckets.loc["20-39", "sample_count"]), 0)

    def test_markdown_report_has_no_optional_dependency(self) -> None:
        rows = pd.DataFrame([{"name": "mapi", "mean_return": 0.0123}])
        report = render_markdown_report("Test", rows)
        self.assertIn("| name | mean_return |", report)
        self.assertIn("experimental", report.lower())

    def test_hand_calculated_long_and_short_execution(self) -> None:
        timestamps = pd.date_range("2025-01-02", periods=7, freq="B", tz="UTC")
        close = pd.Series([100.0, 110.0, 121.0, 108.9, 100.0, 100.0, 100.0])
        prices = normalize_ohlcv(
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "open": close,
                    "high": close + 0.1,
                    "low": close - 0.1,
                    "close": close,
                    "volume": 1000.0,
                }
            )
        )
        labels = forward_path_metrics(prices, horizon_bars=2, signal_delay_bars=1)
        self.assertAlmostEqual(float(labels.iloc[0]["forward_return"]), -0.01)
        self.assertAlmostEqual(float(labels.iloc[0]["intrabar_mfe"]), 121.1 / 110.0 - 1.0)
        self.assertAlmostEqual(float(labels.iloc[0]["intrabar_mae"]), 108.8 / 110.0 - 1.0)

        long_signals = pd.DataFrame(
            {"mapi_score": [100.0] + [0.0] * 6, "mapi_direction": [1.0] + [0.0] * 6},
            index=prices.index,
        )
        short_signals = long_signals.copy()
        short_signals.loc[prices.index[0], "mapi_direction"] = -1.0
        long_metrics = run_event_study(
            long_signals,
            prices,
            horizon_bars=2,
            transaction_cost_bps=0.0,
            spread_bps=0.0,
            slippage_bps=0.0,
        )
        short_metrics = run_event_study(
            short_signals,
            prices,
            horizon_bars=2,
            transaction_cost_bps=0.0,
            spread_bps=0.0,
            slippage_bps=0.0,
        )
        self.assertAlmostEqual(long_metrics.mean_return, -0.01)
        self.assertAlmostEqual(long_metrics.mean_intrabar_mfe, 121.1 / 110.0 - 1.0)
        self.assertAlmostEqual(long_metrics.mean_intrabar_mae, 108.8 / 110.0 - 1.0)
        self.assertAlmostEqual(short_metrics.mean_return, 0.01)
        self.assertAlmostEqual(short_metrics.mean_intrabar_mfe, -(108.8 / 110.0 - 1.0))
        self.assertAlmostEqual(short_metrics.mean_intrabar_mae, -(121.1 / 110.0 - 1.0))

    def test_event_statistics_are_unannualized_and_bootstrapped(self) -> None:
        signals = pd.DataFrame(
            {"mapi_score": 100.0, "mapi_direction": 1.0}, index=self.prices.index
        )
        metrics = run_event_study(
            signals,
            self.prices,
            horizon_bars=3,
            transaction_cost_bps=0.0,
            spread_bps=0.0,
            slippage_bps=0.0,
            bootstrap_samples=200,
        )
        self.assertEqual(metrics.bootstrap_samples, 200)
        self.assertLessEqual(metrics.mean_return_ci_lower, metrics.mean_return)
        self.assertGreaterEqual(metrics.mean_return_ci_upper, metrics.mean_return)
        self.assertAlmostEqual(metrics.sharpe_ratio, metrics.event_return_mean_to_std)
        payload = metrics.to_dict()
        self.assertIn("gross_directional_accuracy", payload)
        self.assertIn("large_move_capture_rate", payload)
        self.assertTrue(any("compatibility aliases" in item for item in payload["warnings"]))

    def test_intrabar_breakout_uses_high_low_not_close(self) -> None:
        timestamps = pd.date_range("2025-02-03", periods=10, freq="B", tz="UTC")
        close = pd.Series([100.0] * 10)
        high = pd.Series([101.0] * 10)
        low = pd.Series([99.0] * 10)
        high.iloc[7] = 102.0
        prices = normalize_ohlcv(
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "open": close,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": 1000.0,
                }
            )
        )
        labels = forward_path_metrics(prices, horizon_bars=2, signal_delay_bars=1)
        self.assertEqual(float(labels.iloc[5]["forward_return"]), 0.0)
        self.assertEqual(float(labels.iloc[5]["intrabar_breakout"]), 1.0)

    def test_intrabar_excursions_include_zero_at_entry(self) -> None:
        timestamps = pd.date_range("2025-03-03", periods=5, freq="B", tz="UTC")
        falling = normalize_ohlcv(
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "open": [100.0, 100.0, 96.0, 95.0, 95.0],
                    "high": [101.0, 101.0, 99.0, 98.0, 96.0],
                    "low": [99.0, 99.0, 94.0, 93.0, 94.0],
                    "close": [100.0, 100.0, 95.0, 94.0, 95.0],
                    "volume": 1000.0,
                }
            )
        )
        rising = normalize_ohlcv(
            pd.DataFrame(
                {
                    "timestamp": timestamps,
                    "open": [100.0, 100.0, 104.0, 105.0, 105.0],
                    "high": [101.0, 101.0, 106.0, 107.0, 106.0],
                    "low": [99.0, 99.0, 101.0, 102.0, 104.0],
                    "close": [100.0, 100.0, 105.0, 106.0, 105.0],
                    "volume": 1000.0,
                }
            )
        )
        falling_labels = forward_path_metrics(falling, 2, 1)
        rising_labels = forward_path_metrics(rising, 2, 1)
        self.assertEqual(float(falling_labels.iloc[0]["intrabar_mfe"]), 0.0)
        self.assertEqual(float(rising_labels.iloc[0]["intrabar_mae"]), 0.0)

    def test_evidence_quality_gates_and_exclusion_counts(self) -> None:
        signals = pd.DataFrame(
            {
                "mapi_score": 100.0,
                "mapi_direction": 1.0,
                "mapi_confidence": [0.1, 1.0, 1.0, 1.0] * 20,
                "data_quality_score": [1.0, 0.1, 1.0, 1.0] * 20,
                "horizon_frequency_compatible": [True, True, False, True] * 20,
            },
            index=self.prices.index,
        )
        metrics = run_event_study(
            signals,
            self.prices,
            horizon_bars=1,
            min_confidence=0.5,
            min_data_quality=0.5,
            require_frequency_compatible=True,
            transaction_cost_bps=0.0,
            spread_bps=0.0,
            slippage_bps=0.0,
        )
        self.assertGreater(metrics.sample_count, 0)
        self.assertEqual(metrics.excluded_low_confidence_count, 20)
        self.assertEqual(metrics.excluded_low_quality_count, 20)
        self.assertEqual(metrics.excluded_frequency_mismatch_count, 20)
        self.assertIn("active_event_ic", metrics.to_dict())

    def test_round_trip_cost_is_charged_once(self) -> None:
        signals = pd.DataFrame(
            {"mapi_score": 100.0, "mapi_direction": 1.0}, index=self.prices.index
        )
        free = run_event_study(
            signals,
            self.prices,
            horizon_bars=2,
            transaction_cost_bps=0.0,
            spread_bps=0.0,
            slippage_bps=0.0,
        )
        costly = run_event_study(
            signals,
            self.prices,
            horizon_bars=2,
            transaction_cost_bps=10.0,
            spread_bps=20.0,
            slippage_bps=30.0,
        )
        self.assertAlmostEqual(free.mean_return - costly.mean_return, 0.006)

    def test_non_overlapping_event_filter_and_warning(self) -> None:
        signals = pd.DataFrame(
            {"mapi_score": 100.0, "mapi_direction": 1.0}, index=self.prices.index
        )
        metrics = run_event_study(signals, self.prices, horizon_bars=5)
        self.assertTrue(metrics.non_overlapping)
        self.assertGreater(metrics.overlapping_candidates_excluded, 0)
        self.assertEqual(metrics.analysis_type, "event_study")
        self.assertTrue(any("not realizable portfolio" in item for item in metrics.warnings))


if __name__ == "__main__":
    unittest.main()
