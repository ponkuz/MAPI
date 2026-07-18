from __future__ import annotations

import unittest
from types import SimpleNamespace

import pandas as pd

from mapi.data.validation import normalize_ohlcv
from mapi.research.baselines import (
    compare_baselines,
    generate_baselines,
    random_control_distribution,
)
from mapi.scoring import calculate_mapi
from mapi.research.matching import (
    apply_frequency_match,
    chronological_masks,
    fit_frequency_matched_threshold,
)
from tests.helpers import make_ohlcv, small_config


class BaselineTests(unittest.TestCase):
    def test_required_negative_controls_are_generated(self) -> None:
        prices = make_ohlcv(120, seed=60)
        sector = make_ohlcv(120, seed=61, start_price=80.0)
        benchmark = make_ohlcv(120, seed=62, start_price=420.0)
        signals = calculate_mapi(
            "TEST", prices, sector, benchmark, small_config()
        )["short_term"]
        baselines = generate_baselines(
            prices,
            seed=42,
            signal_frequency=0.20,
            sector_frame=sector,
            mapi_signals=signals,
        )
        required = {
            "sector_relative_strength",
            "equal_weight_components",
            "shuffled_mapi_scores",
            "isolated_stock_sector_residual",
            "isolated_volatility_anomaly",
        }
        self.assertTrue(required.issubset(baselines))
        for name in required:
            self.assertEqual(len(baselines[name]), len(prices))

    def test_random_control_distribution_is_multi_seed_and_reproducible(self) -> None:
        prices = make_ohlcv(100, seed=63)
        first = random_control_distribution(
            prices,
            horizon_bars=3,
            seeds=(10, 11, 12, 13),
            signal_frequency=0.25,
            transaction_cost_bps=0.0,
            spread_bps=0.0,
            slippage_bps=0.0,
        )
        second = random_control_distribution(
            prices,
            horizon_bars=3,
            seeds=(10, 11, 12, 13),
            signal_frequency=0.25,
            transaction_cost_bps=0.0,
            spread_bps=0.0,
            slippage_bps=0.0,
        )
        self.assertEqual(first["seed"].tolist(), [10, 11, 12, 13])
        pd.testing.assert_frame_equal(first, second)

    def test_baselines_fit_individual_thresholds_and_separate_buy_and_hold(self) -> None:
        prices = make_ohlcv(120, seed=64)
        index = pd.DatetimeIndex(prices["timestamp"])
        mapi = pd.DataFrame(
            {
                "mapi_actionability_score": [
                    80.0 if index % 5 == 0 else 20.0 for index in range(120)
                ],
                "mapi_direction": [1.0] * 120,
                "mapi_confidence": [1.0] * 120,
                "signal": [SimpleNamespace(anomaly_components=[])] * 120,
            },
            index=index,
        )
        rows = compare_baselines(
            prices,
            horizon_bars=3,
            mapi_signals=mapi,
            reference_score_column="mapi_actionability_score",
            reference_score_threshold=60.0,
            transaction_cost_bps=0.0,
            spread_bps=0.0,
            slippage_bps=0.0,
            bootstrap_samples=50,
        )
        names = set(rows["name"])
        self.assertIn("same_event_times_always_long", names)
        self.assertIn("full_period_buy_and_hold", names)
        self.assertNotIn("buy_and_hold", names)
        event_rows = rows[rows["analysis_type"] == "event_study"]
        for column in (
            "fitted_threshold",
            "candidate_test_frequency",
            "selected_event_count",
            "excluded_overlap_count",
        ):
            self.assertTrue(event_rows[column].notna().all())
        always_long = event_rows[
            event_rows["name"] == "same_event_times_always_long"
        ].iloc[0]
        random_control = event_rows[
            event_rows["name"] == "random_same_frequency"
        ].iloc[0]
        self.assertGreater(int(always_long["selected_event_count"]), 0)
        self.assertGreater(int(random_control["selected_event_count"]), 0)

    def test_previous_day_control_uses_prior_score_and_direction(self) -> None:
        prices = normalize_ohlcv(make_ohlcv(80, seed=65))
        baseline = generate_baselines(prices)["previous_day_return"]
        returns = prices["close"].pct_change()
        position = 35
        self.assertEqual(
            float(baseline["mapi_direction"].iloc[position]),
            float(returns.shift(1).apply(lambda value: 0.0 if pd.isna(value) else value).apply(
                lambda value: 1.0 if value > 0 else -1.0 if value < 0 else 0.0
            ).iloc[position]),
        )
        changed = prices.copy()
        changed.iloc[position, changed.columns.get_loc("close")] *= 1.5
        changed.iloc[position, changed.columns.get_loc("high")] = max(
            changed["high"].iloc[position], changed["close"].iloc[position]
        )
        changed_baseline = generate_baselines(changed)["previous_day_return"]
        self.assertEqual(
            float(baseline["baseline_score"].iloc[position]),
            float(changed_baseline["baseline_score"].iloc[position]),
        )
        self.assertEqual(
            float(baseline["mapi_direction"].iloc[position]),
            float(changed_baseline["mapi_direction"].iloc[position]),
        )

    def test_shuffled_mapi_never_crosses_fit_test_boundary(self) -> None:
        prices = make_ohlcv(100, seed=66)
        index = pd.DatetimeIndex(prices["timestamp"])
        fit_mask, test_mask = chronological_masks(index, 0.70)
        signals = pd.DataFrame(
            {
                "mapi_score": range(100),
                "mapi_direction": [1.0] * 100,
                "mapi_confidence": [1.0] * 100,
                "signal": [SimpleNamespace(anomaly_components=[])] * 100,
            },
            index=index,
        )
        shuffled = generate_baselines(
            prices,
            seed=9,
            mapi_signals=signals,
            fit_mask=fit_mask,
            test_mask=test_mask,
        )["shuffled_mapi_scores"]
        self.assertEqual(
            sorted(shuffled.loc[fit_mask, "baseline_score"].tolist()),
            sorted(signals.loc[fit_mask, "mapi_score"].tolist()),
        )
        self.assertEqual(
            sorted(shuffled.loc[test_mask, "baseline_score"].tolist()),
            sorted(signals.loc[test_mask, "mapi_score"].tolist()),
        )

    def test_feasible_event_baselines_select_nonzero_matched_count(self) -> None:
        prices = make_ohlcv(140, seed=67)
        index = pd.DatetimeIndex(prices["timestamp"])
        fit_mask, test_mask = chronological_masks(index, 0.70)
        baselines = generate_baselines(prices, seed=12, signal_frequency=0.20)
        for name, signals in baselines.items():
            fit_eligible = (
                fit_mask
                & (signals["mapi_forecast_direction"].abs() >= 0.10)
            )
            test_eligible = (
                test_mask
                & (signals["mapi_forecast_direction"].abs() >= 0.10)
            )
            if not bool(fit_eligible.any()) or not bool(test_eligible.any()):
                continue
            with self.subTest(name=name):
                match = fit_frequency_matched_threshold(
                    signals, "baseline_score", 0.20, 0.10, fit_mask
                )
                selected = apply_frequency_match(
                    signals, "baseline_score", match, 0.10, test_mask
                )
                self.assertGreater(int(selected.sum()), 0)


if __name__ == "__main__":
    unittest.main()
