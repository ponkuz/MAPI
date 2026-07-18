from __future__ import annotations

import unittest
from types import SimpleNamespace

import pandas as pd

from mapi.research.baselines import (
    compare_baselines,
    generate_baselines,
    random_control_distribution,
)
from mapi.scoring import calculate_mapi
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
                "mapi_actionability_score": [20.0] * 84 + [80.0] * 36,
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
        self.assertIn("always_long_fixed_horizon", names)
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


if __name__ == "__main__":
    unittest.main()
