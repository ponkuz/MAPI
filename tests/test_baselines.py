from __future__ import annotations

import unittest

import pandas as pd

from mapi.research.baselines import generate_baselines, random_control_distribution
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


if __name__ == "__main__":
    unittest.main()
