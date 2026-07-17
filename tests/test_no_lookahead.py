from __future__ import annotations

import unittest

import pandas as pd

from mapi.components.base import ComponentContext
from mapi.components.market_regime import MarketRegimeDivergence
from mapi.components.momentum_disagreement import MomentumDisagreement
from mapi.components.price_volume import PriceVolumeDivergence
from mapi.components.stock_sector import StockSectorDivergence
from mapi.components.volatility import VolatilityAnomaly
from mapi.data.validation import normalize_ohlcv
from mapi.scoring import calculate_mapi
from tests.helpers import make_ohlcv, mutate_after, small_config


class NoLookaheadTests(unittest.TestCase):
    def test_future_mutation_does_not_change_past_signals(self) -> None:
        cutoff = 85
        prices = make_ohlcv(130, seed=10)
        sector = make_ohlcv(130, seed=11, start_price=75.0)
        benchmark = make_ohlcv(130, seed=12, start_price=410.0)
        config = small_config()

        original = calculate_mapi("TEST", prices, sector, benchmark, config)["short_term"]
        changed = calculate_mapi(
            "TEST",
            mutate_after(prices, cutoff),
            mutate_after(sector, cutoff),
            mutate_after(benchmark, cutoff),
            config,
        )["short_term"]

        for column in ("mapi_score", "mapi_direction", "mapi_confidence"):
            pd.testing.assert_series_equal(
                original[column].iloc[:cutoff],
                changed[column].iloc[:cutoff],
                check_exact=False,
                rtol=1e-12,
                atol=1e-12,
            )
        self.assertEqual(
            original["anomaly_state"].iloc[:cutoff].tolist(),
            changed["anomaly_state"].iloc[:cutoff].tolist(),
        )

    def test_each_component_is_unchanged_by_future_mutation(self) -> None:
        cutoff = 70
        prices = normalize_ohlcv(make_ohlcv(110, seed=30))
        sector = normalize_ohlcv(make_ohlcv(110, seed=31, start_price=80.0))
        benchmark = normalize_ohlcv(make_ohlcv(110, seed=32, start_price=420.0))
        changed_prices = normalize_ohlcv(mutate_after(prices.reset_index(drop=True), cutoff))
        changed_sector = normalize_ohlcv(mutate_after(sector.reset_index(drop=True), cutoff))
        changed_benchmark = normalize_ohlcv(
            mutate_after(benchmark.reset_index(drop=True), cutoff)
        )
        config = small_config()
        horizon = config.horizons["short_term"]
        component_cases = [
            (
                PriceVolumeDivergence(),
                ComponentContext(symbol="TEST"),
                prices,
                ComponentContext(symbol="TEST"),
                changed_prices,
            ),
            (
                MomentumDisagreement(),
                ComponentContext(symbol="TEST"),
                prices,
                ComponentContext(symbol="TEST"),
                changed_prices,
            ),
            (
                VolatilityAnomaly(),
                ComponentContext(symbol="TEST"),
                prices,
                ComponentContext(symbol="TEST"),
                changed_prices,
            ),
            (
                StockSectorDivergence(),
                ComponentContext(symbol="TEST", sector_frame=sector),
                prices,
                ComponentContext(symbol="TEST", sector_frame=changed_sector),
                changed_prices,
            ),
            (
                MarketRegimeDivergence(),
                ComponentContext(symbol="TEST", benchmark_frame=benchmark),
                prices,
                ComponentContext(symbol="TEST", benchmark_frame=changed_benchmark),
                changed_prices,
            ),
        ]
        for component, original_context, original_prices, changed_context, future_prices in component_cases:
            with self.subTest(component=component.name):
                original = component.calculate(
                    original_prices, original_context, horizon, config
                )
                changed = component.calculate(
                    future_prices, changed_context, horizon, config
                )
                for column in ("anomaly_strength", "direction", "confidence", "novelty"):
                    pd.testing.assert_series_equal(
                        original[column].iloc[:cutoff],
                        changed[column].iloc[:cutoff],
                        check_exact=False,
                        rtol=1e-12,
                        atol=1e-12,
                    )

    def test_already_realized_score_uses_no_future_rows(self) -> None:
        cutoff = 80
        prices = make_ohlcv(120, seed=33)
        config = small_config(include_cross_asset=False)
        original = calculate_mapi("TEST", prices, config=config)["short_term"]
        changed = calculate_mapi(
            "TEST", mutate_after(prices, cutoff), config=config
        )["short_term"]
        pd.testing.assert_series_equal(
            original["already_realized_score"].iloc[:cutoff],
            changed["already_realized_score"].iloc[:cutoff],
        )
        pd.testing.assert_series_equal(
            original["mapi_actionability_score"].iloc[:cutoff],
            changed["mapi_actionability_score"].iloc[:cutoff],
        )


if __name__ == "__main__":
    unittest.main()
