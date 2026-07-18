from __future__ import annotations

import unittest

import pandas as pd

from mapi.components.base import ComponentContext, finalize_component_frame
from mapi.components.market_regime import MarketRegimeDivergence
from mapi.components.momentum_disagreement import MomentumDisagreement
from mapi.components.price_volume import (
    PriceVolumeDivergence,
    _price_volume_direction_contract,
)
from mapi.components.stock_sector import StockSectorDivergence
from mapi.components.volatility import VolatilityAnomaly
from mapi.data.validation import normalize_ohlcv
from tests.helpers import make_ohlcv, small_config


class ComponentTests(unittest.TestCase):
    def setUp(self) -> None:
        self.prices = normalize_ohlcv(make_ohlcv(90))
        self.config = small_config()
        self.horizon = self.config.horizons["short_term"]

    def test_price_volume_component_respects_output_contract(self) -> None:
        result = PriceVolumeDivergence().calculate(
            self.prices,
            ComponentContext(symbol="TEST"),
            self.horizon,
            self.config,
        )
        self.assertEqual(len(result), len(self.prices))
        for column in (
            "anomaly_strength",
            "confidence",
            "novelty",
            "historical_extremeness",
            "recurrence_rate",
        ):
            self.assertTrue(result[column].between(0.0, 1.0).all())
        self.assertTrue(result["direction"].between(-1.0, 1.0).all())

    def test_missing_sector_returns_zero_confidence(self) -> None:
        result = StockSectorDivergence().calculate(
            self.prices,
            ComponentContext(symbol="TEST", sector_frame=None),
            self.horizon,
            self.config,
        )
        self.assertTrue((result["confidence"] == 0.0).all())
        self.assertTrue(result["reason"].str.contains("unavailable").all())

    def test_momentum_divergence_maps_new_high_bearish_and_new_low_bullish(self) -> None:
        for closes, expected_direction, phrase in (
            ([100.0, 102.0, 104.0, 106.0, 100.0, 102.0, 105.0, 106.5], -1.0, "new high"),
            ([106.0, 104.0, 102.0, 100.0, 106.0, 104.0, 101.0, 99.5], 1.0, "new low"),
        ):
            with self.subTest(phrase=phrase):
                index = pd.date_range("2025-01-02", periods=len(closes), freq="B", tz="UTC")
                close = pd.Series(closes, index=index)
                prices = pd.DataFrame(
                    {
                        "open": close,
                        "high": close + 0.2,
                        "low": close - 0.2,
                        "close": close,
                        "volume": 1000.0,
                    },
                    index=index,
                )
                horizon = type(self.horizon)(
                    "test", return_window=1, rolling_window=7, min_periods=3
                )
                result = MomentumDisagreement().calculate(
                    prices,
                    ComponentContext(symbol="TEST"),
                    horizon,
                    self.config,
                )
                self.assertIn(phrase, result["reason"].iloc[-1].lower())
                self.assertEqual(
                    float(result["forecast_direction"].iloc[-1]),
                    expected_direction * 0.75,
                )
                self.assertEqual(
                    result["direction_semantics"].iloc[-1],
                    (
                        "reversal_hypothesis_bearish_new_high_nonconfirmation"
                        if expected_direction < 0.0
                        else "reversal_hypothesis_bullish_new_low_nonconfirmation"
                    ),
                )
                self.assertEqual(
                    float(result["directional_evidence_strength"].iloc[-1]), 1.0
                )

    def test_price_volume_subtypes_define_distinct_direction_contracts(self) -> None:
        index = pd.RangeIndex(6)
        zero = pd.Series(0.0, index=index)
        breakout = zero.copy()
        flat = zero.copy()
        low_volume = zero.copy()
        mismatch = zero.copy()
        breakout.iloc[0] = 1.0
        flat.iloc[1] = 0.8
        low_volume.iloc[2:4] = 0.7
        mismatch.iloc[4:6] = 0.6
        price_z = pd.Series([2.0, 0.0, 2.0, -2.0, -2.0, 2.0], index=index)
        flow_z = pd.Series([0.0, 0.0, 0.0, 0.0, 2.0, -2.0], index=index)

        direction, capacity, semantics, subtype = _price_volume_direction_contract(
            breakout,
            flat,
            low_volume,
            mismatch,
            price_z,
            flow_z,
        )

        self.assertEqual(
            subtype.tolist(),
            [
                "breakout_on_weak_volume",
                "high_volume_flat_price",
                "large_move_low_volume",
                "large_move_low_volume",
                "directional_flow_mismatch",
                "directional_flow_mismatch",
            ],
        )
        self.assertEqual(capacity.tolist(), [1.0, 0.0, 1.0, 1.0, 1.0, 1.0])
        self.assertLess(float(direction.iloc[0]), 0.0)
        self.assertEqual(float(direction.iloc[1]), 0.0)
        self.assertLess(float(direction.iloc[2]), 0.0)
        self.assertGreater(float(direction.iloc[3]), 0.0)
        self.assertGreater(float(direction.iloc[4]), 0.0)
        self.assertLess(float(direction.iloc[5]), 0.0)
        self.assertEqual(
            semantics.tolist(),
            [
                "reversal_hypothesis_bearish_weak_volume_breakout",
                "direction_neutral_high_volume_flat_price",
                "reversal_hypothesis_bearish_large_up_move_low_volume",
                "reversal_hypothesis_bullish_large_down_move_low_volume",
                "reversal_hypothesis_bullish_directional_flow_against_price",
                "reversal_hypothesis_bearish_directional_flow_against_price",
            ],
        )

    def test_high_volume_decline_is_directionally_confirmed_flow(self) -> None:
        raw = make_ohlcv(90, seed=77)
        previous_close = float(raw.loc[len(raw) - 2, "close"])
        raw.loc[len(raw) - 1, "open"] = previous_close
        raw.loc[len(raw) - 1, "close"] = previous_close * 0.90
        raw.loc[len(raw) - 1, "high"] = previous_close * 1.001
        raw.loc[len(raw) - 1, "low"] = previous_close * 0.899
        raw.loc[len(raw) - 1, "volume"] *= 12.0
        prices = normalize_ohlcv(raw)
        result = PriceVolumeDivergence().calculate(
            prices,
            ComponentContext(symbol="TEST"),
            self.horizon,
            self.config,
        )
        metrics = result["metrics"].iloc[-1]
        self.assertLess(metrics["price_z"], 0.0)
        self.assertGreater(metrics["volume_z"], 0.0)
        self.assertLess(metrics["directional_flow_z"], 0.0)
        self.assertEqual(metrics["directional_flow_mismatch"], 0.0)

    def test_enabled_components_define_explicit_direction_contracts(self) -> None:
        context = ComponentContext(
            symbol="TEST",
            sector_frame=self.prices,
            benchmark_frame=self.prices,
        )
        components = (
            PriceVolumeDivergence(),
            StockSectorDivergence(),
            MomentumDisagreement(),
            VolatilityAnomaly(),
            MarketRegimeDivergence(),
        )
        allowed_semantics = {
            "price_volume_divergence": {
                "reversal_hypothesis_bearish_weak_volume_breakout",
                "direction_neutral_high_volume_flat_price",
                "reversal_hypothesis_bullish_large_down_move_low_volume",
                "reversal_hypothesis_bearish_large_up_move_low_volume",
                "reversal_hypothesis_bullish_directional_flow_against_price",
                "reversal_hypothesis_bearish_directional_flow_against_price",
                "direction_neutral_price_volume_baseline",
            },
            "stock_sector_divergence": {
                "direction_neutral_correlation_breakdown",
                "continuation_hypothesis_beta_adjusted_residual",
                "direction_neutral_insufficient_residual_evidence",
            },
            "momentum_disagreement": {
                "reversal_hypothesis_bearish_new_high_nonconfirmation",
                "reversal_hypothesis_bullish_new_low_nonconfirmation",
                "continuation_hypothesis_short_term_momentum_dominance",
                "direction_neutral_momentum_alignment",
            },
            "volatility_anomaly": {
                "direction_neutral_volatility_compression",
                "direction_neutral_volatility_expansion_without_trend",
                "direction_neutral_abnormal_gap",
                "direction_neutral_volatility_baseline",
            },
            "market_regime_divergence": {
                "continuation_hypothesis_broad_market_relative_strength",
                "direction_neutral_broad_market_divergence",
            },
        }
        for component in components:
            with self.subTest(component=component.name):
                result = component.calculate(
                    self.prices, context, self.horizon, self.config
                )
                for column in (
                    "forecast_direction",
                    "observed_pressure",
                    "directional_evidence_strength",
                    "direction_semantics",
                    "direction_contract_warning",
                ):
                    self.assertIn(column, result)
                self.assertTrue(result["direction_contract_warning"].isna().all())
                actual_semantics = set(result["direction_semantics"].unique())
                self.assertTrue(
                    actual_semantics.issubset(allowed_semantics[component.name]),
                    f"Unexpected semantics for {component.name}: {actual_semantics}",
                )
                self.assertNotIn("hypothesized_forward_direction", actual_semantics)
                self.assertFalse(
                    any(value.startswith("deprecated_implicit") for value in actual_semantics)
                )
                self.assertTrue(
                    result["directional_evidence_strength"].between(0.0, 1.0).all()
                )
                neutral = result["direction_semantics"].str.startswith(
                    "direction_neutral_"
                )
                self.assertTrue(
                    (result.loc[neutral, "directional_evidence_strength"] == 0.0).all()
                )

    def test_volatility_is_direction_neutral_despite_observed_pressure(self) -> None:
        result = VolatilityAnomaly().calculate(
            self.prices,
            ComponentContext(symbol="TEST"),
            self.horizon,
            self.config,
        )
        self.assertTrue((result["forecast_direction"] == 0.0).all())
        self.assertTrue((result["observed_pressure"].abs() > 0.0).any())
        self.assertTrue(
            result["direction_semantics"].str.startswith("direction_neutral_").all()
        )

    def test_legacy_direction_fallback_emits_deterministic_diagnostic(self) -> None:
        legacy = pd.DataFrame(
            {
                "anomaly_strength": 0.5,
                "direction": 1.0,
                "confidence": 1.0,
                "novelty": 1.0,
            },
            index=self.prices.index,
        )
        result = finalize_component_frame(
            legacy, self.prices.index, "legacy component"
        )
        self.assertTrue(
            (
                result["direction_semantics"]
                == "deprecated_implicit_direction_fallback"
            ).all()
        )
        self.assertTrue(
            result["direction_contract_warning"].str.contains(
                "forecast_direction, observed_pressure, "
                "directional_evidence_strength, direction_semantics",
                regex=False,
            ).all()
        )


if __name__ == "__main__":
    unittest.main()
