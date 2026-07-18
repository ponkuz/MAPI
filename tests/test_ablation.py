from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd

from mapi.config import MapiConfig
from mapi.models import HorizonConfig
from mapi.research.ablation import run_ablation
from mapi.research.matching import chronological_masks
from tests.helpers import make_ohlcv


class AblationTests(unittest.TestCase):
    def test_default_backtest_horizon_uses_named_config_window(self) -> None:
        for name, bars in (("swing", 20), ("position", 60)):
            with self.subTest(horizon=name):
                config = MapiConfig()
                config.horizons = {
                    name: HorizonConfig(name, bars, max(80, bars + 1), 20)
                }
                index = pd.DatetimeIndex(make_ohlcv(100)["timestamp"])
                signals = pd.DataFrame(
                    {
                        "mapi_score": 70.0,
                        "mapi_actionability_score": 65.0,
                        "mapi_direction": 1.0,
                    },
                    index=index,
                )
                observed: list[int] = []

                def fake_backtest(*args: object, **kwargs: object) -> SimpleNamespace:
                    observed.append(int(kwargs["horizon_bars"]))
                    return SimpleNamespace(
                        sample_count=2,
                        mean_return=0.01,
                        event_return_mean_to_std=0.2,
                        to_dict=lambda: {
                            "sample_count": 2,
                            "mean_return": 0.01,
                            "event_return_mean_to_std": 0.2,
                        },
                    )

                with patch(
                    "mapi.research.ablation.calculate_mapi",
                    return_value={name: signals},
                ), patch(
                    "mapi.research.ablation.run_backtest",
                    side_effect=fake_backtest,
                ):
                    run_ablation(
                        "TEST", make_ohlcv(100), None, None, config,
                        horizon_name=name,
                    )
                self.assertTrue(observed)
                self.assertTrue(all(value == bars for value in observed))

    def test_ablation_thresholds_ignore_test_period_score_mutation(self) -> None:
        prices = make_ohlcv(100, seed=69)
        index = pd.DatetimeIndex(prices["timestamp"])
        fit_mask, test_mask = chronological_masks(index, 0.70)
        config = MapiConfig()
        config.enabled_components = ["price_volume_divergence"]
        config.component_weights = {"price_volume_divergence": 1.0}
        config.component_reliability = {"price_volume_divergence": 1.0}
        config.horizons = {
            "short_term": HorizonConfig("short_term", 2, 30, 10)
        }
        config.redundancy_window = 24
        config.bootstrap_samples = 20
        signals = pd.DataFrame(
            {
                "mapi_actionability_score": np.linspace(0.0, 100.0, 100),
                "mapi_direction": 1.0,
                "mapi_confidence": 1.0,
                "data_quality_score": 1.0,
                "horizon_frequency_compatible": True,
            },
            index=index,
        )
        changed = signals.copy()
        changed.loc[test_mask, "mapi_actionability_score"] = 0.0

        def evaluate(frame: pd.DataFrame) -> pd.DataFrame:
            with patch(
                "mapi.research.ablation.calculate_mapi",
                return_value={"short_term": frame},
            ):
                return run_ablation(
                    "TEST",
                    prices,
                    None,
                    None,
                    config,
                    fit_mask=fit_mask,
                    test_mask=test_mask,
                )

        original = evaluate(signals)
        mutated = evaluate(changed)
        pd.testing.assert_series_equal(
            original["fitted_threshold"],
            mutated["fitted_threshold"],
        )
        self.assertTrue((original["evaluation_count"] == int(test_mask.sum())).all())


if __name__ == "__main__":
    unittest.main()
