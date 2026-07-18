from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd

from mapi.config import MapiConfig
from mapi.models import HorizonConfig
from mapi.research.ablation import run_ablation
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


if __name__ == "__main__":
    unittest.main()
