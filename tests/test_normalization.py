from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from mapi.config import load_config
from mapi.data.validation import normalize_ohlcv, validate_ohlcv
from mapi.normalization import historical_zscore
from tests.helpers import make_ohlcv


class NormalizationTests(unittest.TestCase):
    def test_normalize_sorts_and_deduplicates(self) -> None:
        frame = make_ohlcv(8)
        duplicate = frame.iloc[[3]].copy()
        duplicate["close"] = 999.0
        mixed = pd.concat([frame.iloc[::-1], duplicate], ignore_index=True)

        normalized = normalize_ohlcv(mixed)

        self.assertTrue(normalized.index.is_monotonic_increasing)
        self.assertTrue(normalized.index.is_unique)
        timestamp = pd.to_datetime(duplicate.iloc[0]["timestamp"], utc=True)
        self.assertEqual(float(normalized.loc[timestamp, "close"]), 999.0)

    def test_historical_zscore_uses_only_prior_baseline(self) -> None:
        values = pd.Series(np.arange(30, dtype=float))
        changed = values.copy()
        changed.iloc[20:] = 10_000.0

        original_z = historical_zscore(values, window=10, min_periods=5)
        changed_z = historical_zscore(changed, window=10, min_periods=5)

        pd.testing.assert_series_equal(original_z.iloc[:20], changed_z.iloc[:20])

    def test_default_yaml_loads_without_pyyaml(self) -> None:
        config = load_config("configs/mapi_v0_1.yaml")
        self.assertEqual(config.signal_version, "mapi_v0.1")
        self.assertEqual(config.horizons["position"].rolling_window, 252)
        self.assertAlmostEqual(config.component_weights["price_volume_divergence"], 0.24)

    def test_validation_reports_invalid_ranges(self) -> None:
        frame = make_ohlcv(5)
        frame.loc[0, "high"] = frame.loc[0, "low"] - 1.0
        warnings = validate_ohlcv(frame)
        self.assertTrue(any("inconsistent" in warning for warning in warnings))


if __name__ == "__main__":
    unittest.main()
