from __future__ import annotations

import json
import unittest

import numpy as np
import pandas as pd

from mapi.data.validation import normalize_ohlcv, validate_ohlcv
from mapi.scoring import calculate_latest_mapi, calculate_mapi
from tests.helpers import make_ohlcv, small_config


class EdgeCaseTests(unittest.TestCase):
    def test_constant_prices_zero_volume_and_short_history_are_finite(self) -> None:
        frame = make_ohlcv(5)
        frame[["open", "high", "low", "close"]] = 100.0
        frame["volume"] = 0.0
        output = calculate_mapi(
            "TEST", frame, config=small_config(include_cross_asset=False)
        )["short_term"]
        numeric = output[["mapi_score", "mapi_direction", "mapi_confidence"]]
        self.assertTrue(np.isfinite(numeric.to_numpy()).all())
        self.assertLess(float(output.iloc[-1]["ohlcv_quality_score"]), 1.0)

    def test_nan_and_infinity_do_not_escape_json_schema(self) -> None:
        frame = make_ohlcv(60)
        frame.loc[10, "close"] = np.inf
        frame.loc[11, "volume"] = np.nan
        result = calculate_latest_mapi(
            "TEST", frame, config=small_config(include_cross_asset=False)
        )
        json.dumps(result, allow_nan=False)

    def test_split_like_jump_produces_strong_warning(self) -> None:
        frame = make_ohlcv(20)
        frame.loc[10:, ["open", "high", "low", "close"]] /= 4.0
        warnings = validate_ohlcv(frame)
        self.assertTrue(any("split- and dividend-adjusted" in item for item in warnings))

    def test_timezone_and_dst_normalize_to_unique_utc_timestamps(self) -> None:
        local = pd.DatetimeIndex(
            ["2025-03-07 16:00", "2025-03-10 16:00"]
        ).tz_localize("America/New_York")
        frame = pd.DataFrame(
            {
                "timestamp": local,
                "open": [100.0, 101.0],
                "high": [101.0, 102.0],
                "low": [99.0, 100.0],
                "close": [100.5, 101.5],
                "volume": [1000.0, 1200.0],
            }
        )
        normalized = normalize_ohlcv(frame)
        self.assertTrue(normalized.index.is_unique)
        self.assertEqual(normalized.index[0].hour, 21)
        self.assertEqual(normalized.index[1].hour, 20)

    def test_json_output_contains_v02_schema(self) -> None:
        payload = calculate_latest_mapi(
            "TEST", make_ohlcv(100), config=small_config(include_cross_asset=False)
        )["mapi_short_term"]
        required = {
            "mapi_score",
            "mapi_raw_score",
            "mapi_actionability_score",
            "mapi_direction",
            "mapi_confidence",
            "data_quality_score",
            "ohlcv_quality_score",
            "evidence_coverage_score",
            "already_realized_score",
            "anomaly_components",
            "machine_reasons",
            "human_summary",
            "signal_version",
        }
        self.assertTrue(required.issubset(payload))
        json.dumps(payload, allow_nan=False)


if __name__ == "__main__":
    unittest.main()
