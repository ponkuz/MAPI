from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from mapi.normalization import event_novelty


class NoveltyTests(unittest.TestCase):
    def test_constant_sequence_is_a_tie_and_not_novel(self) -> None:
        result = event_novelty(pd.Series([0.8] * 20), window=10, min_periods=4)
        self.assertAlmostEqual(float(result["historical_extremeness"].iloc[-1]), 0.5)
        self.assertAlmostEqual(float(result["recurrence_rate"].iloc[-1]), 1.0)
        self.assertAlmostEqual(float(result["novelty"].iloc[-1]), 0.0)

    def test_repeated_near_identical_anomalies_lose_novelty(self) -> None:
        values = pd.Series([0.0] * 8 + [0.90, 0.91, 0.89, 0.90, 0.905])
        result = event_novelty(
            values, window=12, min_periods=4, recurrence_tolerance=0.02
        )
        first = float(result["novelty"].iloc[8])
        last = float(result["novelty"].iloc[-1])
        self.assertGreater(first, last)
        self.assertGreater(float(result["recurrence_rate"].iloc[-1]), 0.0)

    def test_gradually_increasing_anomalies_remain_extreme(self) -> None:
        values = pd.Series(np.arange(0.0, 2.0, 0.1))
        result = event_novelty(
            values, window=10, min_periods=4, recurrence_tolerance=0.02
        )
        self.assertAlmostEqual(float(result["historical_extremeness"].iloc[-1]), 1.0)
        self.assertAlmostEqual(float(result["novelty"].iloc[-1]), 1.0)

    def test_one_off_anomaly_is_novel_against_prior_baseline(self) -> None:
        values = pd.Series([0.1] * 12 + [1.0])
        result = event_novelty(values, window=12, min_periods=4)
        self.assertAlmostEqual(float(result["historical_extremeness"].iloc[-1]), 1.0)
        self.assertAlmostEqual(float(result["recurrence_rate"].iloc[-1]), 0.0)
        self.assertAlmostEqual(float(result["novelty"].iloc[-1]), 1.0)


if __name__ == "__main__":
    unittest.main()
