from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from mapi.redundancy import compute_redundancy_penalties


def component_frame(values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"anomaly_strength": values})


class RedundancyTests(unittest.TestCase):
    def test_penalty_is_symmetric_and_order_invariant(self) -> None:
        values = np.linspace(0.0, 1.0, 30)
        frames = {"left": component_frame(values), "right": component_frame(values * 0.9)}
        forward = compute_redundancy_penalties(frames, 12, 0.8, 0.55, min_periods=4)
        reverse = compute_redundancy_penalties(
            dict(reversed(list(frames.items()))), 12, 0.8, 0.55, min_periods=4
        )
        pd.testing.assert_series_equal(forward["left"], forward["right"])
        pd.testing.assert_series_equal(forward["left"], reverse["left"])
        pd.testing.assert_series_equal(forward["right"], reverse["right"])

    def test_negative_correlation_is_redundant(self) -> None:
        values = np.linspace(0.0, 1.0, 30)
        penalties = compute_redundancy_penalties(
            {"left": component_frame(values), "right": component_frame(1.0 - values)},
            12,
            0.8,
            0.55,
            min_periods=4,
        )
        self.assertLess(float(penalties["left"].iloc[-1]), 1.0)

    def test_zero_variance_and_unavailable_components_are_not_penalized(self) -> None:
        varying = np.linspace(0.0, 1.0, 30)
        penalties = compute_redundancy_penalties(
            {
                "varying": component_frame(varying),
                "constant": component_frame(np.zeros(30)),
            },
            12,
            0.8,
            0.55,
            min_periods=4,
        )
        self.assertTrue((penalties["varying"] == 1.0).all())
        self.assertTrue((penalties["constant"] == 1.0).all())

    def test_current_observation_only_affects_next_penalty(self) -> None:
        left = np.linspace(0.0, 1.0, 30)
        right = left.copy()
        changed = right.copy()
        changed[-1] = 0.0
        original = compute_redundancy_penalties(
            {"left": component_frame(left), "right": component_frame(right)},
            12,
            0.8,
            0.55,
            min_periods=4,
        )
        mutated = compute_redundancy_penalties(
            {"left": component_frame(left), "right": component_frame(changed)},
            12,
            0.8,
            0.55,
            min_periods=4,
        )
        self.assertEqual(
            float(original["left"].iloc[-1]), float(mutated["left"].iloc[-1])
        )


if __name__ == "__main__":
    unittest.main()
