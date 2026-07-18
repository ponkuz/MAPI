from __future__ import annotations

import unittest

import pandas as pd

from mapi.research.matching import (
    apply_frequency_match,
    candidate_frequency,
    chronological_masks,
    fit_frequency_matched_threshold,
    partition_metadata,
)


class FrequencyMatchingTests(unittest.TestCase):
    def test_constant_scores_use_deterministic_tie_selection(self) -> None:
        index = pd.date_range("2025-01-02", periods=100, freq="B", tz="UTC")
        signals = pd.DataFrame(
            {
                "baseline_score": 100.0,
                "mapi_direction": 1.0,
                "mapi_confidence": 1.0,
                "data_quality_score": 1.0,
                "horizon_frequency_compatible": True,
            },
            index=index,
        )
        fit_mask, test_mask = chronological_masks(index, 0.70)
        match = fit_frequency_matched_threshold(
            signals, "baseline_score", 0.20, 0.10, fit_mask
        )
        selected_fit = apply_frequency_match(
            signals, "baseline_score", match, 0.10, fit_mask
        )
        selected_test = apply_frequency_match(
            signals, "baseline_score", match, 0.10, test_mask
        )
        self.assertEqual(int(selected_fit.sum()), 14)
        self.assertEqual(int(selected_test.sum()), 6)
        self.assertAlmostEqual(match.fitted_frequency, 0.20)
        self.assertAlmostEqual(
            candidate_frequency(
                signals,
                "baseline_score",
                match.threshold,
                0.10,
                test_mask,
                frequency_match=match,
            ),
            0.20,
        )

    def test_frequency_fit_uses_same_evidence_eligibility(self) -> None:
        index = pd.date_range("2025-01-02", periods=20, freq="B", tz="UTC")
        signals = pd.DataFrame(
            {
                "baseline_score": 100.0,
                "mapi_direction": 1.0,
                "mapi_confidence": [1.0, 0.0] * 10,
                "data_quality_score": 1.0,
                "horizon_frequency_compatible": [True, True, False, True] * 5,
            },
            index=index,
        )
        mask = pd.Series(True, index=index)
        frequency = candidate_frequency(
            signals,
            "baseline_score",
            60.0,
            0.10,
            mask,
            min_confidence=0.5,
            min_data_quality=0.5,
            require_frequency_compatible=True,
        )
        self.assertEqual(frequency, 0.25)

    def test_fitted_threshold_uses_fit_partition_only(self) -> None:
        index = pd.date_range("2025-01-02", periods=10, freq="B", tz="UTC")
        fit_mask, test_mask = chronological_masks(index, 0.60)
        signals = pd.DataFrame(
            {
                "baseline_score": [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
                + [70.0, 80.0, 90.0, 100.0],
                "mapi_direction": 1.0,
            },
            index=index,
        )
        original = fit_frequency_matched_threshold(
            signals, "baseline_score", 2.0 / 6.0, 0.10, fit_mask
        )
        changed_test = signals.copy()
        changed_test.loc[test_mask, "baseline_score"] = -1000.0
        test_mutation = fit_frequency_matched_threshold(
            changed_test, "baseline_score", 2.0 / 6.0, 0.10, fit_mask
        )
        changed_fit = signals.copy()
        changed_fit.loc[fit_mask, "baseline_score"] += 100.0
        fit_mutation = fit_frequency_matched_threshold(
            changed_fit, "baseline_score", 2.0 / 6.0, 0.10, fit_mask
        )
        self.assertEqual(original, test_mutation)
        self.assertNotEqual(original.threshold, fit_mutation.threshold)
        metadata = partition_metadata(index, fit_mask, test_mask, 0.60)
        self.assertEqual(metadata["fit_start"], index[0])
        self.assertEqual(metadata["fit_end"], index[5])
        self.assertEqual(metadata["test_start"], index[6])
        self.assertEqual(metadata["test_end"], index[-1])


if __name__ == "__main__":
    unittest.main()
