from __future__ import annotations

import unittest

import pandas as pd

from mapi.research.calibration import fit_confidence_calibrator


class CalibrationTests(unittest.TestCase):
    def setUp(self) -> None:
        index = pd.date_range("2024-01-01", periods=8, freq="B", tz="UTC")
        self.signals = pd.DataFrame(
            {"mapi_confidence": [0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9]},
            index=index,
        )
        self.success = pd.Series([0, 1, 0, 1, 1, 1, 0, 1], index=index)

    def test_calibrator_records_fit_period(self) -> None:
        model = fit_confidence_calibrator(self.signals, self.success)
        self.assertEqual(model.fitted_start, self.signals.index.min())
        self.assertEqual(model.fitted_end, self.signals.index.max())
        transformed = model.transform(self.signals["mapi_confidence"])
        self.assertTrue(transformed.between(0.0, 1.0).all())

    def test_calibrator_refuses_test_period_fit(self) -> None:
        with self.assertRaises(ValueError):
            fit_confidence_calibrator(
                self.signals, self.success, dataset_role="test"
            )


if __name__ == "__main__":
    unittest.main()
