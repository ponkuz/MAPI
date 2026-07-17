from __future__ import annotations

import unittest

from mapi.scoring import calculate_latest_mapi
from tests.helpers import make_ohlcv, small_config


class MissingDataTests(unittest.TestCase):
    def test_optional_cross_asset_data_degrades_gracefully(self) -> None:
        result = calculate_latest_mapi(
            "TEST", make_ohlcv(100), config=small_config()
        )["mapi_short_term"]
        components = {item["name"]: item for item in result["anomaly_components"]}

        self.assertEqual(components["stock_sector_divergence"]["confidence"], 0.0)
        self.assertEqual(components["market_regime_divergence"]["confidence"], 0.0)
        self.assertGreaterEqual(result["mapi_score"], 0.0)
        self.assertLessEqual(result["mapi_score"], 100.0)
        self.assertGreater(result["mapi_confidence"], 0.0)


if __name__ == "__main__":
    unittest.main()
