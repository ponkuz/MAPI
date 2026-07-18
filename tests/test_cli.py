from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tests.helpers import make_ohlcv
from mapi.version import ALGORITHM_REVISION, DATA_CONTRACT_VERSION


class CliEndToEndTests(unittest.TestCase):
    def test_cli_tools_write_finite_reproducible_json(self) -> None:
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            stock_path = temp / "TEST.csv"
            sector_path = temp / "SECTOR.csv"
            benchmark_path = temp / "MARKET.csv"
            make_ohlcv(140, seed=70).to_csv(stock_path, index=False)
            make_ohlcv(140, seed=71, start_price=80.0).to_csv(
                sector_path, index=False
            )
            make_ohlcv(140, seed=72, start_price=420.0).to_csv(
                benchmark_path, index=False
            )

            first_mapi = temp / "mapi_first.json"
            second_mapi = temp / "mapi_second.json"
            common_mapi = [
                sys.executable,
                str(root / "examples" / "run_mapi.py"),
                "--symbol",
                "TEST",
                "--prices",
                str(stock_path),
                "--sector",
                str(sector_path),
                "--benchmark",
                str(benchmark_path),
                "--config",
                str(root / "configs" / "mapi_v0_3.yaml"),
                "--log-level",
                "WARNING",
            ]
            self._run(common_mapi + ["--output", str(first_mapi)], root)
            self._run(common_mapi + ["--output", str(second_mapi)], root)
            self.assertEqual(
                first_mapi.read_text(encoding="utf-8"),
                second_mapi.read_text(encoding="utf-8"),
            )
            mapi_payload = json.loads(first_mapi.read_text(encoding="utf-8"))
            self.assertIn("mapi_short_term", mapi_payload)
            self.assertEqual(
                mapi_payload["mapi_short_term"]["signal_version"],
                ALGORITHM_REVISION,
            )
            self.assertEqual(
                mapi_payload["mapi_short_term"]["data_contract_version"],
                DATA_CONTRACT_VERSION,
            )

            backtest_path = temp / "event_study.json"
            self._run(
                [
                    sys.executable,
                    str(root / "examples" / "run_backtest.py"),
                    "--symbol",
                    "TEST",
                    "--prices",
                    str(stock_path),
                    "--sector",
                    str(sector_path),
                    "--benchmark",
                    str(benchmark_path),
                    "--config",
                    str(root / "configs" / "mapi_v0_3.yaml"),
                    "--horizon",
                    "short_term",
                    "--output",
                    str(backtest_path),
                    "--log-level",
                    "WARNING",
                ],
                root,
            )
            backtest_payload = json.loads(backtest_path.read_text(encoding="utf-8"))
            self.assertEqual(
                backtest_payload["score_column_used"], "mapi_actionability_score"
            )
            partition = backtest_payload["evaluation_partition"]
            self.assertEqual(partition["fit_count"] + partition["test_count"], 140)
            self.assertLess(partition["fit_end"], partition["test_start"])
            self.assertAlmostEqual(
                partition["fit_fraction"] + partition["test_fraction"], 1.0
            )
            self.assertEqual(
                backtest_payload["metrics"]["score_column_used"],
                backtest_payload["score_column_used"],
            )
            self.assertTrue(
                all(
                    row["score_column_used"] == backtest_payload["score_column_used"]
                    for row in backtest_payload["score_buckets"]
                )
            )
            full_mapi = next(
                row for row in backtest_payload["ablation"]
                if row["variant"] == "full_mapi"
            )
            self.assertEqual(
                backtest_payload["metrics"]["sample_count"],
                full_mapi["sample_count"],
            )
            self.assertEqual(
                backtest_payload["metrics"]["direction_column_used"],
                "mapi_forecast_direction",
            )
            event_rows = [
                backtest_payload["metrics"],
                *backtest_payload["score_buckets"],
                *[
                    row
                    for row in backtest_payload["baseline_comparisons"]
                    if row["analysis_type"] == "event_study"
                ],
                *backtest_payload["random_control_distribution"],
                *backtest_payload["ablation"],
            ]
            for row in event_rows:
                self.assertEqual(row["evaluation_count"], partition["test_count"])
                self.assertEqual(row["evaluation_start"], partition["test_start"])
                self.assertEqual(row["evaluation_end"], partition["test_end"])
                self.assertTrue(
                    all(
                        partition["test_start"] <= timestamp <= partition["test_end"]
                        for timestamp in row["selected_event_timestamps"]
                    )
                )
            self.assertTrue(
                all(
                    row["reference_score_column"]
                    == backtest_payload["score_column_used"]
                    for row in backtest_payload["baseline_comparisons"]
                )
            )
            self.assertTrue(
                all(
                    row["reference_score_column"]
                    == backtest_payload["score_column_used"]
                    for row in backtest_payload["random_control_distribution"]
                )
            )
            self.assertTrue(
                all(
                    row["score_column_used"] == backtest_payload["score_column_used"]
                    for row in backtest_payload["ablation"]
                )
            )
            self.assertEqual(backtest_payload["metrics"]["analysis_type"], "event_study")
            self.assertEqual(len(backtest_payload["random_control_distribution"]), 10)
            baseline_names = {
                row["name"] for row in backtest_payload["baseline_comparisons"]
            }
            self.assertIn("shuffled_mapi_scores", baseline_names)
            self.assertIn("sector_relative_strength", baseline_names)

    def _run(self, command: list[str], cwd: Path) -> None:
        completed = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        if completed.returncode != 0:
            self.fail(
                f"Command failed ({completed.returncode}): {' '.join(command)}\n"
                f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )


if __name__ == "__main__":
    unittest.main()
