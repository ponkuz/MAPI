from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mapi.config import load_config
from mapi.data.csv_provider import CsvPriceDataProvider
from mapi.logging import configure_structured_logging
from mapi.scoring import calculate_latest_mapi


def _optional_csv(path: str | None) -> pd.DataFrame | None:
    return pd.read_csv(path) if path else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Calculate the latest MAPI signals from OHLCV CSV data.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--prices", required=True)
    parser.add_argument("--sector")
    parser.add_argument("--benchmark")
    parser.add_argument("--config", default="configs/mapi_v0_3.yaml")
    parser.add_argument("--output")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_structured_logging(args.log_level)
    logger = logging.getLogger("mapi.cli")
    config = load_config(args.config)
    prices = CsvPriceDataProvider(symbol_paths={args.symbol: args.prices}).get_ohlcv(
        args.symbol
    )
    result = calculate_latest_mapi(
        symbol=args.symbol,
        price_frame=prices,
        sector_frame=_optional_csv(args.sector),
        benchmark_frame=_optional_csv(args.benchmark),
        config=config,
    )
    payload = json.dumps(result, indent=2, ensure_ascii=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    logger.info(
        "MAPI calculation completed",
        extra={"event": "mapi_calculated", "symbol": args.symbol, "output": args.output},
    )
    print(payload)


if __name__ == "__main__":
    main()
