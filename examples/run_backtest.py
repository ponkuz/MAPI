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
from mapi.models import json_safe
from mapi.research.backtest import compare_score_buckets, run_backtest
from mapi.research.ablation import run_ablation
from mapi.research.baselines import compare_baselines, random_control_distribution
from mapi.research.matching import (
    candidate_frequency,
    chronological_masks,
    partition_metadata,
)
from mapi.scoring import calculate_mapi
from mapi.version import (
    ALGORITHM_REVISION,
    DATA_CONTRACT_VERSION,
    IMPLEMENTATION_VERSION,
)


def _optional_csv(path: str | None) -> pd.DataFrame | None:
    return pd.read_csv(path) if path else None


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a cost-aware MAPI backtest from OHLCV CSV data.")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--prices", required=True)
    parser.add_argument("--sector")
    parser.add_argument("--benchmark")
    parser.add_argument("--config", default="configs/mapi_v0_3.yaml")
    parser.add_argument("--horizon", default="short_term")
    parser.add_argument("--score-threshold", type=float, default=60.0)
    parser.add_argument(
        "--score-column",
        choices=(
            "mapi_score",
            "mapi_raw_score",
            "mapi_intensity_score",
            "mapi_alert_score",
            "mapi_actionability_score",
        ),
    )
    parser.add_argument("--output")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()

    configure_structured_logging(args.log_level)
    logger = logging.getLogger("mapi.cli")
    config = load_config(args.config)
    if args.horizon not in config.horizons:
        raise ValueError(f"Unknown horizon: {args.horizon}")
    score_column_used = args.score_column or config.research_score_column
    direction_column_used = "mapi_forecast_direction"
    prices = CsvPriceDataProvider(symbol_paths={args.symbol: args.prices}).get_ohlcv(
        args.symbol
    )
    sector = _optional_csv(args.sector)
    benchmark = _optional_csv(args.benchmark)
    signals = calculate_mapi(
        symbol=args.symbol,
        price_frame=prices,
        sector_frame=sector,
        benchmark_frame=benchmark,
        config=config,
    )[args.horizon]
    fit_mask, test_mask = chronological_masks(
        signals.index, config.research_fit_fraction
    )
    partition = partition_metadata(
        signals.index, fit_mask, test_mask, config.research_fit_fraction
    )
    eligibility = {
        "min_confidence": config.research_min_confidence,
        "min_data_quality": config.research_min_data_quality,
        "require_frequency_compatible": (
            config.research_require_frequency_compatible
        ),
    }
    holding_bars = config.horizons[args.horizon].return_window
    metrics = run_backtest(
        signals,
        prices,
        horizon_bars=holding_bars,
        score_threshold=args.score_threshold,
        transaction_cost_bps=config.transaction_cost_bps,
        spread_bps=config.spread_bps,
        slippage_bps=config.slippage_bps,
        score_column=score_column_used,
        large_move_threshold=config.large_move_threshold,
        bootstrap_samples=config.bootstrap_samples,
        bootstrap_seed=config.deterministic_seed,
        evaluation_mask=test_mask,
        direction_column=direction_column_used,
        **eligibility,
    )
    buckets = compare_score_buckets(
        signals,
        prices,
        horizon_bars=holding_bars,
        transaction_cost_bps=config.transaction_cost_bps,
        spread_bps=config.spread_bps,
        slippage_bps=config.slippage_bps,
        score_column=score_column_used,
        large_move_threshold=config.large_move_threshold,
        bootstrap_samples=config.bootstrap_samples,
        bootstrap_seed=config.deterministic_seed,
        evaluation_mask=test_mask,
        direction_column=direction_column_used,
        **eligibility,
    )
    fit_signal_frequency = candidate_frequency(
        signals,
        score_column_used,
        args.score_threshold,
        0.10,
        fit_mask,
        direction_column=direction_column_used,
        **eligibility,
    )
    test_signal_frequency = candidate_frequency(
        signals,
        score_column_used,
        args.score_threshold,
        0.10,
        test_mask,
        direction_column=direction_column_used,
        **eligibility,
    )
    baseline_rows = compare_baselines(
        prices,
        horizon_bars=holding_bars,
        sector_frame=sector,
        mapi_signals=signals,
        seed=config.deterministic_seed,
        signal_frequency=fit_signal_frequency,
        reference_score_column=score_column_used,
        reference_score_threshold=args.score_threshold,
        transaction_cost_bps=config.transaction_cost_bps,
        spread_bps=config.spread_bps,
        slippage_bps=config.slippage_bps,
        large_move_threshold=config.large_move_threshold,
        bootstrap_samples=config.bootstrap_samples,
        bootstrap_seed=config.deterministic_seed,
        fit_mask=fit_mask,
        test_mask=test_mask,
        reference_direction_column=direction_column_used,
        **eligibility,
    )
    random_rows = random_control_distribution(
        prices,
        horizon_bars=holding_bars,
        seeds=tuple(config.deterministic_seed + offset for offset in range(10)),
        signal_frequency=fit_signal_frequency,
        mapi_signals=signals,
        reference_score_column=score_column_used,
        reference_score_threshold=args.score_threshold,
        transaction_cost_bps=config.transaction_cost_bps,
        spread_bps=config.spread_bps,
        slippage_bps=config.slippage_bps,
        large_move_threshold=config.large_move_threshold,
        bootstrap_samples=config.bootstrap_samples,
        bootstrap_seed=config.deterministic_seed,
        fit_mask=fit_mask,
        test_mask=test_mask,
        reference_direction_column=direction_column_used,
        **eligibility,
    )
    ablation_rows = run_ablation(
        args.symbol,
        prices,
        sector,
        benchmark,
        config,
        horizon_name=args.horizon,
        score_column=score_column_used,
        score_threshold=args.score_threshold,
        fit_mask=fit_mask,
        test_mask=test_mask,
        direction_column=direction_column_used,
        **eligibility,
    )
    full_ablation = ablation_rows.loc[
        ablation_rows["variant"] == "full_mapi"
    ]
    if len(full_ablation) != 1 or int(full_ablation.iloc[0]["sample_count"]) != metrics.sample_count:
        raise RuntimeError(
            "Main MAPI and full_mapi ablation sample counts must agree under the shared partition"
        )
    result = {
        "score_column_used": score_column_used,
        "direction_column_used": direction_column_used,
        "implementation_version": IMPLEMENTATION_VERSION,
        "algorithm_revision": ALGORITHM_REVISION,
        "data_contract_version": DATA_CONTRACT_VERSION,
        "config_fingerprint": config.fingerprint(),
        "evaluation_partition": json_safe(partition),
        "eligibility": eligibility,
        "fit_signal_frequency": fit_signal_frequency,
        "test_signal_frequency": test_signal_frequency,
        "metrics": metrics.to_dict(),
        "score_buckets": json_safe(buckets.to_dict(orient="records")),
        "baseline_comparisons": json_safe(baseline_rows.to_dict(orient="records")),
        "random_control_distribution": json_safe(
            random_rows.to_dict(orient="records")
        ),
        "ablation": json_safe(ablation_rows.to_dict(orient="records")),
    }
    payload = json.dumps(result, indent=2, ensure_ascii=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(payload + "\n", encoding="utf-8")
    logger.info(
        "MAPI backtest completed",
        extra={
            "event": "mapi_backtest_completed",
            "symbol": args.symbol,
            "horizon": args.horizon,
            "output": args.output,
            "sample_count": metrics.sample_count,
        },
    )
    print(payload)


if __name__ == "__main__":
    main()
