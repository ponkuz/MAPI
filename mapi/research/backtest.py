from __future__ import annotations

import numpy as np
import pandas as pd

from mapi.config import MapiConfig
from mapi.data.validation import normalize_ohlcv
from mapi.models import BacktestMetrics
from mapi.research.labels import forward_path_metrics
from mapi.research.matching import (
    candidate_exclusions,
    candidate_mask,
    evidence_eligibility_mask,
)


def run_event_study(
    signals: pd.DataFrame,
    price_frame: pd.DataFrame,
    horizon_bars: int,
    name: str = "mapi",
    score_threshold: float = 60.0,
    min_direction: float = 0.10,
    signal_delay_bars: int = 1,
    transaction_cost_bps: float = 10.0,
    spread_bps: float = 2.0,
    slippage_bps: float = 5.0,
    allow_overlapping: bool = False,
    score_column: str = "mapi_score",
    large_move_threshold: float = 0.02,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 42,
    evaluation_mask: pd.Series | None = None,
    min_confidence: float = 0.0,
    min_data_quality: float = 0.0,
    require_frequency_compatible: bool = False,
    direction_column: str = "mapi_direction",
    selection_mask: pd.Series | None = None,
) -> BacktestMetrics:
    """Evaluate signal events; this is not a capital-aware portfolio simulation."""

    prices = normalize_ohlcv(price_frame) if "timestamp" in price_frame.columns else price_frame
    aligned = signals.reindex(prices.index)
    if score_column not in aligned.columns:
        raise ValueError(f"Score column is unavailable: {score_column}")
    score = aligned[score_column].fillna(0.0)
    if direction_column not in aligned.columns:
        raise ValueError(f"Direction column is unavailable: {direction_column}")
    direction = aligned[direction_column].fillna(0.0).clip(-1.0, 1.0)
    evaluation = (
        evaluation_mask.reindex(prices.index, fill_value=False).astype(bool)
        if evaluation_mask is not None
        else pd.Series(True, index=prices.index)
    )
    evaluation_index = prices.index[evaluation.to_numpy()]
    evaluation_start = evaluation_index[0] if len(evaluation_index) else None
    evaluation_end = evaluation_index[-1] if len(evaluation_index) else None
    exclusions = candidate_exclusions(
        aligned,
        score_column,
        score_threshold,
        min_direction,
        evaluation,
        min_confidence,
        min_data_quality,
        require_frequency_compatible,
        direction_column,
        selection_mask,
    )
    selected_candidates = (
        selection_mask.reindex(prices.index, fill_value=False)
        & evidence_eligibility_mask(
            aligned,
            evaluation,
            min_confidence,
            min_data_quality,
            require_frequency_compatible,
        )
        & (direction.abs() >= min_direction)
        if selection_mask is not None
        else candidate_mask(
            aligned,
            score_column,
            score_threshold,
            min_direction,
            evaluation,
            min_confidence,
            min_data_quality,
            require_frequency_compatible,
            direction_column,
        )
    )
    candidate_side = pd.Series(
        np.where(selected_candidates, np.sign(direction), 0.0),
        index=prices.index,
        dtype=float,
    )
    labels = forward_path_metrics(prices, horizon_bars, signal_delay_bars)
    valid_candidate = (candidate_side != 0.0) & labels["forward_return"].notna()
    selected = valid_candidate.copy()
    excluded_overlap = 0
    if not allow_overlapping:
        selected[:] = False
        next_allowed_position = 0
        for position in np.flatnonzero(valid_candidate.to_numpy()):
            if position < next_allowed_position:
                excluded_overlap += 1
                continue
            selected.iloc[position] = True
            next_allowed_position = position + horizon_bars + 1
    side = candidate_side.where(selected, 0.0)
    gross_return = side * labels["forward_return"]
    round_trip_cost = (transaction_cost_bps + spread_bps + slippage_bps) / 10000.0
    net_return = gross_return - np.where(side != 0.0, round_trip_cost, 0.0)
    trades = net_return[(side != 0.0) & net_return.notna()]
    warnings = [
        "Event-study statistics are not realizable portfolio metrics; capital, sizing, borrow, and capacity are not modeled."
    ]
    if allow_overlapping and int(valid_candidate.sum()) > 1:
        warnings.append(
            "Forward-return observations may overlap; Sharpe, drawdown, and profit factor are descriptive event-sequence statistics."
        )
    if trades.empty:
        return BacktestMetrics(
            name=name,
            horizon_bars=horizon_bars,
            sample_count=0,
            mean_return=0.0,
            median_return=0.0,
            gross_directional_accuracy=0.0,
            net_profitable_event_rate=0.0,
            large_move_capture_rate=0.0,
            event_return_mean_to_std=0.0,
            event_return_mean_to_downside_std=0.0,
            mean_return_ci_lower=0.0,
            mean_return_ci_upper=0.0,
            bootstrap_samples=bootstrap_samples,
            maximum_drawdown=0.0,
            profit_factor=0.0,
            active_event_ic=0.0,
            mean_intrabar_mfe=0.0,
            mean_intrabar_mae=0.0,
            mean_absolute_return=0.0,
            mean_future_volatility=0.0,
            intrabar_breakout_rate=0.0,
            reversal_rate=0.0,
            mean_time_to_move=0.0,
            non_overlapping=not allow_overlapping,
            overlapping_candidates_excluded=excluded_overlap,
            score_column_used=score_column,
            direction_column_used=direction_column,
            excluded_low_confidence_count=exclusions.low_confidence,
            excluded_low_quality_count=exclusions.low_quality,
            excluded_frequency_mismatch_count=exclusions.frequency_mismatch,
            evaluation_count=int(evaluation.sum()),
            evaluation_start=evaluation_start,
            evaluation_end=evaluation_end,
            selected_event_timestamps=[],
            warnings=warnings,
        )

    sample_count = int(trades.count())
    mean_return = float(trades.mean())
    median_return = float(trades.median())
    net_profitable_event_rate = float((trades > 0).mean())
    std = float(trades.std(ddof=0))
    mean_to_std = float(mean_return / std) if std > 0 else 0.0
    downside = trades[trades < 0]
    downside_std = float(downside.std(ddof=0)) if not downside.empty else 0.0
    mean_to_downside = float(mean_return / downside_std) if downside_std > 0 else 0.0
    ci_lower, ci_upper = _bootstrap_mean_ci(
        trades, samples=bootstrap_samples, seed=bootstrap_seed
    )
    equity = (1.0 + trades.fillna(0.0)).cumprod()
    drawdown = equity / equity.cummax() - 1.0
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0
    gains = float(trades[trades > 0].sum())
    losses = float(trades[trades < 0].sum())
    profit_factor = gains / abs(losses) if losses < 0 else (np.inf if gains > 0 else 0.0)
    realized = labels["forward_return"].reindex(trades.index)
    trade_side = side.reindex(trades.index)
    correct_direction = np.sign(realized) == np.sign(trade_side)
    gross_directional_accuracy = (
        float(correct_direction.mean()) if len(correct_direction) else 0.0
    )
    opportunities = (
        evaluation & (labels["forward_return"].abs() >= large_move_threshold)
    )
    correct_opportunities = (
        opportunities
        & (side != 0.0)
        & (np.sign(labels["forward_return"]) == np.sign(side))
    )
    large_move_capture_rate = float(
        correct_opportunities.sum() / max(1, opportunities.sum())
    )
    prediction = (score * direction).where(selected).replace([np.inf, -np.inf], np.nan)
    ic_frame = pd.concat([prediction, labels["forward_return"]], axis=1).dropna()
    has_ic_variation = (
        len(ic_frame) >= 3
        and float(ic_frame.iloc[:, 0].std(ddof=0)) > 0.0
        and float(ic_frame.iloc[:, 1].std(ddof=0)) > 0.0
    )
    active_event_ic = (
        float(ic_frame.iloc[:, 0].corr(ic_frame.iloc[:, 1]))
        if has_ic_variation
        else 0.0
    )
    trade_labels = labels.reindex(trades.index)
    directional_mfe = pd.Series(
        np.where(
            trade_side > 0,
            trade_labels["intrabar_mfe"],
            -trade_labels["intrabar_mae"],
        ),
        index=trades.index,
    )
    directional_mae = pd.Series(
        np.where(
            trade_side > 0,
            trade_labels["intrabar_mae"],
            -trade_labels["intrabar_mfe"],
        ),
        index=trades.index,
    )
    return BacktestMetrics(
        name=name,
        horizon_bars=horizon_bars,
        sample_count=sample_count,
        mean_return=mean_return,
        median_return=median_return,
        gross_directional_accuracy=gross_directional_accuracy,
        net_profitable_event_rate=net_profitable_event_rate,
        large_move_capture_rate=large_move_capture_rate,
        event_return_mean_to_std=mean_to_std,
        event_return_mean_to_downside_std=mean_to_downside,
        mean_return_ci_lower=ci_lower,
        mean_return_ci_upper=ci_upper,
        bootstrap_samples=bootstrap_samples,
        maximum_drawdown=max_drawdown,
        profit_factor=float(profit_factor),
        active_event_ic=active_event_ic,
        mean_intrabar_mfe=float(directional_mfe.mean()) if not directional_mfe.empty else 0.0,
        mean_intrabar_mae=float(directional_mae.mean()) if not directional_mae.empty else 0.0,
        mean_absolute_return=float(trade_labels["absolute_return"].mean()),
        mean_future_volatility=float(trade_labels["future_volatility"].mean()),
        intrabar_breakout_rate=float(trade_labels["intrabar_breakout"].mean()),
        reversal_rate=float(trade_labels["reversal"].mean()),
        mean_time_to_move=float(trade_labels["time_to_move"].mean()),
        non_overlapping=not allow_overlapping,
        overlapping_candidates_excluded=excluded_overlap,
        score_column_used=score_column,
        direction_column_used=direction_column,
        excluded_low_confidence_count=exclusions.low_confidence,
        excluded_low_quality_count=exclusions.low_quality,
        excluded_frequency_mismatch_count=exclusions.frequency_mismatch,
        evaluation_count=int(evaluation.sum()),
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
        selected_event_timestamps=list(trades.index),
        warnings=warnings,
    )


def run_backtest(
    signals: pd.DataFrame,
    price_frame: pd.DataFrame,
    horizon_bars: int,
    name: str = "mapi",
    score_threshold: float = 60.0,
    min_direction: float = 0.10,
    signal_delay_bars: int = 1,
    transaction_cost_bps: float = 10.0,
    spread_bps: float = 2.0,
    slippage_bps: float = 5.0,
    allow_overlapping: bool = False,
    score_column: str = "mapi_score",
    large_move_threshold: float = 0.02,
    bootstrap_samples: int = 1000,
    bootstrap_seed: int = 42,
    evaluation_mask: pd.Series | None = None,
    min_confidence: float = 0.0,
    min_data_quality: float = 0.0,
    require_frequency_compatible: bool = False,
    direction_column: str = "mapi_direction",
    selection_mask: pd.Series | None = None,
) -> BacktestMetrics:
    """Compatibility wrapper for the event-study engine."""

    return run_event_study(
        signals=signals,
        price_frame=price_frame,
        horizon_bars=horizon_bars,
        name=name,
        score_threshold=score_threshold,
        min_direction=min_direction,
        signal_delay_bars=signal_delay_bars,
        transaction_cost_bps=transaction_cost_bps,
        spread_bps=spread_bps,
        slippage_bps=slippage_bps,
        allow_overlapping=allow_overlapping,
        score_column=score_column,
        large_move_threshold=large_move_threshold,
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
        evaluation_mask=evaluation_mask,
        min_confidence=min_confidence,
        min_data_quality=min_data_quality,
        require_frequency_compatible=require_frequency_compatible,
        direction_column=direction_column,
        selection_mask=selection_mask,
    )


def run_configured_event_study(
    signals: pd.DataFrame,
    price_frame: pd.DataFrame,
    config: MapiConfig,
    horizon_name: str,
    name: str = "mapi",
    score_threshold: float = 60.0,
    min_direction: float = 0.10,
    **overrides: object,
) -> BacktestMetrics:
    """Apply CLI-equivalent event eligibility and configured study settings.

    The caller must pass ``evaluation_mask`` when holdout-only evaluation is
    required; this helper does not create the CLI chronological partition.
    """

    config.validate()
    if horizon_name not in config.horizons:
        raise ValueError(f"Unknown horizon: {horizon_name}")
    options: dict[str, object] = {
        "horizon_bars": config.horizons[horizon_name].return_window,
        "name": name,
        "score_threshold": score_threshold,
        "min_direction": min_direction,
        "transaction_cost_bps": config.transaction_cost_bps,
        "spread_bps": config.spread_bps,
        "slippage_bps": config.slippage_bps,
        "score_column": config.research_score_column,
        "large_move_threshold": config.large_move_threshold,
        "bootstrap_samples": config.bootstrap_samples,
        "bootstrap_seed": config.deterministic_seed,
        "min_confidence": config.research_min_confidence,
        "min_data_quality": config.research_min_data_quality,
        "require_frequency_compatible": (
            config.research_require_frequency_compatible
        ),
        "direction_column": "mapi_forecast_direction",
    }
    options.update(overrides)
    return run_event_study(signals, price_frame, **options)  # type: ignore[arg-type]


def compare_score_buckets(
    signals: pd.DataFrame,
    price_frame: pd.DataFrame,
    horizon_bars: int,
    buckets: tuple[int, ...] = (0, 20, 40, 60, 80, 101),
    score_column: str = "mapi_score",
    **kwargs: object,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    score = signals[score_column].fillna(0.0)
    for left, right in zip(buckets[:-1], buckets[1:]):
        bucket_signals = signals.copy()
        in_bucket = (score >= left) & (score < right)
        bucket_signals.loc[~in_bucket, score_column] = 0.0
        for direction_column in ("mapi_direction", "mapi_forecast_direction"):
            if direction_column in bucket_signals:
                bucket_signals.loc[~in_bucket, direction_column] = 0.0
        metrics = run_backtest(
            bucket_signals,
            price_frame,
            horizon_bars=horizon_bars,
            name=f"score_{left}_{right - 1}",
            score_threshold=max(float(left), 1e-9),
            score_column=score_column,
            **kwargs,
        )
        row = metrics.to_dict()
        row["bucket"] = f"{left}-{right - 1}"
        rows.append(row)
    return pd.DataFrame(rows)


def _bootstrap_mean_ci(
    returns: pd.Series, samples: int, seed: int
) -> tuple[float, float]:
    values = returns.dropna().to_numpy(dtype=float)
    if len(values) == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(samples, len(values)))
    means = values[indices].mean(axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975])
    return float(lower), float(upper)
