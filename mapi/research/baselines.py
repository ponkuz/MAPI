from __future__ import annotations

import numpy as np
import pandas as pd

from mapi.data.alignment import align_point_in_time
from mapi.data.validation import normalize_ohlcv
from mapi.normalization import historical_zscore, rolling_percentile_rank
from mapi.research.backtest import run_event_study
from mapi.research.matching import (
    apply_frequency_match,
    candidate_mask,
    candidate_frequency,
    chronological_masks,
    fit_frequency_matched_threshold,
)


def generate_baselines(
    price_frame: pd.DataFrame,
    seed: int = 42,
    signal_frequency: float = 0.10,
    sector_frame: pd.DataFrame | None = None,
    mapi_signals: pd.DataFrame | None = None,
    score_column: str = "mapi_score",
    reference_score_threshold: float = 60.0,
    min_direction: float = 0.10,
    fit_mask: pd.Series | None = None,
    test_mask: pd.Series | None = None,
    min_confidence: float = 0.0,
    min_data_quality: float = 0.0,
    require_frequency_compatible: bool = False,
    reference_direction_column: str = "mapi_direction",
) -> dict[str, pd.DataFrame]:
    prices = normalize_ohlcv(price_frame) if "timestamp" in price_frame.columns else price_frame
    close = prices["close"]
    returns = close.pct_change()
    baselines: dict[str, pd.DataFrame] = {}
    baselines["random_same_frequency"] = _random_signal(prices.index, seed, signal_frequency)
    baselines["previous_day_return"] = _frame(
        prices.index,
        score=rolling_percentile_rank(
            returns.abs().shift(1), 60, 20
        ).fillna(0.0) * 100.0,
        direction=np.sign(returns.shift(1).fillna(0.0)),
    )
    baselines["pure_volume_zscore"] = _frame(
        prices.index,
        score=(historical_zscore(np.log1p(prices["volume"].astype(float)), 60, 20).abs() / 3.0)
        .clip(0.0, 1.0)
        .fillna(0.0)
        * 100.0,
        direction=np.sign(returns.fillna(0.0)),
    )
    baselines["moving_average_crossover"] = _moving_average_crossover(prices)
    baselines["rsi_reversal"] = _rsi_reversal(prices)
    baselines["macd_direction"] = _macd_direction(prices)
    if sector_frame is not None:
        baselines["sector_relative_strength"] = _sector_relative_strength(
            prices, sector_frame
        )
    if mapi_signals is not None:
        aligned_mapi = mapi_signals.reindex(prices.index)
        all_rows = pd.Series(True, index=prices.index)
        reference_events = candidate_mask(
            aligned_mapi,
            score_column,
            reference_score_threshold,
            min_direction,
            all_rows,
            min_confidence,
            min_data_quality,
            require_frequency_compatible,
            reference_direction_column,
        )
        baselines["same_event_times_always_long"] = _frame(
            prices.index,
            score=reference_events.astype(float) * 100.0,
            direction=reference_events.astype(float),
        )
        baselines["equal_weight_components"] = _equal_weight_components(mapi_signals)
        baselines["shuffled_mapi_scores"] = _shuffled_mapi(
            mapi_signals,
            seed,
            score_column,
            fit_mask,
            test_mask,
            reference_direction_column,
        )
        baselines["isolated_stock_sector_residual"] = _isolated_component(
            mapi_signals, "stock_sector_divergence"
        )
        baselines["isolated_volatility_anomaly"] = _isolated_component(
            mapi_signals, "volatility_anomaly"
        )
    else:
        schedule = _deterministic_schedule(prices.index, signal_frequency)
        baselines["same_event_times_always_long"] = _frame(
            prices.index,
            score=schedule.astype(float) * 100.0,
            direction=schedule.astype(float),
        )
    return baselines


def compare_baselines(
    price_frame: pd.DataFrame,
    horizon_bars: int,
    sector_frame: pd.DataFrame | None = None,
    mapi_signals: pd.DataFrame | None = None,
    seed: int = 42,
    signal_frequency: float = 0.10,
    reference_score_column: str = "mapi_score",
    reference_score_threshold: float = 60.0,
    min_direction: float = 0.10,
    fit_fraction: float = 0.70,
    fit_mask: pd.Series | None = None,
    test_mask: pd.Series | None = None,
    min_confidence: float = 0.0,
    min_data_quality: float = 0.0,
    require_frequency_compatible: bool = False,
    reference_direction_column: str = "mapi_direction",
    **event_study_kwargs: object,
) -> pd.DataFrame:
    """Evaluate each negative control under identical event-study assumptions."""

    prices = normalize_ohlcv(price_frame) if "timestamp" in price_frame.columns else price_frame
    if fit_mask is None or test_mask is None:
        fit_mask, test_mask = chronological_masks(prices.index, fit_fraction)
    fit_mask = fit_mask.reindex(prices.index, fill_value=False)
    test_mask = test_mask.reindex(prices.index, fill_value=False)
    event_options = _event_options(event_study_kwargs)
    target_frequency = signal_frequency
    if mapi_signals is not None:
        target_frequency = candidate_frequency(
            mapi_signals.reindex(prices.index),
            reference_score_column,
            reference_score_threshold,
            min_direction,
            fit_mask,
            min_confidence,
            min_data_quality,
            require_frequency_compatible,
            reference_direction_column,
        )
    rows: list[dict[str, object]] = []
    for name, signals in generate_baselines(
        price_frame,
        seed=seed,
        signal_frequency=target_frequency,
        sector_frame=sector_frame,
        mapi_signals=mapi_signals,
        score_column=reference_score_column,
        reference_score_threshold=reference_score_threshold,
        min_direction=min_direction,
        fit_mask=fit_mask,
        test_mask=test_mask,
        min_confidence=min_confidence,
        min_data_quality=min_data_quality,
        require_frequency_compatible=require_frequency_compatible,
        reference_direction_column=reference_direction_column,
    ).items():
        match = fit_frequency_matched_threshold(
            signals,
            "baseline_score",
            target_frequency,
            min_direction,
            fit_mask,
            min_confidence,
            min_data_quality,
            require_frequency_compatible,
            "mapi_forecast_direction",
        )
        test_selection = apply_frequency_match(
            signals,
            "baseline_score",
            match,
            min_direction,
            test_mask,
            min_confidence,
            min_data_quality,
            require_frequency_compatible,
            "mapi_forecast_direction",
        )
        metrics = run_event_study(
            signals,
            prices,
            horizon_bars=horizon_bars,
            name=name,
            score_threshold=match.threshold,
            min_direction=min_direction,
            score_column="baseline_score",
            evaluation_mask=test_mask,
            min_confidence=min_confidence,
            min_data_quality=min_data_quality,
            require_frequency_compatible=require_frequency_compatible,
            direction_column="mapi_forecast_direction",
            selection_mask=test_selection,
            **event_options,
        )
        row = metrics.to_dict()
        row.update(
            {
                "fitted_threshold": match.threshold,
                "target_fit_frequency": target_frequency,
                "candidate_fit_frequency": match.fitted_frequency,
                "candidate_test_frequency": candidate_frequency(
                    signals,
                    "baseline_score",
                    match.threshold,
                    min_direction,
                    test_mask,
                    min_confidence,
                    min_data_quality,
                    require_frequency_compatible,
                    "mapi_forecast_direction",
                    match,
                ),
                "selected_event_count": metrics.sample_count,
                "excluded_overlap_count": metrics.overlapping_candidates_excluded,
                "reference_score_column": reference_score_column,
            }
        )
        rows.append(row)
    rows.append(
        _full_period_buy_and_hold(
            prices,
            test_mask,
            event_options,
            reference_score_column,
        )
    )
    return pd.DataFrame(rows)


def random_control_distribution(
    price_frame: pd.DataFrame,
    horizon_bars: int,
    seeds: tuple[int, ...] = tuple(range(10)),
    signal_frequency: float = 0.10,
    fit_fraction: float = 0.70,
    min_direction: float = 0.10,
    mapi_signals: pd.DataFrame | None = None,
    reference_score_column: str = "mapi_score",
    reference_score_threshold: float = 60.0,
    fit_mask: pd.Series | None = None,
    test_mask: pd.Series | None = None,
    min_confidence: float = 0.0,
    min_data_quality: float = 0.0,
    require_frequency_compatible: bool = False,
    reference_direction_column: str = "mapi_direction",
    **event_study_kwargs: object,
) -> pd.DataFrame:
    """Return multiple deterministic random-control outcomes, one per seed."""

    prices = normalize_ohlcv(price_frame) if "timestamp" in price_frame.columns else price_frame
    if fit_mask is None or test_mask is None:
        fit_mask, test_mask = chronological_masks(prices.index, fit_fraction)
    fit_mask = fit_mask.reindex(prices.index, fill_value=False)
    test_mask = test_mask.reindex(prices.index, fill_value=False)
    event_options = _event_options(event_study_kwargs)
    target_frequency = signal_frequency
    if mapi_signals is not None:
        target_frequency = candidate_frequency(
            mapi_signals.reindex(prices.index),
            reference_score_column,
            reference_score_threshold,
            min_direction,
            fit_mask,
            min_confidence,
            min_data_quality,
            require_frequency_compatible,
            reference_direction_column,
        )
    rows: list[dict[str, object]] = []
    for seed in seeds:
        signals = _random_signal(prices.index, seed, target_frequency)
        match = fit_frequency_matched_threshold(
            signals,
            "baseline_score",
            target_frequency,
            min_direction,
            fit_mask,
            min_confidence,
            min_data_quality,
            require_frequency_compatible,
            "mapi_forecast_direction",
        )
        test_selection = apply_frequency_match(
            signals,
            "baseline_score",
            match,
            min_direction,
            test_mask,
            min_confidence,
            min_data_quality,
            require_frequency_compatible,
            "mapi_forecast_direction",
        )
        metrics = run_event_study(
            signals,
            prices,
            horizon_bars=horizon_bars,
            name=f"random_seed_{seed}",
            score_threshold=match.threshold,
            min_direction=min_direction,
            score_column="baseline_score",
            evaluation_mask=test_mask,
            min_confidence=min_confidence,
            min_data_quality=min_data_quality,
            require_frequency_compatible=require_frequency_compatible,
            direction_column="mapi_forecast_direction",
            selection_mask=test_selection,
            **event_options,
        )
        rows.append(
            {
                "seed": seed,
                **metrics.to_dict(),
                "fitted_threshold": match.threshold,
                "target_fit_frequency": target_frequency,
                "candidate_fit_frequency": match.fitted_frequency,
                "candidate_test_frequency": candidate_frequency(
                    signals,
                    "baseline_score",
                    match.threshold,
                    min_direction,
                    test_mask,
                    min_confidence,
                    min_data_quality,
                    require_frequency_compatible,
                    "mapi_forecast_direction",
                    match,
                ),
                "selected_event_count": metrics.sample_count,
                "excluded_overlap_count": metrics.overlapping_candidates_excluded,
                "reference_score_column": reference_score_column,
            }
        )
    return pd.DataFrame(rows)


def _frame(
    index: pd.Index, score: pd.Series, direction: pd.Series | np.ndarray
) -> pd.DataFrame:
    forecast = pd.Series(direction, index=index).fillna(0.0).clip(-1.0, 1.0)
    return pd.DataFrame(
        {
            "baseline_score": pd.Series(score, index=index).fillna(0.0).clip(0.0, 100.0),
            "mapi_direction": forecast,
            "mapi_forecast_direction": forecast,
            "mapi_direction_semantics": "hypothesized_forward_direction",
            "mapi_confidence": 1.0,
            "data_quality_score": 1.0,
            "horizon_frequency_compatible": True,
        },
        index=index,
    )


def _random_signal(index: pd.Index, seed: int, signal_frequency: float) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    active = rng.random(len(index)) < signal_frequency
    direction = rng.choice([-1.0, 1.0], size=len(index))
    score = np.where(active, 100.0, 0.0)
    return _frame(index, pd.Series(score, index=index), pd.Series(direction, index=index))


def _moving_average_crossover(prices: pd.DataFrame) -> pd.DataFrame:
    close = prices["close"]
    short = close.rolling(20, min_periods=10).mean()
    long = close.rolling(50, min_periods=25).mean()
    spread = (short / long - 1.0).replace([np.inf, -np.inf], np.nan)
    score = (historical_zscore(spread.abs(), 80, 25).abs() / 3.0).clip(0.0, 1.0) * 100.0
    return _frame(prices.index, score.fillna(0.0), np.sign(spread.fillna(0.0)))


def _rsi_reversal(prices: pd.DataFrame) -> pd.DataFrame:
    close = prices["close"]
    delta = close.diff()
    gain = delta.clip(lower=0.0).rolling(14, min_periods=14).mean()
    loss = (-delta.clip(upper=0.0)).rolling(14, min_periods=14).mean()
    rs = gain / loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    direction = pd.Series(0.0, index=prices.index)
    direction[rsi < 30.0] = 1.0
    direction[rsi > 70.0] = -1.0
    score = ((rsi - 50.0).abs() * 2.0).clip(0.0, 100.0)
    return _frame(prices.index, score.fillna(0.0), direction)


def _macd_direction(prices: pd.DataFrame) -> pd.DataFrame:
    close = prices["close"]
    fast = close.ewm(span=12, adjust=False, min_periods=12).mean()
    slow = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = fast - slow
    signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    spread = macd - signal
    score = (historical_zscore(spread.abs(), 80, 25).abs() / 3.0).clip(0.0, 1.0) * 100.0
    return _frame(prices.index, score.fillna(0.0), np.sign(spread.fillna(0.0)))


def _sector_relative_strength(
    prices: pd.DataFrame, sector_frame: pd.DataFrame
) -> pd.DataFrame:
    sector = (
        normalize_ohlcv(sector_frame)
        if "timestamp" in sector_frame.columns
        else sector_frame
    )
    alignment = align_point_in_time(prices, sector)
    relative_return = prices["close"].pct_change(5) - alignment.frame["close"].pct_change(
        5, fill_method=None
    )
    relative_z = historical_zscore(relative_return, 60, 20)
    return _frame(
        prices.index,
        (relative_z.abs() / 3.0).clip(0.0, 1.0).fillna(0.0) * 100.0,
        np.sign(relative_z.fillna(0.0)),
    )


def _equal_weight_components(signals: pd.DataFrame) -> pd.DataFrame:
    scores: list[float] = []
    directions: list[float] = []
    confidences: list[float] = []
    for signal in signals["signal"]:
        active = [item for item in signal.anomaly_components if item.confidence > 0.0]
        if not active:
            scores.append(0.0)
            directions.append(0.0)
            confidences.append(0.0)
            continue
        component_scores = [item.anomaly_strength * item.novelty for item in active]
        scores.append(100.0 * float(np.mean(component_scores)))
        directions.append(float(np.mean([item.direction for item in active])))
        confidences.append(float(np.mean([item.confidence for item in active])))
    return pd.DataFrame(
        {
            "baseline_score": scores,
            "mapi_direction": directions,
            "mapi_forecast_direction": directions,
            "mapi_direction_semantics": "hypothesized_forward_direction",
            "mapi_confidence": confidences,
            "data_quality_score": _source_column(
                signals, "data_quality_score", 1.0
            ),
            "horizon_frequency_compatible": _source_column(
                signals, "horizon_frequency_compatible", True
            ),
        },
        index=signals.index,
    )


def _shuffled_mapi(
    signals: pd.DataFrame,
    seed: int,
    score_column: str,
    fit_mask: pd.Series | None,
    test_mask: pd.Series | None,
    direction_column: str,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    source_score = signals[score_column]
    positions = np.arange(len(signals))
    if fit_mask is None or test_mask is None:
        positions = rng.permutation(len(signals))
    else:
        for partition in (fit_mask, test_mask):
            partition_positions = np.flatnonzero(
                partition.reindex(signals.index, fill_value=False).to_numpy()
            )
            positions[partition_positions] = rng.permutation(partition_positions)
    source_direction = signals[direction_column]
    source_confidence = _source_column(signals, "mapi_confidence", 1.0)
    return pd.DataFrame(
        {
            "baseline_score": source_score.to_numpy()[positions],
            "mapi_direction": source_direction.to_numpy()[positions],
            "mapi_forecast_direction": source_direction.to_numpy()[positions],
            "mapi_direction_semantics": "hypothesized_forward_direction",
            "mapi_confidence": source_confidence.to_numpy()[positions],
            "data_quality_score": _source_column(
                signals, "data_quality_score", 1.0
            ),
            "horizon_frequency_compatible": _source_column(
                signals, "horizon_frequency_compatible", True
            ),
        },
        index=signals.index,
    )


def _isolated_component(signals: pd.DataFrame, component_name: str) -> pd.DataFrame:
    scores: list[float] = []
    directions: list[float] = []
    confidences: list[float] = []
    for signal in signals["signal"]:
        component = next(
            (item for item in signal.anomaly_components if item.name == component_name),
            None,
        )
        if component is None or component.confidence <= 0.0:
            scores.append(0.0)
            directions.append(0.0)
            confidences.append(0.0)
            continue
        scores.append(100.0 * component.anomaly_strength * component.novelty)
        directions.append(component.direction)
        confidences.append(component.confidence)
    return pd.DataFrame(
        {
            "baseline_score": scores,
            "mapi_direction": directions,
            "mapi_forecast_direction": directions,
            "mapi_direction_semantics": "hypothesized_forward_direction",
            "mapi_confidence": confidences,
            "data_quality_score": _source_column(
                signals, "data_quality_score", 1.0
            ),
            "horizon_frequency_compatible": _source_column(
                signals, "horizon_frequency_compatible", True
            ),
        },
        index=signals.index,
    )


def _full_period_buy_and_hold(
    prices: pd.DataFrame,
    test_mask: pd.Series,
    event_study_kwargs: dict[str, object],
    reference_score_column: str,
) -> dict[str, object]:
    test_prices = prices.loc[test_mask]
    total_return = 0.0
    if len(test_prices) >= 2 and float(test_prices["close"].iloc[0]) > 0.0:
        total_return = float(
            test_prices["close"].iloc[-1] / test_prices["close"].iloc[0] - 1.0
        )
        costs = sum(
            float(event_study_kwargs.get(name, 0.0))
            for name in ("transaction_cost_bps", "spread_bps", "slippage_bps")
        ) / 10000.0
        total_return -= costs
    return {
        "name": "full_period_buy_and_hold",
        "analysis_type": "full_period_benchmark",
        "sample_count": 1 if len(test_prices) >= 2 else 0,
        "selected_event_count": 1 if len(test_prices) >= 2 else 0,
        "total_return": total_return,
        "mean_return": total_return,
        "fitted_threshold": None,
        "target_fit_frequency": None,
        "candidate_fit_frequency": None,
        "candidate_test_frequency": 1.0 if len(test_prices) >= 2 else 0.0,
        "excluded_overlap_count": 0,
        "reference_score_column": reference_score_column,
        "warnings": [
            "Full-period buy-and-hold is a capital-path benchmark, not an event study."
        ],
    }


def _deterministic_schedule(index: pd.Index, frequency: float) -> pd.Series:
    count = min(len(index), int(round(frequency * len(index))))
    if frequency > 0.0 and count == 0 and len(index) > 0:
        count = 1
    selected = pd.Series(False, index=index)
    if count > 0:
        positions = np.linspace(0, len(index) - 1, count, dtype=int)
        selected.iloc[np.unique(positions)] = True
    return selected


def _event_options(options: dict[str, object]) -> dict[str, object]:
    reserved = {
        "horizon_bars",
        "name",
        "score_threshold",
        "min_direction",
        "score_column",
        "evaluation_mask",
        "min_confidence",
        "min_data_quality",
        "require_frequency_compatible",
        "direction_column",
        "selection_mask",
    }
    return {name: value for name, value in options.items() if name not in reserved}


def _source_column(
    signals: pd.DataFrame, name: str, default: object
) -> pd.Series:
    return signals[name] if name in signals else pd.Series(default, index=signals.index)
