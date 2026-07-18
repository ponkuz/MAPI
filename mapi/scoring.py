from __future__ import annotations

from typing import Iterable

import pandas as pd

from mapi.components import (
    FundamentalExpectationsDivergence,
    MarketRegimeDivergence,
    MomentumDisagreement,
    NewsReactionDivergence,
    OptionsAnomaly,
    PriceVolumeDivergence,
    SentimentCrowdingAnomaly,
    StockSectorDivergence,
    VolatilityAnomaly,
)
from mapi.components.base import AnomalyComponent, ComponentContext, finalize_component_frame
from mapi.config import MapiConfig
from mapi.data.frequency import FrequencyValidation, validate_horizon_frequency
from mapi.data.validation import normalize_ohlcv, ohlcv_quality_score
from mapi.explainability import build_summary, dominant_anomalies, machine_reasons
from mapi.models import ComponentSignal, HorizonConfig, MapiSignal, clamp
from mapi.normalization import prior_percentile_rank, rolling_percentile_rank
from mapi.redundancy import compute_redundancy_penalties
from mapi.realization import directional_realization_score
from mapi.regimes import detect_market_regime
from mapi.weights import WeightInputs, calculate_dynamic_weight


def default_components() -> list[AnomalyComponent]:
    return [
        PriceVolumeDivergence(),
        StockSectorDivergence(),
        MomentumDisagreement(),
        VolatilityAnomaly(),
        MarketRegimeDivergence(),
        SentimentCrowdingAnomaly(),
        OptionsAnomaly(),
        FundamentalExpectationsDivergence(),
        NewsReactionDivergence(),
    ]


def calculate_mapi(
    symbol: str,
    price_frame: pd.DataFrame,
    sector_frame: pd.DataFrame | None = None,
    benchmark_frame: pd.DataFrame | None = None,
    config: MapiConfig | None = None,
    components: Iterable[AnomalyComponent] | None = None,
) -> dict[str, pd.DataFrame]:
    config = config or MapiConfig()
    prices = normalize_ohlcv(price_frame) if "timestamp" in price_frame.columns else price_frame.copy()
    sector = (
        normalize_ohlcv(sector_frame)
        if sector_frame is not None and "timestamp" in sector_frame.columns
        else sector_frame
    )
    benchmark = (
        normalize_ohlcv(benchmark_frame)
        if benchmark_frame is not None and "timestamp" in benchmark_frame.columns
        else benchmark_frame
    )
    selected = {
        component.name: component for component in (components or default_components())
    }
    config.validate(set(selected))
    outputs: dict[str, pd.DataFrame] = {}
    for horizon_name, horizon in config.horizons.items():
        frequency = validate_horizon_frequency(prices.index, horizon)
        if not frequency.compatible and config.frequency_mismatch_policy == "error":
            raise ValueError(frequency.warning or "Input frequency is incompatible")
        regime_info = detect_market_regime(prices, benchmark, horizon, config)
        context = ComponentContext(
            symbol=symbol,
            sector_frame=sector,
            benchmark_frame=benchmark,
            regime=regime_info["regime"],
        )
        component_frames: dict[str, pd.DataFrame] = {}
        for component_name in config.enabled_components:
            component = selected.get(component_name)
            if component is None:
                continue
            calculated = component.calculate(prices, context, horizon, config)
            component_frames[component_name] = finalize_component_frame(
                calculated,
                prices.index,
                f"{component_name} unavailable",
            )
        outputs[horizon_name] = _score_horizon(
            symbol=symbol,
            prices=prices,
            horizon=horizon,
            regime_info=regime_info,
            frequency=frequency,
            component_frames=component_frames,
            components=selected,
            config=config,
        )
    return outputs


def calculate_latest_mapi(
    symbol: str,
    price_frame: pd.DataFrame,
    sector_frame: pd.DataFrame | None = None,
    benchmark_frame: pd.DataFrame | None = None,
    config: MapiConfig | None = None,
) -> dict[str, dict[str, object]]:
    frames = calculate_mapi(
        symbol=symbol,
        price_frame=price_frame,
        sector_frame=sector_frame,
        benchmark_frame=benchmark_frame,
        config=config,
    )
    latest: dict[str, dict[str, object]] = {}
    for horizon, frame in frames.items():
        key = f"mapi_{horizon}"
        latest[key] = frame.iloc[-1]["signal"].to_dict()
    return latest


def _score_horizon(
    symbol: str,
    prices: pd.DataFrame,
    horizon: HorizonConfig,
    regime_info: pd.DataFrame,
    frequency: FrequencyValidation,
    component_frames: dict[str, pd.DataFrame],
    components: dict[str, AnomalyComponent],
    config: MapiConfig,
) -> pd.DataFrame:
    penalties = compute_redundancy_penalties(
        component_frames,
        window=max(8, min(config.redundancy_window, horizon.rolling_window)),
        threshold=config.redundancy_threshold,
        minimum_penalty=config.min_redundancy_penalty,
        min_periods=config.redundancy_min_periods,
    )
    ohlcv_quality = ohlcv_quality_score(
        prices,
        horizon.rolling_window,
        penalize_split_like=config.adjusted_prices_required,
    )
    freshness = _freshness_score(prices.index, horizon.rolling_window)
    liquidity = rolling_percentile_rank(
        (prices["close"] * prices["volume"]).where(
            (prices["close"] > 0.0) & (prices["volume"] >= 0.0)
        ),
        horizon.rolling_window,
        horizon.min_periods,
    ).fillna(0.5)
    trailing_return = prices["close"].pct_change(horizon.return_window)
    recent_move_extremeness = prior_percentile_rank(
        trailing_return.abs(), horizon.rolling_window, horizon.min_periods
    ).fillna(0.0)
    realization_scale = trailing_return.abs().rolling(
        horizon.rolling_window, min_periods=horizon.min_periods
    ).quantile(0.90).shift(1)

    records: list[dict[str, object]] = []
    anomaly_age = 0
    first_detected_at: object | None = None
    first_detected_price: float | None = None
    previous_score = 0.0
    previous_state = "normal"
    component_base_sum = sum(
        config.component_weights.get(name, 0.0) for name in component_frames
    ) or 1.0

    for idx, timestamp in enumerate(prices.index):
        current_regime = str(regime_info["regime"].iloc[idx])
        current_regime_source = str(regime_info["regime_source"].iloc[idx])
        current_regime_confidence = clamp(
            float(regime_info["regime_confidence"].iloc[idx]), 0.0, 1.0
        )
        component_signals: list[ComponentSignal] = []
        effective_sum = 0.0
        capacity_sum = 0.0
        direction_numerator = 0.0
        direction_denominator = 0.0
        confidence_numerator = 0.0
        confidence_denominator = 0.0
        evidence_weight_sum = 0.0

        for component_name, frame in component_frames.items():
            component = components[component_name]
            row = frame.iloc[idx]
            confidence = clamp(float(row["confidence"]), 0.0, 1.0)
            base_weight = config.component_weights.get(component_name, 0.0)
            persistence = float(
                frame["anomaly_strength"].iloc[max(0, idx - 2) : idx + 1].mean()
            )
            weight_decision = calculate_dynamic_weight(
                component_name,
                base_weight,
                WeightInputs(
                    regime=current_regime,
                    freshness=float(freshness.iloc[idx]),
                    reliability=config.component_reliability.get(component_name, 1.0),
                    liquidity=float(liquidity.iloc[idx]),
                    persistence=persistence,
                ),
            )
            penalty = clamp(float(penalties[component_name].iloc[idx]), 0.0, 1.0)
            signal = ComponentSignal(
                name=component_name,
                family=component.family,
                anomaly_strength=clamp(float(row["anomaly_strength"]), 0.0, 1.0),
                direction=clamp(float(row["direction"]), -1.0, 1.0),
                confidence=confidence,
                weight=weight_decision.weight,
                novelty=clamp(float(row["novelty"]), 0.0, 1.0),
                historical_extremeness=clamp(
                    float(row.get("historical_extremeness", row["novelty"])), 0.0, 1.0
                ),
                recurrence_rate=clamp(float(row.get("recurrence_rate", 0.0)), 0.0, 1.0),
                redundancy_penalty=penalty,
                reason=str(row["reason"]),
                metrics=row["metrics"] if isinstance(row["metrics"], dict) else {},
                weight_factors=weight_decision.factors,
            )
            component_signals.append(signal)
            effective = signal.effective_score
            effective_sum += effective
            capacity = signal.weight * confidence * penalty
            capacity_sum += capacity
            direction_numerator += signal.direction * effective
            direction_denominator += abs(effective)
            confidence_numerator += confidence * signal.weight * penalty
            confidence_denominator += signal.weight * penalty
            evidence_weight_sum += base_weight * confidence * penalty

        raw_score = 100.0 * effective_sum / capacity_sum if capacity_sum > 0 else 0.0
        direction = (
            direction_numerator / direction_denominator if direction_denominator > 0 else 0.0
        )
        conditional_confidence = (
            confidence_numerator / confidence_denominator
            if confidence_denominator > 0
            else 0.0
        )
        evidence_coverage = clamp(evidence_weight_sum / component_base_sum, 0.0, 1.0)
        current_ohlcv_quality = float(ohlcv_quality.iloc[idx])
        frequency_factor = 1.0 if frequency.compatible else 0.0
        data_quality = clamp(
            current_ohlcv_quality * evidence_coverage * frequency_factor,
            0.0,
            1.0,
        )
        confidence = clamp(
            conditional_confidence
            * current_ohlcv_quality
            * evidence_coverage
            * frequency_factor,
            0.0,
            1.0,
        )
        confirmation_count = sum(
            1
            for component in component_signals
            if component.anomaly_strength >= 0.5 and component.confidence >= 0.3
        )
        prior_active = previous_state in {"emerging", "confirmed", "fading"}
        if raw_score >= config.high_anomaly_threshold:
            if prior_active and first_detected_at is not None:
                anomaly_age += 1
            else:
                anomaly_age = 1
                first_detected_at = timestamp
                current_close = float(prices["close"].iloc[idx])
                first_detected_price = current_close if current_close > 0.0 else None
            state = (
                "confirmed"
                if anomaly_age >= 3
                or (
                    raw_score >= config.strong_anomaly_threshold
                    and confirmation_count >= 2
                )
                else "emerging"
            )
        elif prior_active and raw_score >= 20.0:
            anomaly_age += 1
            state = "fading"
        elif prior_active:
            anomaly_age += 1
            state = "invalidated"
        else:
            anomaly_age = 0
            first_detected_at = None
            first_detected_price = None
            state = "normal"
        score_change = raw_score - previous_score
        anomaly_trend = (
            "increasing" if score_change > 2.0 else "decreasing" if score_change < -2.0 else "stable"
        )
        recent_extremeness = clamp(
            float(recent_move_extremeness.iloc[idx]), 0.0, 1.0
        )
        directional_move_since_detection = 0.0
        if first_detected_price is not None and first_detected_price > 0.0:
            current_close = float(prices["close"].iloc[idx])
            if current_close > 0.0:
                directional_move_since_detection = (
                    current_close / first_detected_price - 1.0
                )
        scale = float(realization_scale.iloc[idx])
        directional_realization = directional_realization_score(
            direction,
            directional_move_since_detection,
            scale if pd.notna(scale) else 0.0,
        )
        actionability_score = clamp(
            raw_score
            * (
                1.0
                - config.directional_realization_penalty
                * directional_realization
            ),
            0.0,
            100.0,
        )
        public_score = (
            actionability_score
            if config.score_semantics == "legacy_actionability"
            else clamp(raw_score, 0.0, 100.0)
        )

        reasons = dominant_anomalies(component_signals)
        summary = build_summary(
            score=raw_score,
            direction=direction,
            regime=current_regime,
            state=state,
            confirmation_count=confirmation_count,
            reasons=reasons,
        )
        signal = MapiSignal(
            symbol=symbol,
            timestamp=timestamp,
            horizon=horizon.name,
            mapi_score=public_score,
            mapi_raw_score=raw_score,
            mapi_actionability_score=actionability_score,
            mapi_direction=direction,
            mapi_confidence=confidence,
            mapi_regime=current_regime,
            regime_source=current_regime_source,
            regime_confidence=current_regime_confidence,
            anomaly_components=component_signals,
            dominant_anomalies=reasons,
            data_quality_score=data_quality,
            ohlcv_quality_score=current_ohlcv_quality,
            evidence_coverage_score=evidence_coverage,
            signal_version=config.signal_version,
            anomaly_state=state,
            anomaly_first_detected_at=first_detected_at,
            anomaly_age_bars=anomaly_age,
            anomaly_trend=anomaly_trend,
            confirmation_count=confirmation_count,
            recent_move_extremeness=recent_extremeness,
            directional_move_since_detection=directional_move_since_detection,
            directional_realization_score=directional_realization,
            already_realized_score=recent_extremeness,
            input_interval_seconds=frequency.median_interval_seconds,
            horizon_frequency_compatible=frequency.compatible,
            horizon_warning=frequency.warning,
            machine_reasons=machine_reasons(component_signals),
            human_summary=summary,
        )
        records.append(
            {
                "timestamp": timestamp,
                "symbol": symbol,
                "horizon": horizon.name,
                "mapi_score": public_score,
                "mapi_raw_score": raw_score,
                "mapi_actionability_score": actionability_score,
                "mapi_direction": direction,
                "mapi_confidence": confidence,
                "mapi_regime": current_regime,
                "regime_source": current_regime_source,
                "regime_confidence": current_regime_confidence,
                "dominant_anomalies": reasons,
                "data_quality_score": data_quality,
                "ohlcv_quality_score": current_ohlcv_quality,
                "evidence_coverage_score": evidence_coverage,
                "signal_version": config.signal_version,
                "anomaly_state": state,
                "anomaly_first_detected_at": first_detected_at,
                "anomaly_age_bars": anomaly_age,
                "anomaly_trend": anomaly_trend,
                "confirmation_count": confirmation_count,
                "recent_move_extremeness": recent_extremeness,
                "directional_move_since_detection": directional_move_since_detection,
                "directional_realization_score": directional_realization,
                "already_realized_score": recent_extremeness,
                "input_interval_seconds": frequency.median_interval_seconds,
                "horizon_frequency_compatible": frequency.compatible,
                "horizon_warning": frequency.warning,
                "human_summary": summary,
                "signal": signal,
            }
        )
        previous_score = raw_score
        previous_state = state
        if state == "invalidated":
            anomaly_age = 0
            first_detected_at = None
            first_detected_price = None
    return pd.DataFrame(records, index=prices.index)


def _freshness_score(index: pd.Index, rolling_window: int) -> pd.Series:
    timestamps = pd.Series(pd.to_datetime(index, utc=True), index=index)
    gaps = timestamps.diff().dt.total_seconds()
    expected = gaps.rolling(rolling_window, min_periods=3).median().shift(1)
    ratio = gaps / expected.replace(0.0, pd.NA)
    score = (1.0 / ratio.clip(lower=1.0).div(3.0).clip(lower=1.0)).clip(0.0, 1.0)
    return score.fillna(1.0).astype(float)
