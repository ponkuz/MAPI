from __future__ import annotations

from dataclasses import dataclass

from mapi.models import clamp


@dataclass(frozen=True)
class WeightInputs:
    """Point-in-time inputs used by the transparent Phase 1 weight policy."""

    regime: str
    freshness: float
    reliability: float
    liquidity: float
    persistence: float
    regime_confidence: float = 1.0


@dataclass(frozen=True)
class WeightDecision:
    weight: float
    factors: dict[str, float]


def calculate_dynamic_weight(
    component_name: str,
    base_weight: float,
    inputs: WeightInputs,
) -> WeightDecision:
    """Apply inspectable multipliers that can later be replaced by a learned policy."""

    regime_multiplier = 1.0
    if inputs.regime in {"panic", "risk_off"} and component_name in {
        "market_regime_divergence",
        "stock_sector_divergence",
        "volatility_anomaly",
    }:
        regime_multiplier = 1.25
    elif inputs.regime == "risk_on_high_volatility" and component_name in {
        "volatility_anomaly",
        "price_volume_divergence",
    }:
        regime_multiplier = 1.15
    elif inputs.regime == "risk_on_low_volatility" and component_name in {
        "momentum_disagreement",
        "price_volume_divergence",
    }:
        regime_multiplier = 1.10

    regime_confidence = clamp(inputs.regime_confidence, 0.0, 1.0)
    confidence_scaled_regime = 1.0 + (regime_multiplier - 1.0) * regime_confidence
    multiplicative_factors = {
        "regime": confidence_scaled_regime,
        "freshness": 0.75 + 0.25 * clamp(inputs.freshness, 0.0, 1.0),
        "reliability": clamp(inputs.reliability, 0.0, 1.0),
        "liquidity": 0.80 + 0.20 * clamp(inputs.liquidity, 0.0, 1.0),
        "persistence": 0.85 + 0.30 * clamp(inputs.persistence, 0.0, 1.0),
    }
    weight = base_weight
    for factor in multiplicative_factors.values():
        weight *= factor
    diagnostics = {
        **multiplicative_factors,
        "regime_confidence": regime_confidence,
    }
    return WeightDecision(
        weight=clamp(weight, 0.0, 1.0),
        factors={name: round(value, 6) for name, value in diagnostics.items()},
    )
