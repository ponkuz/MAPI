from __future__ import annotations

from mapi.models import ComponentSignal


def direction_label(direction: float) -> str:
    if direction > 0.25:
        return "bullish"
    if direction < -0.25:
        return "bearish"
    return "direction-neutral"


def intensity_label(score: float) -> str:
    if score >= 80:
        return "extreme"
    if score >= 60:
        return "strong"
    if score >= 40:
        return "meaningful"
    if score >= 20:
        return "mild"
    return "low"


def dominant_anomalies(components: list[ComponentSignal], limit: int = 3) -> list[str]:
    ranked = sorted(components, key=lambda item: item.effective_score, reverse=True)
    reasons: list[str] = []
    for component in ranked:
        if component.effective_score <= 0.0 or component.confidence <= 0.0:
            continue
        if component.reason:
            reasons.append(component.reason)
        if len(reasons) >= limit:
            break
    return reasons


def machine_reasons(components: list[ComponentSignal]) -> list[dict[str, object]]:
    return [
        {
            "component": component.name,
            "family": component.family,
            "strength": round(component.anomaly_strength, 4),
            "direction": round(component.direction, 4),
            "confidence": round(component.confidence, 4),
            "effective_score": round(component.effective_score, 4),
            "reason": component.reason,
            "metrics": component.metrics,
            "weight_factors": component.weight_factors,
        }
        for component in components
        if component.confidence > 0.0
    ]


def build_summary(
    score: float,
    direction: float,
    regime: str,
    state: str,
    confirmation_count: int,
    reasons: list[str],
) -> str:
    label = intensity_label(score)
    directional = direction_label(direction)
    if not reasons:
        return (
            f"MAPI detected {label} anomaly pressure with a {directional} lean in "
            f"the {regime} regime. The signal is {state} with "
            f"{confirmation_count} confirming components."
        )
    joined = " ".join(reasons[:3])
    return (
        f"MAPI detected {label} {directional} anomaly pressure in the {regime} "
        f"regime. {joined} The signal is {state} with {confirmation_count} "
        f"confirming components."
    )
