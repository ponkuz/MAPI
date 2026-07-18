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


def _dominant_anomalies(
    components: list[ComponentSignal], contribution: str, limit: int
) -> list[str]:
    ranked = sorted(
        components, key=lambda item: float(getattr(item, contribution)), reverse=True
    )
    reasons: list[str] = []
    for component in ranked:
        if float(getattr(component, contribution)) <= 0.0 or component.confidence <= 0.0:
            continue
        if component.reason:
            reasons.append(component.reason)
        if len(reasons) >= limit:
            break
    return reasons


def dominant_intensity_anomalies(
    components: list[ComponentSignal], limit: int = 3
) -> list[str]:
    return _dominant_anomalies(components, "intensity_effective_score", limit)


def dominant_alert_anomalies(
    components: list[ComponentSignal], limit: int = 3
) -> list[str]:
    return _dominant_anomalies(components, "alert_effective_score", limit)


def dominant_anomalies(components: list[ComponentSignal], limit: int = 3) -> list[str]:
    """Compatibility alias for intensity-ranked explanations in v0.3.1."""

    return dominant_intensity_anomalies(components, limit)


def machine_reasons(components: list[ComponentSignal]) -> list[dict[str, object]]:
    return [
        {
            "component": component.name,
            "family": component.family,
            "strength": round(component.anomaly_strength, 4),
            "direction": round(component.direction, 4),
            "observed_pressure": round(component.observed_pressure, 4),
            "direction_semantics": component.direction_semantics,
            "direction_contract_warning": component.direction_contract_warning,
            "confidence": round(component.confidence, 4),
            "historical_extremeness": round(component.historical_extremeness, 4),
            "recurrence_rate": round(component.recurrence_rate, 4),
            "novelty": round(component.novelty, 4),
            "effective_score": round(component.effective_score, 4),
            "intensity_effective_score": round(
                component.intensity_effective_score, 4
            ),
            "alert_effective_score": round(component.alert_effective_score, 4),
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
    alert_score: float | None = None,
) -> str:
    label = intensity_label(score)
    directional = direction_label(direction)
    recurrence_note = (
        " The anomaly remains strong but is no longer novel, so alert and "
        "actionability scores are suppressed."
        if score >= 40.0
        and alert_score is not None
        and alert_score <= max(5.0, score * 0.10)
        else ""
    )
    if not reasons:
        return (
            f"MAPI detected {label} anomaly pressure with a {directional} lean in "
            f"the {regime} regime. The signal is {state} with "
            f"{confirmation_count} confirming components.{recurrence_note}"
        )
    joined = " ".join(reasons[:3])
    return (
        f"MAPI detected {label} {directional} anomaly pressure in the {regime} "
        f"regime. {joined} The signal is {state} with {confirmation_count} "
        f"confirming components.{recurrence_note}"
    )
