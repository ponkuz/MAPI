from __future__ import annotations

from mapi.models import clamp


def directional_realization_score(
    direction: float,
    move_since_detection: float,
    historical_move_scale: float,
    minimum_direction: float = 0.10,
) -> float:
    """Score only price movement aligned with the current anomaly direction."""

    if abs(direction) < minimum_direction or historical_move_scale <= 0.0:
        return 0.0
    aligned_move = direction * move_since_detection
    if aligned_move <= 0.0:
        return 0.0
    return clamp(aligned_move / historical_move_scale, 0.0, 1.0)
