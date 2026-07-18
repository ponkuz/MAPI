from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import isfinite
from pathlib import Path
from typing import Any

from mapi.models import HorizonConfig


DEFAULT_HORIZONS: dict[str, HorizonConfig] = {
    "intraday": HorizonConfig(
        "intraday", return_window=1, rolling_window=20, min_periods=8,
        expected_frequency="intraday",
    ),
    "short_term": HorizonConfig(
        "short_term", return_window=5, rolling_window=60, min_periods=20,
        expected_frequency="daily",
    ),
    "swing": HorizonConfig(
        "swing", return_window=20, rolling_window=120, min_periods=40,
        expected_frequency="daily",
    ),
    "position": HorizonConfig(
        "position", return_window=60, rolling_window=252, min_periods=80,
        expected_frequency="daily",
    ),
}

DEFAULT_COMPONENT_WEIGHTS: dict[str, float] = {
    "price_volume_divergence": 0.24,
    "stock_sector_divergence": 0.20,
    "momentum_disagreement": 0.18,
    "volatility_anomaly": 0.18,
    "market_regime_divergence": 0.20,
}


KNOWN_COMPONENTS = {
    *DEFAULT_COMPONENT_WEIGHTS,
    "sentiment_crowding_anomaly",
    "options_anomaly",
    "fundamental_expectations_divergence",
    "news_reaction_divergence",
}


DEFAULT_COMPONENT_RELIABILITY: dict[str, float] = {
    name: 1.0 for name in DEFAULT_COMPONENT_WEIGHTS
}


@dataclass
class MapiConfig:
    signal_version: str = "mapi_v0.2"
    score_semantics: str = "pure_anomaly"
    research_score_column: str = "mapi_actionability_score"
    enabled_components: list[str] = field(
        default_factory=lambda: list(DEFAULT_COMPONENT_WEIGHTS)
    )
    component_weights: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_COMPONENT_WEIGHTS)
    )
    component_reliability: dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_COMPONENT_RELIABILITY)
    )
    horizons: dict[str, HorizonConfig] = field(
        default_factory=lambda: dict(DEFAULT_HORIZONS)
    )
    redundancy_threshold: float = 0.82
    redundancy_window: int = 80
    redundancy_min_periods: int = 8
    min_redundancy_penalty: float = 0.55
    cross_asset_max_staleness_bars: float = 1.5
    cross_asset_frequency_tolerance: float = 1.5
    high_anomaly_threshold: float = 40.0
    strong_anomaly_threshold: float = 60.0
    directional_realization_penalty: float = 0.15
    transaction_cost_bps: float = 10.0
    spread_bps: float = 2.0
    slippage_bps: float = 5.0
    deterministic_seed: int = 42
    adjusted_prices_required: bool = True
    frequency_mismatch_policy: str = "warn"
    bootstrap_samples: int = 1000
    large_move_threshold: float = 0.02

    def with_disabled_component(self, component_name: str) -> "MapiConfig":
        enabled = [name for name in self.enabled_components if name != component_name]
        return replace(self, enabled_components=enabled)

    @property
    def already_realized_penalty(self) -> float:
        """Compatibility alias for directional_realization_penalty."""

        return self.directional_realization_penalty

    @already_realized_penalty.setter
    def already_realized_penalty(self, value: float) -> None:
        self.directional_realization_penalty = value

    def validate(self, available_components: set[str] | None = None) -> None:
        if self.score_semantics not in {"pure_anomaly", "legacy_actionability"}:
            raise ValueError(f"Invalid score_semantics: {self.score_semantics}")
        if self.research_score_column not in {
            "mapi_score", "mapi_raw_score", "mapi_actionability_score"
        }:
            raise ValueError(
                f"Invalid research_score_column: {self.research_score_column}"
            )
        explicitly_available = available_components or set()
        unknown = set(self.enabled_components) - KNOWN_COMPONENTS - explicitly_available
        if unknown:
            raise ValueError(f"Unknown enabled components: {sorted(unknown)}")
        if len(self.enabled_components) != len(set(self.enabled_components)):
            raise ValueError("enabled_components must not contain duplicates")
        if available_components is not None:
            unavailable = set(self.enabled_components) - available_components
            if unavailable:
                raise ValueError(
                    f"Enabled components have no implementation: {sorted(unavailable)}"
                )
        for name, weight in self.component_weights.items():
            if not isfinite(weight) or weight < 0.0:
                raise ValueError(
                    f"Component weight for {name} must be finite and non-negative"
                )
        for name, reliability in self.component_reliability.items():
            if not isfinite(reliability) or not 0.0 <= reliability <= 1.0:
                raise ValueError(f"Component reliability for {name} must be in [0, 1]")
        enabled_weights = []
        for name in self.enabled_components:
            weight = self.component_weights.get(name)
            if weight is None:
                raise ValueError(f"Component weight for {name} is required")
            enabled_weights.append(weight)
        if not enabled_weights or sum(enabled_weights) <= 0.0:
            raise ValueError("Total enabled component weight must be positive")
        if not self.horizons:
            raise ValueError("At least one horizon is required")
        for name, horizon in self.horizons.items():
            if horizon.return_window <= 0 or horizon.rolling_window <= 0:
                raise ValueError(f"Horizon {name} windows must be positive")
            if horizon.min_periods <= 0 or horizon.min_periods > horizon.rolling_window:
                raise ValueError(
                    f"Horizon {name} requires 0 < min_periods <= rolling_window"
                )
            if horizon.expected_frequency not in {"any", "intraday", "daily"}:
                raise ValueError(
                    f"Horizon {name} has invalid expected_frequency: "
                    f"{horizon.expected_frequency}"
                )
        if not isfinite(self.redundancy_threshold) or not 0.0 <= self.redundancy_threshold <= 1.0:
            raise ValueError("redundancy_threshold must be in [0, 1]")
        if self.redundancy_window <= 0 or self.redundancy_min_periods <= 0:
            raise ValueError("Redundancy windows must be positive")
        if self.redundancy_min_periods > self.redundancy_window:
            raise ValueError("redundancy_min_periods must not exceed redundancy_window")
        for name, horizon in self.horizons.items():
            effective_window = max(8, min(self.redundancy_window, horizon.rolling_window))
            if self.redundancy_min_periods > effective_window:
                raise ValueError(
                    f"redundancy_min_periods exceeds the effective window for {name}"
                )
        if not isfinite(self.min_redundancy_penalty) or not 0.0 <= self.min_redundancy_penalty <= 1.0:
            raise ValueError("min_redundancy_penalty must be in [0, 1]")
        if not isfinite(self.cross_asset_max_staleness_bars) or self.cross_asset_max_staleness_bars < 0.0:
            raise ValueError("cross_asset_max_staleness_bars must be non-negative")
        if not isfinite(self.cross_asset_frequency_tolerance) or self.cross_asset_frequency_tolerance <= 0.0:
            raise ValueError("cross_asset_frequency_tolerance must be positive")
        if not 0.0 <= self.high_anomaly_threshold < self.strong_anomaly_threshold <= 100.0:
            raise ValueError(
                "Anomaly thresholds must satisfy 0 <= high < strong <= 100"
            )
        if not 0.0 <= self.directional_realization_penalty <= 1.0:
            raise ValueError("directional_realization_penalty must be in [0, 1]")
        for name in ("transaction_cost_bps", "spread_bps", "slippage_bps"):
            value = getattr(self, name)
            if not isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.frequency_mismatch_policy not in {"warn", "error"}:
            raise ValueError("frequency_mismatch_policy must be 'warn' or 'error'")
        if self.bootstrap_samples <= 0:
            raise ValueError("bootstrap_samples must be positive")
        if not isfinite(self.large_move_threshold) or self.large_move_threshold <= 0.0:
            raise ValueError("large_move_threshold must be finite and positive")


def load_config(path: str | Path | None = None) -> MapiConfig:
    if path is None:
        config = MapiConfig()
        config.validate()
        return config
    raw = _load_mapping(Path(path))
    config = _config_from_mapping(raw)
    config.validate()
    return config


def _load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        return _parse_simple_yaml(text)
    loaded = yaml.safe_load(text)
    return loaded or {}


def _config_from_mapping(raw: dict[str, Any]) -> MapiConfig:
    config = MapiConfig()
    if "signal_version" in raw:
        config.signal_version = str(raw["signal_version"])
    if "score_semantics" in raw:
        config.score_semantics = str(raw["score_semantics"])
    if "research_score_column" in raw:
        config.research_score_column = str(raw["research_score_column"])
    if "enabled_components" in raw:
        config.enabled_components = [str(item) for item in raw["enabled_components"]]
    if "component_weights" in raw:
        merged = dict(config.component_weights)
        merged.update({str(k): float(v) for k, v in raw["component_weights"].items()})
        config.component_weights = merged
    if "component_reliability" in raw:
        merged_reliability = dict(config.component_reliability)
        merged_reliability.update(
            {str(k): float(v) for k, v in raw["component_reliability"].items()}
        )
        config.component_reliability = merged_reliability
    if "horizons" in raw:
        horizons = dict(config.horizons)
        for name, values in raw["horizons"].items():
            existing = horizons.get(str(name), DEFAULT_HORIZONS.get(str(name)))
            if existing is None:
                existing = HorizonConfig(str(name), 1, 20, 8)
            horizons[str(name)] = HorizonConfig(
                name=str(name),
                return_window=int(values.get("return_window", existing.return_window)),
                rolling_window=int(values.get("rolling_window", existing.rolling_window)),
                min_periods=int(values.get("min_periods", existing.min_periods)),
                expected_frequency=str(
                    values.get("expected_frequency", existing.expected_frequency)
                ),
            )
        config.horizons = horizons
    for key in (
        "redundancy_threshold",
        "min_redundancy_penalty",
        "cross_asset_max_staleness_bars",
        "cross_asset_frequency_tolerance",
        "high_anomaly_threshold",
        "strong_anomaly_threshold",
        "directional_realization_penalty",
        "transaction_cost_bps",
        "spread_bps",
        "slippage_bps",
        "large_move_threshold",
    ):
        if key in raw:
            setattr(config, key, float(raw[key]))
    if (
        "already_realized_penalty" in raw
        and "directional_realization_penalty" not in raw
    ):
        config.directional_realization_penalty = float(raw["already_realized_penalty"])
    if "redundancy_window" in raw:
        config.redundancy_window = int(raw["redundancy_window"])
    if "redundancy_min_periods" in raw:
        config.redundancy_min_periods = int(raw["redundancy_min_periods"])
    if "deterministic_seed" in raw:
        config.deterministic_seed = int(raw["deterministic_seed"])
    if "adjusted_prices_required" in raw:
        config.adjusted_prices_required = bool(raw["adjusted_prices_required"])
    if "frequency_mismatch_policy" in raw:
        config.frequency_mismatch_policy = str(raw["frequency_mismatch_policy"])
    if "bootstrap_samples" in raw:
        config.bootstrap_samples = int(raw["bootstrap_samples"])
    return config


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for raw_line in text.splitlines():
        stripped_comment = raw_line.split("#", 1)[0].rstrip()
        if not stripped_comment.strip():
            continue
        indent = len(stripped_comment) - len(stripped_comment.lstrip(" "))
        key, separator, value = stripped_comment.strip().partition(":")
        if separator != ":":
            continue
        while stack and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        value = value.strip()
        if not value:
            child: dict[str, Any] = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _parse_scalar(value)
    return root


def _parse_scalar(value: str) -> Any:
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        if not inner:
            return []
        return [_parse_scalar(part.strip()) for part in inner.split(",")]
    lowered = value.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if lowered in {"null", "none"}:
        return None
    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]
    try:
        if any(ch in value for ch in (".", "e", "E")):
            return float(value)
        return int(value)
    except ValueError:
        return value
