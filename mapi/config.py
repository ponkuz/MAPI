from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from mapi.models import HorizonConfig


DEFAULT_HORIZONS: dict[str, HorizonConfig] = {
    "intraday": HorizonConfig("intraday", return_window=1, rolling_window=20, min_periods=8),
    "short_term": HorizonConfig(
        "short_term", return_window=5, rolling_window=60, min_periods=20
    ),
    "swing": HorizonConfig("swing", return_window=20, rolling_window=120, min_periods=40),
    "position": HorizonConfig(
        "position", return_window=60, rolling_window=252, min_periods=80
    ),
}


DEFAULT_COMPONENT_WEIGHTS: dict[str, float] = {
    "price_volume_divergence": 0.24,
    "stock_sector_divergence": 0.20,
    "momentum_disagreement": 0.18,
    "volatility_anomaly": 0.18,
    "market_regime_divergence": 0.20,
}


DEFAULT_COMPONENT_RELIABILITY: dict[str, float] = {
    name: 1.0 for name in DEFAULT_COMPONENT_WEIGHTS
}


@dataclass
class MapiConfig:
    signal_version: str = "mapi_v0.2"
    score_semantics: str = "pure_anomaly"
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
    already_realized_penalty: float = 0.15
    transaction_cost_bps: float = 10.0
    spread_bps: float = 2.0
    slippage_bps: float = 5.0
    deterministic_seed: int = 42
    adjusted_prices_required: bool = True

    def with_disabled_component(self, component_name: str) -> "MapiConfig":
        enabled = [name for name in self.enabled_components if name != component_name]
        return replace(self, enabled_components=enabled)


def load_config(path: str | Path | None = None) -> MapiConfig:
    if path is None:
        return MapiConfig()
    raw = _load_mapping(Path(path))
    return _config_from_mapping(raw)


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
            )
        config.horizons = horizons
    for key in (
        "redundancy_threshold",
        "min_redundancy_penalty",
        "cross_asset_max_staleness_bars",
        "cross_asset_frequency_tolerance",
        "high_anomaly_threshold",
        "strong_anomaly_threshold",
        "already_realized_penalty",
        "transaction_cost_bps",
        "spread_bps",
        "slippage_bps",
    ):
        if key in raw:
            setattr(config, key, float(raw[key]))
    if "redundancy_window" in raw:
        config.redundancy_window = int(raw["redundancy_window"])
    if "redundancy_min_periods" in raw:
        config.redundancy_min_periods = int(raw["redundancy_min_periods"])
    if "deterministic_seed" in raw:
        config.deterministic_seed = int(raw["deterministic_seed"])
    if "adjusted_prices_required" in raw:
        config.adjusted_prices_required = bool(raw["adjusted_prices_required"])
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
