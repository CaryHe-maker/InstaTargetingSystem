"""Strict, immutable configuration loading for runtime components."""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite, pi
from pathlib import Path
from typing import cast

from instatarget.core.errors import ConfigError

SUPPORTED_SCHEMA_VERSION = 1
VISUALIZATION_STAGES = frozenset(
    {"local_rgb", "depth_rgb", "backend_box", "geometry_box"}
)


@dataclass(frozen=True, slots=True)
class ModelConfig:
    backend: str
    variant: str
    weights: Path
    precision: str

    def __post_init__(self) -> None:
        if self.backend not in {"pytorch", "onnxruntime", "tensorrt"}:
            raise ConfigError(f"unsupported model.backend: {self.backend}")
        if not self.variant.strip():
            raise ConfigError("model.variant must be non-empty")
        if self.precision not in {"fp32", "fp16"}:
            raise ConfigError(f"unsupported model.precision: {self.precision}")


@dataclass(frozen=True, slots=True)
class GeometryConfig:
    viewWidthPx: int
    viewHeightPx: int
    boundarySamplesPerEdge: int
    minFovRad: float
    maxFovRad: float

    def __post_init__(self) -> None:
        if self.viewWidthPx <= 0 or self.viewHeightPx <= 0:
            raise ConfigError("geometry view dimensions must be positive")
        if self.boundarySamplesPerEdge < 2:
            raise ConfigError("geometry.boundarySamplesPerEdge must be at least 2")
        if not 0.0 < self.minFovRad < self.maxFovRad < pi:
            raise ConfigError("geometry FOV must satisfy 0 < minFovRad < maxFovRad < pi")


@dataclass(frozen=True, slots=True)
class DepthConfig:
    enabled: bool
    minValidRatio: float
    maxDepthJumpRatio: float

    def __post_init__(self) -> None:
        _requireProbability("depth.minValidRatio", self.minValidRatio)
        _requireProbability("depth.maxDepthJumpRatio", self.maxDepthJumpRatio)


@dataclass(frozen=True, slots=True)
class BackendFusionConfig:
    depthScoreWeight: float

    def __post_init__(self) -> None:
        _requireProbability("backendFusion.depthScoreWeight", self.depthScoreWeight)


@dataclass(frozen=True, slots=True)
class DecisionGateConfig:
    motionScoreWeight: float
    scaleScoreWeight: float

    def __post_init__(self) -> None:
        _requireProbability("decisionGate.motionScoreWeight", self.motionScoreWeight)
        _requireProbability("decisionGate.scaleScoreWeight", self.scaleScoreWeight)
        if self.motionScoreWeight + self.scaleScoreWeight > 1.0:
            raise ConfigError("decision gate motion and scale weights must sum to at most 1")


@dataclass(frozen=True, slots=True)
class TrackingConfig:
    acceptThreshold: float
    uncertainThreshold: float
    stableFramesBeforeUpdate: int
    windowLength: int

    def __post_init__(self) -> None:
        _requireProbability("tracking.acceptThreshold", self.acceptThreshold)
        _requireProbability("tracking.uncertainThreshold", self.uncertainThreshold)
        if self.uncertainThreshold >= self.acceptThreshold:
            raise ConfigError("tracking thresholds must satisfy uncertain < accept")
        if self.stableFramesBeforeUpdate <= 0:
            raise ConfigError("tracking.stableFramesBeforeUpdate must be positive")
        if self.windowLength < 2:
            raise ConfigError("tracking.windowLength must be at least 2")


@dataclass(frozen=True, slots=True)
class RecoveryConfig:
    maxViewsPerFrame: int
    globalSearchInterval: int

    def __post_init__(self) -> None:
        if self.maxViewsPerFrame <= 0 or self.globalSearchInterval <= 0:
            raise ConfigError("recovery limits must be positive")


@dataclass(frozen=True, slots=True)
class RuntimeConfig:
    decodeQueueCapacity: int
    inferRequestQueueCapacity: int
    inferResponseQueueCapacity: int
    resultQueueCapacity: int

    def __post_init__(self) -> None:
        capacities = (
            self.decodeQueueCapacity,
            self.inferRequestQueueCapacity,
            self.inferResponseQueueCapacity,
            self.resultQueueCapacity,
        )
        if any(capacity <= 0 for capacity in capacities):
            raise ConfigError("runtime queue capacities must be positive")


@dataclass(frozen=True, slots=True)
class VisualizationConfig:
    enabled: bool
    outputRoot: Path
    stages: frozenset[str]

    def __post_init__(self) -> None:
        unknown = self.stages - VISUALIZATION_STAGES
        if unknown:
            raise ConfigError(f"unsupported visualization stages: {sorted(unknown)}")
        if self.enabled and not self.stages:
            raise ConfigError("visualization.stages must not be empty when enabled")


@dataclass(frozen=True, slots=True)
class AppConfig:
    schemaVersion: int
    model: ModelConfig
    geometry: GeometryConfig
    depth: DepthConfig
    backendFusion: BackendFusionConfig
    decisionGate: DecisionGateConfig
    tracking: TrackingConfig
    recovery: RecoveryConfig
    runtime: RuntimeConfig
    visualization: VisualizationConfig
    sourcePath: Path


def loadConfig(path: str | Path) -> AppConfig:
    """Load a YAML configuration and reject unknown or missing fields."""
    configPath = Path(path).expanduser().resolve()
    try:
        import yaml
    except ModuleNotFoundError as error:
        raise ConfigError("PyYAML is required to load configuration files") from error

    try:
        with configPath.open("r", encoding="utf-8") as stream:
            raw = yaml.safe_load(stream)
    except OSError as error:
        raise ConfigError(f"cannot read config file {configPath}: {error}") from error
    except yaml.YAMLError as error:
        raise ConfigError(f"invalid YAML in {configPath}: {error}") from error

    root = _requireMapping("config", raw)
    _requireKeys(
        "config",
        root,
        {
            "schemaVersion",
            "model",
            "geometry",
            "depth",
            "backendFusion",
            "decisionGate",
            "tracking",
            "recovery",
            "runtime",
            "visualization",
        },
    )

    schemaVersion = _requireInt("schemaVersion", root["schemaVersion"])
    if schemaVersion != SUPPORTED_SCHEMA_VERSION:
        raise ConfigError(
            f"unsupported schemaVersion: expected={SUPPORTED_SCHEMA_VERSION}, "
            f"actual={schemaVersion}"
        )

    modelRaw = _section(root, "model", {"backend", "variant", "weights", "precision"})
    geometryRaw = _section(
        root,
        "geometry",
        {"viewWidthPx", "viewHeightPx", "boundarySamplesPerEdge", "minFovDeg", "maxFovDeg"},
    )
    depthRaw = _section(root, "depth", {"enabled", "minValidRatio", "maxDepthJumpRatio"})
    fusionRaw = _section(root, "backendFusion", {"depthScoreWeight"})
    gateRaw = _section(root, "decisionGate", {"motionScoreWeight", "scaleScoreWeight"})
    trackingRaw = _section(
        root,
        "tracking",
        {"acceptThreshold", "uncertainThreshold", "stableFramesBeforeUpdate", "windowLength"},
    )
    recoveryRaw = _section(root, "recovery", {"maxViewsPerFrame", "globalSearchInterval"})
    runtimeRaw = _section(
        root,
        "runtime",
        {
            "decodeQueueCapacity",
            "inferRequestQueueCapacity",
            "inferResponseQueueCapacity",
            "resultQueueCapacity",
        },
    )
    visualizationRaw = _section(root, "visualization", {"enabled", "outputRoot", "stages"})

    weightsValue = _requireStr("model.weights", modelRaw["weights"])
    weightsPath = Path(weightsValue).expanduser()
    if not weightsPath.is_absolute():
        weightsPath = (configPath.parent / weightsPath).resolve()

    outputRootValue = _requireStr("visualization.outputRoot", visualizationRaw["outputRoot"])
    outputRoot = Path(outputRootValue).expanduser()
    if not outputRoot.is_absolute():
        outputRoot = (configPath.parent / outputRoot).resolve()

    return AppConfig(
        schemaVersion=schemaVersion,
        model=ModelConfig(
            backend=_requireStr("model.backend", modelRaw["backend"]),
            variant=_requireStr("model.variant", modelRaw["variant"]),
            weights=weightsPath,
            precision=_requireStr("model.precision", modelRaw["precision"]),
        ),
        geometry=GeometryConfig(
            viewWidthPx=_requireInt("geometry.viewWidthPx", geometryRaw["viewWidthPx"]),
            viewHeightPx=_requireInt("geometry.viewHeightPx", geometryRaw["viewHeightPx"]),
            boundarySamplesPerEdge=_requireInt(
                "geometry.boundarySamplesPerEdge", geometryRaw["boundarySamplesPerEdge"]
            ),
            minFovRad=_degreesToRadians(
                "geometry.minFovDeg", _requireFloat("geometry.minFovDeg", geometryRaw["minFovDeg"])
            ),
            maxFovRad=_degreesToRadians(
                "geometry.maxFovDeg", _requireFloat("geometry.maxFovDeg", geometryRaw["maxFovDeg"])
            ),
        ),
        depth=DepthConfig(
            enabled=_requireBool("depth.enabled", depthRaw["enabled"]),
            minValidRatio=_requireFloat("depth.minValidRatio", depthRaw["minValidRatio"]),
            maxDepthJumpRatio=_requireFloat(
                "depth.maxDepthJumpRatio", depthRaw["maxDepthJumpRatio"]
            ),
        ),
        backendFusion=BackendFusionConfig(
            depthScoreWeight=_requireFloat(
                "backendFusion.depthScoreWeight", fusionRaw["depthScoreWeight"]
            )
        ),
        decisionGate=DecisionGateConfig(
            motionScoreWeight=_requireFloat(
                "decisionGate.motionScoreWeight", gateRaw["motionScoreWeight"]
            ),
            scaleScoreWeight=_requireFloat(
                "decisionGate.scaleScoreWeight", gateRaw["scaleScoreWeight"]
            ),
        ),
        tracking=TrackingConfig(
            acceptThreshold=_requireFloat(
                "tracking.acceptThreshold", trackingRaw["acceptThreshold"]
            ),
            uncertainThreshold=_requireFloat(
                "tracking.uncertainThreshold", trackingRaw["uncertainThreshold"]
            ),
            stableFramesBeforeUpdate=_requireInt(
                "tracking.stableFramesBeforeUpdate", trackingRaw["stableFramesBeforeUpdate"]
            ),
            windowLength=_requireInt("tracking.windowLength", trackingRaw["windowLength"]),
        ),
        recovery=RecoveryConfig(
            maxViewsPerFrame=_requireInt(
                "recovery.maxViewsPerFrame", recoveryRaw["maxViewsPerFrame"]
            ),
            globalSearchInterval=_requireInt(
                "recovery.globalSearchInterval", recoveryRaw["globalSearchInterval"]
            ),
        ),
        runtime=RuntimeConfig(
            decodeQueueCapacity=_requireInt(
                "runtime.decodeQueueCapacity", runtimeRaw["decodeQueueCapacity"]
            ),
            inferRequestQueueCapacity=_requireInt(
                "runtime.inferRequestQueueCapacity", runtimeRaw["inferRequestQueueCapacity"]
            ),
            inferResponseQueueCapacity=_requireInt(
                "runtime.inferResponseQueueCapacity", runtimeRaw["inferResponseQueueCapacity"]
            ),
            resultQueueCapacity=_requireInt(
                "runtime.resultQueueCapacity", runtimeRaw["resultQueueCapacity"]
            ),
        ),
        visualization=VisualizationConfig(
            enabled=_requireBool("visualization.enabled", visualizationRaw["enabled"]),
            outputRoot=outputRoot,
            stages=_requireStringSet("visualization.stages", visualizationRaw["stages"]),
        ),
        sourcePath=configPath,
    )


def _requireProbability(name: str, value: float) -> None:
    if not 0.0 <= value <= 1.0:
        raise ConfigError(f"{name} must be in [0, 1], actual={value}")


def _requireMapping(name: str, value: object) -> dict[str, object]:
    if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
        raise ConfigError(f"{name} must be a mapping with string keys")
    return cast(dict[str, object], value)


def _requireKeys(name: str, value: dict[str, object], expected: set[str]) -> None:
    actual = set(value)
    missing = sorted(expected - actual)
    unknown = sorted(actual - expected)
    if missing or unknown:
        raise ConfigError(f"{name} fields invalid: missing={missing}, unknown={unknown}")


def _section(
    root: dict[str, object],
    name: str,
    expected: set[str],
) -> dict[str, object]:
    section = _requireMapping(name, root[name])
    _requireKeys(name, section, expected)
    return section


def _requireStr(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConfigError(f"{name} must be a non-empty string")
    return value


def _requireBool(name: str, value: object) -> bool:
    if not isinstance(value, bool):
        raise ConfigError(f"{name} must be a boolean")
    return value


def _requireInt(name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be an integer")
    return value


def _requireFloat(name: str, value: object) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a number")
    result = float(value)
    if not isfinite(result):
        raise ConfigError(f"{name} must be finite")
    return result


def _requireStringSet(name: str, value: object) -> frozenset[str]:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ConfigError(f"{name} must be a list of non-empty strings")
    if len(value) != len(set(value)):
        raise ConfigError(f"{name} must not contain duplicates")
    return frozenset(value)


def _degreesToRadians(name: str, valueDeg: float) -> float:
    if not 0.0 < valueDeg < 180.0:
        raise ConfigError(f"{name} must be in (0, 180), actual={valueDeg}")
    return valueDeg * pi / 180.0
