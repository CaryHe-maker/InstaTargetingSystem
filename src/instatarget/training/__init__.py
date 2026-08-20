"""Training APIs without eagerly loading dataset-only runtime dependencies."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from instatarget.training.dataset import (
        AirSim360TrainingDataset,
        ManifestPairDataset,
        ManifestRecord,
        TrainingPair,
        TrainingSample,
        loadManifest,
    )

_DATASET_EXPORTS = frozenset(
    {
        "AirSim360TrainingDataset",
        "ManifestPairDataset",
        "ManifestRecord",
        "TrainingPair",
        "TrainingSample",
        "loadManifest",
    }
)


def __getattr__(name: str) -> Any:
    if name not in _DATASET_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from instatarget.training import dataset

    value = getattr(dataset, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted((*globals(), *_DATASET_EXPORTS))


__all__ = [
    "AirSim360TrainingDataset",
    "ManifestPairDataset",
    "ManifestRecord",
    "TrainingPair",
    "TrainingSample",
    "loadManifest",
]
