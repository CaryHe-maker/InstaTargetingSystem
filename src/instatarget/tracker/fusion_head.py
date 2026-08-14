"""Lightweight score fusion for RGB HiT, depth HiT and context evidence."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from instatarget.core.errors import ModelError


@dataclass(frozen=True, slots=True)
class FusionInput:
    rgbScore: float
    depthScore: float
    contextScore: float = 0.0


class FusionHead:
    """A bounded weighted fusion head with an explicit RGB-only fallback."""

    def __init__(
        self,
        rgbInitWeight: float | object = 0.70,
        depthInitWeight: float = 0.20,
        contextInitWeight: float = 0.10,
        depthScoreWeight: float | None = None,
    ) -> None:
        if not isinstance(rgbInitWeight, (int, float)) and hasattr(rgbInitWeight, "rgbInitWeight"):
            config = rgbInitWeight
            rgbInitWeight = config.rgbInitWeight
            depthInitWeight = config.depthInitWeight
            contextInitWeight = config.contextInitWeight
        weights = (rgbInitWeight, depthInitWeight, contextInitWeight)
        if any(not np.isfinite(weight) or weight < 0.0 for weight in weights):
            raise ModelError("fusion weights must be finite and non-negative")
        if sum(weights) <= 0.0:
            raise ModelError("fusion weights must contain a positive value")
        self.rgbInitWeight, self.depthInitWeight, self.contextInitWeight = map(float, weights)
        self.depthScoreWeight = (
            float(depthScoreWeight) if depthScoreWeight is not None else self.depthInitWeight
        )
        if not 0.0 <= self.depthScoreWeight <= 1.0:
            raise ModelError("depthScoreWeight must be in [0, 1]")

    def fuse(
        self,
        rgbScore: float | FusionInput,
        depthScore: float | None = None,
        contextScore: float = 0.0,
        depthAvailable: bool = True,
    ) -> float:
        if isinstance(rgbScore, FusionInput):
            inputs = rgbScore
            rgbScore, depthScore, contextScore = (
                inputs.rgbScore,
                inputs.depthScore,
                inputs.contextScore,
            )
        if depthScore is None:
            depthScore = 0.0
        _validateScore("rgbScore", rgbScore)
        _validateScore("depthScore", depthScore)
        _validateScore("contextScore", contextScore)
        if not depthAvailable or self.depthScoreWeight == 0.0:
            return float(rgbScore)
        # backendFusion.depthScoreWeight is the externally registered knob;
        # the remaining mass stays with RGB and optional context.
        depthWeight = self.depthScoreWeight
        rgbWeight = self.rgbInitWeight * (1.0 - depthWeight)
        contextWeight = self.contextInitWeight * (1.0 - depthWeight)
        total = rgbWeight + depthWeight + contextWeight
        fused = (
            rgbWeight * rgbScore + depthWeight * depthScore + contextWeight * contextScore
        ) / total
        return float(np.clip(fused, 0.0, 1.0))

    __call__ = fuse


def _validateScore(name: str, value: float) -> None:
    if not np.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ModelError(f"{name} must be finite and in [0, 1], actual={value}")


__all__ = ["FusionHead", "FusionInput"]
