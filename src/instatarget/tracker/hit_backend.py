"""Narrow adapter for a HiT RGB inference session.

The project deliberately keeps the third-party HiT implementation outside the
core package.  A production adapter can wrap the official PyTorch, ONNX, or
TensorRT session while this module owns input/output validation and error
translation.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from instatarget.core.errors import ModelError, ProtocolError
from instatarget.core.types import BBoxXYWH


@dataclass(frozen=True, slots=True)
class HiTPrediction:
    """Raw local prediction returned by the HiT main trunk."""

    bbox: BBoxXYWH
    modelScore: float
    appearanceScore: float
    presenceLogit: float | None = None
    qualityLogit: float | None = None
    presenceProbability: float | None = None
    qualityProbability: float | None = None
    predictedIoU: float | None = None
    cornerScore: float | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("modelScore", self.modelScore),
            ("appearanceScore", self.appearanceScore),
        ):
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ModelError(f"{name} must be finite and in [0, 1], actual={value}")
        for name, value in (
            ("presenceLogit", self.presenceLogit),
            ("qualityLogit", self.qualityLogit),
        ):
            if value is not None and not np.isfinite(value):
                raise ModelError(f"{name} must be finite, actual={value}")
        for name, value in (
            ("presenceProbability", self.presenceProbability),
            ("qualityProbability", self.qualityProbability),
            ("predictedIoU", self.predictedIoU),
            ("cornerScore", self.cornerScore),
        ):
            if value is not None and (not np.isfinite(value) or not 0.0 <= value <= 1.0):
                raise ModelError(f"{name} must be finite and in [0, 1], actual={value}")


@runtime_checkable
class HiTSession(Protocol):
    """The only model-facing interface required by the tracker backend."""

    @property
    def supportsOnlineTemplates(self) -> bool: ...

    def encodeTemplate(self, rgb: NDArray[np.uint8], bbox: BBoxXYWH) -> object: ...

    def infer(
        self,
        rgb: NDArray[np.uint8],
        templateFeatures: Sequence[object],
    ) -> HiTPrediction: ...

    def close(self) -> None: ...


class HiTBackend:
    """Validated, exception-translating facade around a HiT session."""

    def __init__(self, session: HiTSession) -> None:
        if not isinstance(session, HiTSession):
            raise ProtocolError("session must implement the HiTSession protocol")
        self._session = session
        self._closed = False

    @property
    def supportsOnlineTemplates(self) -> bool:
        return self._session.supportsOnlineTemplates

    @property
    def lastProfile(self) -> dict[str, int | float | bool | str]:
        value = getattr(self._session, "lastProfile", {})
        return dict(value) if isinstance(value, dict) else {}

    def encodeTemplate(self, rgb: NDArray[np.uint8], bbox: BBoxXYWH) -> object:
        self._requireOpen()
        _requireRgb(rgb)
        try:
            return self._session.encodeTemplate(rgb, bbox)
        except (ModelError, ProtocolError):
            raise
        except Exception as error:
            raise ModelError(f"HiT template encoding failed: {error}") from error

    def infer(self, rgb: NDArray[np.uint8], templateFeatures: Sequence[object]) -> HiTPrediction:
        return self.inferBatch((rgb,), templateFeatures)[0]

    def inferBatch(
        self,
        rgbs: Sequence[NDArray[np.uint8]],
        templateFeatures: Sequence[object],
    ) -> tuple[HiTPrediction, ...]:
        self._requireOpen()
        images = tuple(rgbs)
        for rgb in images:
            _requireRgb(rgb)
        if not images:
            return ()
        if not templateFeatures:
            raise ProtocolError("HiT inference requires at least one template feature")
        try:
            batchInfer = getattr(self._session, "inferBatch", None)
            predictions = (
                tuple(batchInfer(images, templateFeatures))
                if callable(batchInfer)
                else tuple(self._session.infer(rgb, templateFeatures) for rgb in images)
            )
        except (ModelError, ProtocolError):
            raise
        except Exception as error:
            raise ModelError(f"HiT batch inference failed: {error}") from error
        if len(predictions) != len(images):
            raise ModelError(
                "HiT session returned an invalid batch size: "
                f"expected={len(images)}, actual={len(predictions)}"
            )
        if any(not isinstance(prediction, HiTPrediction) for prediction in predictions):
            raise ModelError("HiT session returned an invalid prediction object")
        return predictions

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._session.close()
        except Exception as error:
            raise ModelError(f"HiT session close failed: {error}") from error
        finally:
            self._closed = True

    def _requireOpen(self) -> None:
        if self._closed:
            raise ProtocolError("HiT backend is closed")


def _requireRgb(rgb: NDArray[np.uint8]) -> None:
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8:
        raise ProtocolError("HiT input rgb must be a uint8 NumPy array")
    if rgb.ndim != 3 or rgb.shape[2] != 3 or 0 in rgb.shape[:2]:
        raise ProtocolError(f"HiT input rgb must have shape [H, W, 3], actual={rgb.shape}")


__all__ = ["HiTBackend", "HiTPrediction", "HiTSession"]
