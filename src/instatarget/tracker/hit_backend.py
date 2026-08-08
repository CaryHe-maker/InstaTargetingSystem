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

    def __post_init__(self) -> None:
        for name, value in (
            ("modelScore", self.modelScore),
            ("appearanceScore", self.appearanceScore),
        ):
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
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
        self._requireOpen()
        _requireRgb(rgb)
        if not templateFeatures:
            raise ProtocolError("HiT inference requires at least one template feature")
        try:
            prediction = self._session.infer(rgb, templateFeatures)
        except (ModelError, ProtocolError):
            raise
        except Exception as error:
            raise ModelError(f"HiT inference failed: {error}") from error
        if not isinstance(prediction, HiTPrediction):
            raise ModelError("HiT session returned an invalid prediction object")
        return prediction

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
