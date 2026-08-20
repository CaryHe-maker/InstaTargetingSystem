"""Domain-conservative RGB augmentations for local training views."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class LocalViewAugmenter:
    brightness: float = 0.15
    contrast: float = 0.15
    noiseStd: float = 3.0
    blurProbability: float = 0.10
    jpegProbability: float = 0.10

    def __call__(
        self, rgb: NDArray[np.uint8], rng: np.random.Generator
    ) -> NDArray[np.uint8]:
        import cv2

        image = rgb.astype(np.float32)
        contrast = float(rng.uniform(1.0 - self.contrast, 1.0 + self.contrast))
        brightness = float(rng.uniform(-self.brightness, self.brightness) * 255.0)
        image = image * contrast + brightness
        if self.noiseStd > 0.0:
            image += rng.normal(0.0, self.noiseStd, image.shape).astype(np.float32)
        result = np.clip(image, 0.0, 255.0).astype(np.uint8)
        if rng.random() < self.blurProbability:
            result = cv2.GaussianBlur(result, (3, 3), 0.0)
        if rng.random() < self.jpegProbability:
            quality = int(rng.integers(55, 96))
            ok, encoded = cv2.imencode(
                ".jpg",
                cv2.cvtColor(result, cv2.COLOR_RGB2BGR),
                [cv2.IMWRITE_JPEG_QUALITY, quality],
            )
            if ok:
                decoded = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                result = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        return np.ascontiguousarray(result)


__all__ = ["LocalViewAugmenter"]
