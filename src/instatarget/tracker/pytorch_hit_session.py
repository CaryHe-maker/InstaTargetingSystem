"""Production PyTorch adapter for the official HiT-Small implementation."""

from __future__ import annotations

import math
import os
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType
from typing import Any

import numpy as np
from numpy.typing import NDArray

from instatarget.core.config import ModelConfig
from instatarget.core.errors import ModelError, ProtocolError
from instatarget.core.types import BBoxXYWH
from instatarget.tracker.hit_backend import HiTPrediction

_SUPPORTED_VARIANT = "hit_small"
_TEMPLATE_FACTOR = 2.0
_TEMPLATE_SIZE = 128
_SEARCH_SIZE = 256


class PyTorchHiTSession:
    """Load and run the official HiT-Small PyTorch model on CUDA."""

    supportsOnlineTemplates = True

    def __init__(self, config: ModelConfig, *, hitRoot: str | Path | None = None) -> None:
        if config.backend != "pytorch":
            raise ModelError(f"PyTorch HiT session cannot use backend={config.backend}")
        if config.variant.lower() != _SUPPORTED_VARIANT:
            raise ModelError(
                f"unsupported HiT variant: {config.variant}; expected={_SUPPORTED_VARIANT}"
            )
        self._closed = False
        self._precision = config.precision
        self._torch = _importTorch()
        if not self._torch.cuda.is_available():
            raise ModelError("official HiT-Small requires a CUDA-capable PyTorch runtime")

        self._hitRoot = _resolveHitRoot(hitRoot)
        self._weights = Path(config.weights).expanduser().resolve()
        if not self._weights.is_file():
            raise ModelError(f"HiT checkpoint does not exist: {self._weights}")
        self._device = self._torch.device("cuda")
        self._model, self._heatmaps = self._loadModel()
        self._mean = self._torch.tensor(
            [0.485, 0.456, 0.406], device=self._device, dtype=self._torch.float32
        ).view(1, 3, 1, 1)
        self._std = self._torch.tensor(
            [0.229, 0.224, 0.225], device=self._device, dtype=self._torch.float32
        ).view(1, 3, 1, 1)

    def encodeTemplate(self, rgb: NDArray[np.uint8], bbox: BBoxXYWH) -> object:
        self._requireOpen()
        _requireRgb(rgb)
        patch = _sampleTarget(rgb, bbox, _TEMPLATE_FACTOR, _TEMPLATE_SIZE)
        return self._preprocess(patch)

    def infer(
        self,
        rgb: NDArray[np.uint8],
        templateFeatures: list[object] | tuple[object, ...],
    ) -> HiTPrediction:
        self._requireOpen()
        _requireRgb(rgb)
        if not templateFeatures:
            raise ProtocolError("HiT inference requires at least one template")
        template = templateFeatures[-1]
        if not self._torch.is_tensor(template):
            raise ProtocolError("HiT template feature must be a torch tensor")

        search = self._preprocess(_resizeRgb(rgb, _SEARCH_SIZE))
        self._heatmaps.clear()
        boxes = self._forward(search, template, useFp16=self._precision == "fp16")
        fp16Invalid = not bool(self._torch.isfinite(boxes).all()) or not _heatmapsAreFinite(
            self._heatmaps, self._torch
        )
        if self._precision == "fp16" and fp16Invalid:
            self._heatmaps.clear()
            boxes = self._forward(search, template, useFp16=False)
        cx, cy, width, height = boxes.reshape(-1, 4).float().mean(dim=0).tolist()
        bbox = _normalizedBoxToPixels(cx, cy, width, height, rgb.shape[1], rgb.shape[0])
        certainty = _heatmapCertainty(self._heatmaps, self._torch)
        return HiTPrediction(bbox=bbox, modelScore=certainty, appearanceScore=certainty)

    def close(self) -> None:
        if self._closed:
            return
        for hook in getattr(self, "_heatmapHooks", ()):
            hook.remove()
        self._model = None
        self._heatmaps.clear()
        self._closed = True
        self._torch.cuda.empty_cache()

    def _loadModel(self) -> tuple[Any, list[Any]]:
        _activateVendorTree(self._hitRoot)
        try:
            import lib.models.HiT.backbone as backboneModule
            from lib.config.HiT.config import cfg, update_config_from_file
            from lib.models.HiT import build_hit
            from lib.models.HiT.levit_utils import replace_batchnorm
        except Exception as error:
            raise ModelError(
                f"cannot import official HiT source from {self._hitRoot}: {error}"
            ) from error

        yamlPath = self._hitRoot / "experiments" / "HiT" / "HiT_Small.yaml"
        if not yamlPath.is_file():
            raise ModelError(f"HiT-Small config does not exist: {yamlPath}")
        update_config_from_file(str(yamlPath))
        backboneModule.is_main_process = lambda: False
        try:
            model = build_hit(cfg)
            checkpoint = _loadCheckpoint(self._torch, self._weights)
            state = checkpoint.get("net") if isinstance(checkpoint, dict) else None
            if not isinstance(state, dict):
                raise ModelError("HiT checkpoint has no 'net' state dictionary")
            model.load_state_dict(state, strict=True)
            replace_batchnorm(model.backbone.body)
            model = model.to(self._device).eval()
        except ModelError:
            raise
        except Exception as error:
            raise ModelError(f"cannot construct HiT-Small from {self._weights}: {error}") from error

        heatmaps: list[Any] = []

        def captureHeatmap(_module: Any, _inputs: Any, output: Any) -> None:
            heatmaps.append(output.detach())

        self._heatmapHooks = (
            model.box_head.conv5_tl.register_forward_hook(captureHeatmap),
            model.box_head.conv5_br.register_forward_hook(captureHeatmap),
        )
        return model, heatmaps

    def _preprocess(self, rgb: NDArray[np.uint8]) -> Any:
        tensor = self._torch.from_numpy(np.ascontiguousarray(rgb)).to(
            device=self._device, dtype=self._torch.float32
        )
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).div_(255.0)
        return (tensor - self._mean) / self._std

    def _forward(self, search: Any, template: Any, *, useFp16: bool) -> Any:
        autocast = (
            self._torch.autocast(device_type="cuda", dtype=self._torch.float16)
            if useFp16
            else nullcontext()
        )
        with self._torch.inference_mode(), autocast:
            features = self._model.forward_backbone(
                [search, template], first_score=None, threshold=0.9
            )
            output, _, _ = self._model.forward_head(features)
        boxes = output.get("pred_boxes")
        if boxes is None or boxes.numel() == 0:
            raise ModelError("HiT returned no predicted boxes")
        return boxes

    def _requireOpen(self) -> None:
        if self._closed:
            raise ProtocolError("HiT session is closed")


def _importTorch() -> ModuleType:
    try:
        import torch
    except ImportError as error:
        raise ModelError("PyTorch HiT requires torch and torchvision") from error
    return torch


def _resolveHitRoot(value: str | Path | None) -> Path:
    candidates = []
    if value is not None:
        candidates.append(Path(value))
    environment = os.environ.get("HIT_ROOT")
    if environment:
        candidates.append(Path(environment))
    candidates.append(Path(__file__).resolve().parents[3] / "third_party" / "HiT")
    for candidate in candidates:
        root = candidate.expanduser().resolve()
        if (root / "lib" / "models" / "HiT").is_dir():
            return root
    checked = ", ".join(str(item.expanduser()) for item in candidates)
    raise ModelError(f"official HiT source tree was not found; checked: {checked}")


def _activateVendorTree(root: Path) -> None:
    rootText = str(root)
    if rootText not in sys.path:
        sys.path.insert(0, rootText)


def _loadCheckpoint(torch: ModuleType, path: Path) -> dict[str, Any]:
    try:
        from lib.train.admin.local import EnvironmentSettings
        from lib.train.admin.settings import Settings
        from lib.train.admin.stats import AverageMeter, StatValue

        safeTypes = [AverageMeter, StatValue, Settings, EnvironmentSettings]
        with torch.serialization.safe_globals(safeTypes):
            return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ModelError(f"cannot safely load HiT checkpoint {path}: {error}") from error


def _sampleTarget(
    rgb: NDArray[np.uint8], bbox: BBoxXYWH, factor: float, outputSize: int
) -> NDArray[np.uint8]:
    import cv2

    cropSize = math.ceil(math.sqrt(bbox.widthPx * bbox.heightPx) * factor)
    if cropSize < 1:
        raise ProtocolError("HiT template box is too small")
    x1 = round(bbox.xPx + 0.5 * bbox.widthPx - 0.5 * cropSize)
    y1 = round(bbox.yPx + 0.5 * bbox.heightPx - 0.5 * cropSize)
    x2, y2 = x1 + cropSize, y1 + cropSize
    left, right = max(0, -x1), max(x2 - rgb.shape[1] + 1, 0)
    top, bottom = max(0, -y1), max(y2 - rgb.shape[0] + 1, 0)
    crop = rgb[y1 + top : y2 - bottom, x1 + left : x2 - right]
    padded = cv2.copyMakeBorder(crop, top, bottom, left, right, cv2.BORDER_CONSTANT)
    return np.ascontiguousarray(cv2.resize(padded, (outputSize, outputSize)))


def _resizeRgb(rgb: NDArray[np.uint8], size: int) -> NDArray[np.uint8]:
    if rgb.shape[:2] == (size, size):
        return np.ascontiguousarray(rgb)
    import cv2

    return np.ascontiguousarray(cv2.resize(rgb, (size, size)))


def _normalizedBoxToPixels(
    cx: float, cy: float, width: float, height: float, imageWidth: int, imageHeight: int
) -> BBoxXYWH:
    values = np.asarray([cx, cy, width, height], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ModelError(f"HiT returned a non-finite box: {values.tolist()}")
    widthPx = float(np.clip(width * imageWidth, 1.0, imageWidth))
    heightPx = float(np.clip(height * imageHeight, 1.0, imageHeight))
    centerX = float(np.clip(cx * imageWidth, 0.0, imageWidth))
    centerY = float(np.clip(cy * imageHeight, 0.0, imageHeight))
    x = float(np.clip(centerX - widthPx / 2.0, 0.0, imageWidth - widthPx))
    y = float(np.clip(centerY - heightPx / 2.0, 0.0, imageHeight - heightPx))
    return BBoxXYWH(xPx=x, yPx=y, widthPx=widthPx, heightPx=heightPx)


def _heatmapCertainty(heatmaps: list[Any], torch: ModuleType) -> float:
    if len(heatmaps) != 2:
        raise ModelError("HiT corner head did not expose both confidence heatmaps")
    if not _heatmapsAreFinite(heatmaps, torch):
        raise ModelError("HiT corner head returned a non-finite confidence heatmap")
    certainties = []
    for heatmap in heatmaps:
        logits = heatmap.float().reshape(-1)
        probabilities = torch.softmax(logits, dim=0)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum()
        normalized = 1.0 - float(entropy.item()) / math.log(max(2, probabilities.numel()))
        certainties.append(float(np.clip(normalized, 0.0, 1.0)))
    concentration = float(np.mean(certainties))
    return float(np.clip(1.0 - math.exp(-5.0 * concentration), 0.0, 1.0))


def _heatmapsAreFinite(heatmaps: list[Any], torch: ModuleType) -> bool:
    return len(heatmaps) == 2 and all(bool(torch.isfinite(item).all()) for item in heatmaps)


def _requireRgb(rgb: NDArray[np.uint8]) -> None:
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8:
        raise ProtocolError("HiT input must be a uint8 RGB NumPy array")
    if rgb.ndim != 3 or rgb.shape[2] != 3 or 0 in rgb.shape[:2]:
        raise ProtocolError(f"HiT input must have shape [H, W, 3], actual={rgb.shape}")


__all__ = ["PyTorchHiTSession"]
