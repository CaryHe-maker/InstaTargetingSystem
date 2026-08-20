"""Production PyTorch adapter for the official HiT-Small implementation."""

from __future__ import annotations

import math
import os
import sys
from collections.abc import Sequence
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
        self._model = self._loadModel()
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
            raise ProtocolError("HiT inference requires at least one template")
        template = templateFeatures[-1]
        if not self._torch.is_tensor(template):
            raise ProtocolError("HiT template feature must be a torch tensor")
        if template.ndim != 4 or template.shape[0] != 1:
            raise ProtocolError(
                "HiT template feature must have shape [1, C, H, W], "
                f"actual={tuple(template.shape)}"
            )

        batchSize = len(images)
        resized = np.stack([_resizeRgb(rgb, _SEARCH_SIZE) for rgb in images])
        search = self._preprocessBatch(resized)
        batchTemplate = template.expand(batchSize, -1, -1, -1)
        output = self._forward(search, batchTemplate, useFp16=self._precision == "fp16")
        fp16Invalid = not _outputsAreFinite(output, self._torch)
        if self._precision == "fp16" and fp16Invalid:
            output = self._forward(search, batchTemplate, useFp16=False)
        boxes = output["predBoxes"]
        if boxes.numel() == 0 or boxes.shape[0] != batchSize:
            raise ModelError(
                "HiT returned an invalid prediction batch: "
                f"expected={batchSize}, actual={tuple(boxes.shape)}"
            )
        boxRows = boxes.reshape(batchSize, -1, 4).float().mean(dim=1).tolist()
        certainties = _heatmapCertainties(
            (output["cornerHeatmapTl"], output["cornerHeatmapBr"]),
            self._torch,
            batchSize,
        )
        presenceLogits = output["presenceLogit"].float().reshape(-1).tolist()
        qualityLogits = output["qualityLogit"].float().reshape(-1).tolist()
        presenceProbabilities = output["presenceProbability"].float().reshape(-1).tolist()
        qualityProbabilities = output["qualityProbability"].float().reshape(-1).tolist()
        return tuple(
            HiTPrediction(
                bbox=_normalizedBoxToPixels(
                    *boxRow,
                    imageWidth=rgb.shape[1],
                    imageHeight=rgb.shape[0],
                ),
                modelScore=(
                    float(presenceProbability * qualityProbability)
                ),
                appearanceScore=(
                    float(presenceProbability * qualityProbability)
                ),
                presenceLogit=presenceLogit,
                qualityLogit=qualityLogit,
                presenceProbability=presenceProbability,
                qualityProbability=qualityProbability,
                predictedIoU=qualityProbability,
                cornerScore=certainty,
            )
            for (
                rgb,
                boxRow,
                certainty,
                presenceLogit,
                qualityLogit,
                presenceProbability,
                qualityProbability,
            ) in zip(
                images,
                boxRows,
                certainties,
                presenceLogits,
                qualityLogits,
                presenceProbabilities,
                qualityProbabilities,
                strict=True,
            )
        )

    def close(self) -> None:
        if self._closed:
            return
        self._model = None
        self._closed = True
        self._torch.cuda.empty_cache()

    def _loadModel(self) -> Any:
        runtimeModel = _constructRuntimeModel(
            self._torch,
            weights=self._weights,
            hitRoot=self._hitRoot,
        )
        return runtimeModel.to(self._device).eval()

    def _preprocess(self, rgb: NDArray[np.uint8]) -> Any:
        return self._preprocessBatch(np.ascontiguousarray(rgb)[None, ...])

    def _preprocessBatch(self, rgbs: NDArray[np.uint8]) -> Any:
        tensor = self._torch.from_numpy(np.ascontiguousarray(rgbs)).to(
            device=self._device, dtype=self._torch.float32
        )
        tensor = tensor.permute(0, 3, 1, 2).div_(255.0)
        return (tensor - self._mean) / self._std

    def _forward(self, search: Any, template: Any, *, useFp16: bool) -> dict[str, Any]:
        autocast = (
            self._torch.autocast(device_type="cuda", dtype=self._torch.float16)
            if useFp16
            else nullcontext()
        )
        with self._torch.inference_mode(), autocast:
            output = self._model(template, search)
        boxes = output.get("predBoxes")
        if boxes is None or boxes.numel() == 0:
            raise ModelError("HiT returned no predicted boxes")
        return output

    def _requireOpen(self) -> None:
        if self._closed:
            raise ProtocolError("HiT session is closed")


def _importTorch() -> ModuleType:
    try:
        import torch
    except ImportError as error:
        raise ModelError("PyTorch HiT requires torch and torchvision") from error
    return torch


def validateHiTCheckpoint(
    weights: str | Path,
    *,
    hitRoot: str | Path | None = None,
) -> int:
    """Strictly load the production checkpoint into the bundled HiT model on CPU."""
    torch = _importTorch()
    model = _constructRuntimeModel(
        torch,
        weights=Path(weights).expanduser().resolve(),
        hitRoot=_resolveHitRoot(hitRoot),
        cpuOnly=True,
    )
    return sum(parameter.numel() for parameter in model.parameters())


def _constructRuntimeModel(
    torch: ModuleType,
    *,
    weights: Path,
    hitRoot: Path,
    cpuOnly: bool = False,
) -> Any:
    _activateVendorTree(hitRoot)
    try:
        import lib.models.HiT.backbone as backboneModule
        from lib.config.HiT.config import cfg, update_config_from_file
        from lib.models.HiT import build_hit
        from lib.models.HiT.levit_utils import replace_batchnorm

        from instatarget.training.model import HiTTrainingModel
    except Exception as error:
        raise ModelError(f"cannot import bundled HiT runtime from {hitRoot}: {error}") from error

    yamlPath = hitRoot / "configs" / "HiT_Small.yaml"
    if not yamlPath.is_file():
        raise ModelError(f"HiT-Small config does not exist: {yamlPath}")
    update_config_from_file(str(yamlPath))
    backboneModule.is_main_process = lambda: False
    try:
        baseModel = _buildBaseModel(torch, build_hit, cfg, cpuOnly=cpuOnly)
        checkpoint = _loadCheckpoint(torch, weights)
        state = checkpoint.get("model") if isinstance(checkpoint, dict) else None
        if not isinstance(state, dict):
            raise ModelError("Stage 3 HiT checkpoint has no 'model' state")
        runtimeModel = HiTTrainingModel(baseModel)
        runtimeModel.load_state_dict(state, strict=True)
        replace_batchnorm(baseModel.backbone.body)
        return runtimeModel
    except ModelError:
        raise
    except Exception as error:
        raise ModelError(f"cannot construct HiT-Small from {weights}: {error}") from error


def _buildBaseModel(torch: ModuleType, buildHit: Any, cfg: Any, *, cpuOnly: bool) -> Any:
    if not cpuOnly:
        return buildHit(cfg)
    originalCuda = torch.Tensor.cuda
    setattr(torch.Tensor, "cuda", lambda tensor, *args, **kwargs: tensor)
    try:
        return buildHit(cfg)
    finally:
        setattr(torch.Tensor, "cuda", originalCuda)


def _resolveHitRoot(value: str | Path | None = None) -> Path:
    candidates = []
    if value is not None:
        candidates.append(Path(value))
    environment = os.environ.get("HIT_ROOT")
    if environment:
        candidates.append(Path(environment))
    candidates.append(Path(__file__).resolve().parents[1] / "vendor" / "hit")
    for candidate in candidates:
        root = candidate.expanduser().resolve()
        if (root / "lib" / "models" / "HiT").is_dir():
            return root
    checked = ", ".join(str(item.expanduser()) for item in candidates)
    raise ModelError(f"bundled HiT runtime was not found; checked: {checked}")


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


def _heatmapCertainties(
    heatmaps: Sequence[Any], torch: ModuleType, batchSize: int
) -> tuple[float, ...]:
    if len(heatmaps) != 2:
        raise ModelError("HiT corner head did not expose both confidence heatmaps")
    if not all(bool(torch.isfinite(item).all()) for item in heatmaps):
        raise ModelError("HiT corner head returned a non-finite confidence heatmap")
    concentrations = []
    for heatmap in heatmaps:
        if heatmap.ndim == 0 or heatmap.shape[0] != batchSize:
            raise ModelError(
                "HiT corner head returned an invalid heatmap batch: "
                f"expected={batchSize}, actual={tuple(heatmap.shape)}"
            )
        logits = heatmap.float().reshape(batchSize, -1)
        probabilities = torch.softmax(logits, dim=1)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)
        normalized = 1.0 - entropy / math.log(max(2, probabilities.shape[1]))
        concentrations.append(normalized.clamp(0.0, 1.0))
    meanConcentration = torch.stack(concentrations).mean(dim=0)
    certainties = (1.0 - torch.exp(-5.0 * meanConcentration)).clamp(0.0, 1.0)
    return tuple(float(value) for value in certainties.tolist())


def _outputsAreFinite(output: dict[str, Any], torch: ModuleType) -> bool:
    required = (
        "predBoxes",
        "cornerHeatmapTl",
        "cornerHeatmapBr",
        "presenceLogit",
        "qualityLogit",
        "presenceProbability",
        "qualityProbability",
    )
    return all(name in output and bool(torch.isfinite(output[name]).all()) for name in required)


def _requireRgb(rgb: NDArray[np.uint8]) -> None:
    if not isinstance(rgb, np.ndarray) or rgb.dtype != np.uint8:
        raise ProtocolError("HiT input must be a uint8 RGB NumPy array")
    if rgb.ndim != 3 or rgb.shape[2] != 3 or 0 in rgb.shape[:2]:
        raise ProtocolError(f"HiT input must have shape [H, W, 3], actual={rgb.shape}")


__all__ = ["PyTorchHiTSession", "validateHiTCheckpoint"]
