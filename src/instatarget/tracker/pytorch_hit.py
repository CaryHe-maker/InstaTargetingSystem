"""Adapter for the official kangben258/HiT PyTorch implementation."""

from __future__ import annotations

import importlib
import math
import sys
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from instatarget.core.config import ModelConfig
from instatarget.core.errors import ModelError, ProtocolError
from instatarget.core.types import BBoxXYWH
from instatarget.tracker.hit_backend import HiTPrediction


@dataclass(frozen=True, slots=True)
class PytorchHiTTemplate:
    tensor: object
    bbox: BBoxXYWH


class PytorchHiTSession:
    """Run one official HiT network while retaining project-owned templates."""

    supportsOnlineTemplates = True

    def __init__(self, config: ModelConfig) -> None:
        if not config.source.is_dir():
            raise ModelError(
                f"official HiT source directory does not exist: {config.source}; "
                "clone https://github.com/kangben258/HiT there"
            )
        if not config.weights.is_file():
            raise ModelError(f"HiT weights file does not exist: {config.weights}")
        self._torch = _importTorch()
        self._device = self._resolveDevice(config.device)
        self._precision = config.precision
        self._closed = False
        self._configureOfficialSource(config.source, config.variant)
        self._network = self._buildNetwork(config.weights, config.precision)

    def encodeTemplate(self, rgb: NDArray[np.uint8], bbox: BBoxXYWH) -> object:
        self._requireOpen()
        patch, _ = _sampleTarget(rgb, bbox, self._templateFactor, self._templateSize)
        return PytorchHiTTemplate(self._preprocess(patch), bbox)

    def infer(
        self,
        rgb: NDArray[np.uint8],
        templateFeatures: tuple[object, ...],
    ) -> HiTPrediction:
        self._requireOpen()
        templates = [_requireTemplate(value) for value in templateFeatures]
        predicted: list[BBoxXYWH] = []
        queryAgreement: list[float] = []
        autocast = (
            self._torch.autocast(device_type="cuda", dtype=self._torch.float16)
            if self._precision == "fp16"
            else nullcontext()
        )
        with self._torch.inference_mode(), autocast:
            for template in templates:
                searchPatch, resizeFactor = _sampleTarget(
                    rgb, template.bbox, self._searchFactor, self._searchSize
                )
                search = self._preprocess(searchPatch)
                features = self._network(
                    images_list=[search, template.tensor],
                    mode="backbone",
                    first_score=None,
                    threshold=0.0,
                )
                output, _, _ = self._network(xz=features, mode="head")
                queries = output["pred_boxes"].reshape(-1, 4).detach().float().cpu().numpy()
                if queries.size == 0 or not np.isfinite(queries).all():
                    raise ModelError("HiT returned empty or non-finite pred_boxes")
                meanQuery = queries.mean(axis=0)
                predicted.append(
                    _mapBoxBack(meanQuery, template.bbox, self._searchSize, resizeFactor)
                )
                queryAgreement.append(
                    _queryAgreement(queries)
                    * _transitionConfidence(
                        predicted[-1], template.bbox, rgb.shape[1], rgb.shape[0]
                    )
                )
        bbox = _meanBox(predicted)
        templateAgreement = _boxAgreement(predicted, rgb.shape[1], rgb.shape[0])
        score = float(np.clip(np.mean(queryAgreement) * templateAgreement, 0.0, 1.0))
        return HiTPrediction(bbox=bbox, modelScore=score, appearanceScore=score)

    def close(self) -> None:
        if self._closed:
            return
        self._network = None
        if self._device.type == "cuda":
            self._torch.cuda.empty_cache()
        self._closed = True

    def _configureOfficialSource(self, source: Path, variant: str) -> None:
        sourceText = str(source.resolve())
        if sourceText not in sys.path:
            sys.path.insert(0, sourceText)
        try:
            configModule = importlib.import_module("lib.config.HiT.config")
            yamlPath = source / "experiments" / "HiT" / f"{variant}.yaml"
            if not yamlPath.is_file():
                raise ModelError(f"official HiT experiment config does not exist: {yamlPath}")
            configModule.update_config_from_file(str(yamlPath))
            self._cfg = configModule.cfg
            self._templateFactor = float(self._cfg.TEST.TEMPLATE_FACTOR)
            self._templateSize = int(self._cfg.TEST.TEMPLATE_SIZE)
            self._searchFactor = float(self._cfg.TEST.SEARCH_FACTOR)
            self._searchSize = int(self._cfg.TEST.SEARCH_SIZE)
            self._buildHiT = importlib.import_module("lib.models.HiT").build_hit
        except ModelError:
            raise
        except Exception as error:
            raise ModelError(f"cannot import official HiT source from {source}: {error}") from error

    def _buildNetwork(self, weights: Path, precision: str) -> Any:
        try:
            network = self._buildHiT(self._cfg)
            # Official HiT checkpoints include training statistics objects in addition to
            # tensors. PyTorch 2.6 defaults to weights_only=True, so explicitly opt into
            # the legacy loader for this trusted, project-configured checkpoint.
            checkpoint = self._torch.load(
                str(weights), map_location="cpu", weights_only=False
            )
            state = (
                checkpoint.get("net", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            )
            network.load_state_dict(state, strict=True)
            if "LeViT" in str(self._cfg.MODEL.BACKBONE.TYPE):
                utils = importlib.import_module("lib.models.HiT.levit_utils")
                utils.replace_batchnorm(network.backbone.body)
            network = network.to(self._device).eval()
            if precision == "fp16" and self._device.type != "cuda":
                raise ModelError("HiT fp16 inference requires a CUDA device")
            return network
        except ModelError:
            raise
        except Exception as error:
            raise ModelError(f"cannot build HiT network from {weights}: {error}") from error

    def _resolveDevice(self, value: str) -> Any:
        device = self._torch.device(value)
        if device.type == "cuda" and not self._torch.cuda.is_available():
            raise ModelError(f"HiT requested CUDA but CUDA is unavailable: {value}")
        return device

    def _preprocess(self, patch: NDArray[np.uint8]) -> object:
        tensor = self._torch.from_numpy(np.ascontiguousarray(patch)).to(self._device)
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
        mean = tensor.new_tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1)
        std = tensor.new_tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
        return tensor

    def _requireOpen(self) -> None:
        if self._closed:
            raise ProtocolError("HiT session is closed")


def createHiTSession(config: ModelConfig) -> PytorchHiTSession:
    if config.backend != "pytorch":
        raise ModelError(
            f"model.backend={config.backend!r} is not implemented; use pytorch for official HiT"
        )
    return PytorchHiTSession(config)


def _importTorch() -> Any:
    try:
        return importlib.import_module("torch")
    except ModuleNotFoundError as error:
        raise ModelError("PyTorch is required for model.backend=pytorch") from error


def _sampleTarget(
    image: NDArray[np.uint8],
    bbox: BBoxXYWH,
    factor: float,
    outputSize: int,
) -> tuple[NDArray[np.uint8], float]:
    cropSize = max(1, math.ceil(math.sqrt(bbox.widthPx * bbox.heightPx) * factor))
    x0 = round(bbox.xPx + 0.5 * bbox.widthPx - 0.5 * cropSize)
    y0 = round(bbox.yPx + 0.5 * bbox.heightPx - 0.5 * cropSize)
    x1, y1 = x0 + cropSize, y0 + cropSize
    sourceX0, sourceY0 = max(0, x0), max(0, y0)
    sourceX1, sourceY1 = min(image.shape[1], x1), min(image.shape[0], y1)
    crop = np.zeros((cropSize, cropSize, 3), dtype=np.uint8)
    if sourceX1 > sourceX0 and sourceY1 > sourceY0:
        crop[
            sourceY0 - y0 : sourceY1 - y0,
            sourceX0 - x0 : sourceX1 - x0,
        ] = image[sourceY0:sourceY1, sourceX0:sourceX1]
    if cropSize != outputSize:
        crop = _resizeNearest(crop, outputSize)
    return crop, outputSize / cropSize


def _resizeNearest(image: NDArray[np.uint8], outputSize: int) -> NDArray[np.uint8]:
    y = np.minimum(
        (np.arange(outputSize, dtype=np.float64) * image.shape[0] / outputSize).astype(int),
        image.shape[0] - 1,
    )
    x = np.minimum(
        (np.arange(outputSize, dtype=np.float64) * image.shape[1] / outputSize).astype(int),
        image.shape[1] - 1,
    )
    return np.ascontiguousarray(image[np.ix_(y, x)])


def _mapBoxBack(
    normalizedCxCyWh: NDArray[np.floating],
    previous: BBoxXYWH,
    searchSize: int,
    resizeFactor: float,
) -> BBoxXYWH:
    cx, cy, width, height = [float(value) * searchSize / resizeFactor for value in normalizedCxCyWh]
    previousCx = previous.xPx + 0.5 * previous.widthPx
    previousCy = previous.yPx + 0.5 * previous.heightPx
    halfSide = 0.5 * searchSize / resizeFactor
    cx += previousCx - halfSide
    cy += previousCy - halfSide
    return BBoxXYWH(cx - 0.5 * width, cy - 0.5 * height, max(width, 1e-6), max(height, 1e-6))


def _requireTemplate(value: object) -> PytorchHiTTemplate:
    if not isinstance(value, PytorchHiTTemplate):
        raise ModelError(f"unsupported HiT template feature: {type(value).__name__}")
    return value


def _meanBox(boxes: list[BBoxXYWH]) -> BBoxXYWH:
    values = np.asarray(
        [(box.xPx, box.yPx, box.widthPx, box.heightPx) for box in boxes], dtype=np.float64
    )
    mean = values.mean(axis=0)
    return BBoxXYWH(*[float(value) for value in mean])


def _queryAgreement(queries: NDArray[np.floating]) -> float:
    if len(queries) <= 1:
        return 1.0
    spread = float(np.mean(np.std(queries, axis=0)))
    return float(np.exp(-8.0 * spread))


def _boxAgreement(boxes: list[BBoxXYWH], width: int, height: int) -> float:
    if len(boxes) <= 1:
        return 1.0
    values = np.asarray(
        [
            (
                (box.xPx + 0.5 * box.widthPx) / max(width, 1),
                (box.yPx + 0.5 * box.heightPx) / max(height, 1),
                box.widthPx / max(width, 1),
                box.heightPx / max(height, 1),
            )
            for box in boxes
        ],
        dtype=np.float64,
    )
    return float(np.exp(-8.0 * float(np.mean(np.std(values, axis=0)))))


def _transitionConfidence(
    current: BBoxXYWH,
    previous: BBoxXYWH,
    imageWidth: int,
    imageHeight: int,
) -> float:
    dx = (current.xPx + 0.5 * current.widthPx) - (previous.xPx + 0.5 * previous.widthPx)
    dy = (current.yPx + 0.5 * current.heightPx) - (previous.yPx + 0.5 * previous.heightPx)
    displacement = math.hypot(dx / max(imageWidth, 1), dy / max(imageHeight, 1))
    scale = abs(math.log(max(current.widthPx * current.heightPx, 1e-6) / max(
        previous.widthPx * previous.heightPx, 1e-6
    )))
    return float(np.exp(-2.0 * displacement - 0.35 * scale))


__all__ = ["PytorchHiTSession", "PytorchHiTTemplate", "createHiTSession"]
