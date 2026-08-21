"""Production PyTorch adapter for the official HiT-Small implementation."""

from __future__ import annotations

import gc
import math
import os
import sys
from collections.abc import Sequence
from contextlib import nullcontext
from pathlib import Path
from time import perf_counter_ns
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
        self._profileEnabled = _envFlag("INSTARGET_PROFILE")
        self._benchmark = _envFlag("INSTARGET_CUDNN_BENCHMARK")
        self._channelsLast = _envFlag("INSTARGET_CHANNELS_LAST")
        self._reuseBuffers = _envFlag("INSTARGET_REUSE_BUFFERS")
        self._pinnedMemory = _envFlag("INSTARGET_PINNED_MEMORY")
        self._nonBlocking = _envFlag("INSTARGET_NON_BLOCKING")
        if self._nonBlocking and not self._pinnedMemory:
            raise ModelError("INSTARGET_NON_BLOCKING requires INSTARGET_PINNED_MEMORY")
        self._cpuBuffers: dict[tuple[int, ...], Any] = {}
        self._pinnedBuffers: dict[tuple[int, ...], Any] = {}
        self._gpuBuffers: dict[tuple[int, ...], Any] = {}
        self._lastProfile: dict[str, int | float | bool | str] = {}
        self._fp16FallbackCount = 0
        self._oomCount = 0
        self._deviceBatchLimit: int | None = None
        self._torch = _importTorch()
        if not self._torch.cuda.is_available():
            raise ModelError("official HiT-Small requires a CUDA-capable PyTorch runtime")

        self._hitRoot = _resolveHitRoot(hitRoot)
        self._weights = Path(config.weights).expanduser().resolve()
        if not self._weights.is_file():
            raise ModelError(f"HiT checkpoint does not exist: {self._weights}")
        self._device = self._torch.device("cuda")
        self._stream = self._torch.cuda.default_stream(self._device)
        self._previousCudnnBenchmark = bool(self._torch.backends.cudnn.benchmark)
        self._torch.backends.cudnn.benchmark = self._benchmark
        self._model = self._loadModel()
        self._mean = self._torch.tensor(
            [0.485, 0.456, 0.406], device=self._device, dtype=self._torch.float32
        ).view(1, 3, 1, 1)
        self._std = self._torch.tensor(
            [0.229, 0.224, 0.225], device=self._device, dtype=self._torch.float32
        ).view(1, 3, 1, 1)

    @property
    def lastProfile(self) -> dict[str, int | float | bool | str]:
        return dict(self._lastProfile)

    @property
    def fp16FallbackCount(self) -> int:
        return self._fp16FallbackCount

    def encodeTemplate(self, rgb: NDArray[np.uint8], bbox: BBoxXYWH) -> object:
        self._requireOpen()
        _requireRgb(rgb)
        patch = _sampleTarget(rgb, bbox, _TEMPLATE_FACTOR, _TEMPLATE_SIZE)
        return self._preprocess(patch)

    def encodeTemplateDevice(self, deviceRgb: Any, bbox: BBoxXYWH) -> object:
        self._requireOpen()
        with self._torch.cuda.stream(self._stream):
            tensor = self._validateDeviceRgb(deviceRgb)
            template = _sampleTargetDevice(
                tensor,
                bbox,
                _TEMPLATE_FACTOR,
                _TEMPLATE_SIZE,
                self._torch,
            ).unsqueeze(0)
        self._stream.synchronize()
        return template

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
                f"HiT template feature must have shape [1, C, H, W], actual={tuple(template.shape)}"
            )

        batchSize = len(images)
        preprocessStartedNs = perf_counter_ns() if self._profileEnabled else None
        resized = self._resizeBatch(images)
        preprocessNs = (
            perf_counter_ns() - preprocessStartedNs if preprocessStartedNs is not None else 0
        )
        search, hostToDeviceNs = self._preprocessBatch(resized)
        batchTemplate = template.expand(batchSize, -1, -1, -1)
        if self._channelsLast:
            search = search.contiguous(memory_format=self._torch.channels_last)
            batchTemplate = batchTemplate.contiguous(memory_format=self._torch.channels_last)
        try:
            output, cudaForwardNs = self._forward(
                search, batchTemplate, useFp16=self._precision == "fp16"
            )
        except self._torch.cuda.OutOfMemoryError as error:
            self._oomCount += 1
            self._lastProfile = self._profileSnapshot(
                preprocessNs=preprocessNs,
                hostToDeviceNs=hostToDeviceNs,
                cudaForwardNs=0,
                batchSize=batchSize,
                fp16Fallback=False,
            )
            raise ModelError(f"HiT CUDA out of memory for batch size {batchSize}") from error
        outputsInvalid = not _outputsAreFinite(output, self._torch)
        if self._precision == "fp16" and outputsInvalid:
            self._fp16FallbackCount += 1
            try:
                output, fallbackForwardNs = self._forward(
                    search,
                    batchTemplate,
                    useFp16=False,
                )
            except self._torch.cuda.OutOfMemoryError as error:
                self._oomCount += 1
                self._lastProfile = self._profileSnapshot(
                    preprocessNs=preprocessNs,
                    hostToDeviceNs=hostToDeviceNs,
                    cudaForwardNs=cudaForwardNs,
                    batchSize=batchSize,
                    fp16Fallback=True,
                )
                raise ModelError(
                    f"HiT FP32 fallback ran out of memory for batch size {batchSize}"
                ) from error
            cudaForwardNs += fallbackForwardNs
            if not _outputsAreFinite(output, self._torch):
                raise ModelError("HiT FP32 fallback returned non-finite outputs")
        elif outputsInvalid:
            raise ModelError("HiT FP32 inference returned non-finite outputs")
        self._lastProfile = self._profileSnapshot(
            preprocessNs=preprocessNs,
            hostToDeviceNs=hostToDeviceNs,
            cudaForwardNs=cudaForwardNs,
            batchSize=batchSize,
            fp16Fallback=bool(self._precision == "fp16" and outputsInvalid),
        )
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
                modelScore=(float(presenceProbability * qualityProbability)),
                appearanceScore=(float(presenceProbability * qualityProbability)),
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

    def inferDeviceBatch(
        self,
        deviceRgbs: Sequence[Any],
        imageSizes: Sequence[tuple[int, int]],
        templateFeatures: Sequence[object],
    ) -> tuple[HiTPrediction, ...]:
        """Infer directly from normalized GPU tensors produced by GPU Geometry."""
        self._requireOpen()
        if not deviceRgbs:
            return ()
        if len(deviceRgbs) != len(imageSizes):
            raise ProtocolError("device RGBs and image sizes must have equal lengths")
        if not templateFeatures:
            raise ProtocolError("HiT device inference requires at least one template")
        template = templateFeatures[-1]
        if not self._torch.is_tensor(template) or template.ndim != 4 or template.shape[0] != 1:
            raise ProtocolError("HiT device template must have shape [1,C,H,W]")
        tensors = tuple(self._validateDeviceRgb(item) for item in deviceRgbs)
        batchSize = len(tensors)
        if self._deviceBatchLimit is not None and batchSize > self._deviceBatchLimit:
            return self._inferDeviceBatchChunked(
                tensors,
                imageSizes,
                template,
                chunkSize=self._deviceBatchLimit,
                recoveredFromOom=False,
            )
        try:
            return self._inferDeviceBatchOnce(tensors, imageSizes, template)
        except self._torch.cuda.OutOfMemoryError:
            pass

        self._oomCount += 1
        self._recoverCudaOom()
        if batchSize == 1:
            self._lastProfile = self._profileSnapshot(
                preprocessNs=0,
                hostToDeviceNs=0,
                cudaForwardNs=0,
                batchSize=1,
                fp16Fallback=False,
            )
            raise ModelError("HiT CUDA out of memory for one device view") from None

        # A recovery frame may request up to twelve views.  The full batch has
        # the best throughput, but its activation peak can exceed a 24 GB card
        # even though every view fits independently.  After one OOM, keep future
        # batches bounded as well instead of repeatedly stressing the allocator.
        self._deviceBatchLimit = 1
        return self._inferDeviceBatchChunked(
            tensors,
            imageSizes,
            template,
            chunkSize=1,
            recoveredFromOom=True,
        )

    def _inferDeviceBatchChunked(
        self,
        tensors: Sequence[Any],
        imageSizes: Sequence[tuple[int, int]],
        template: Any,
        *,
        chunkSize: int,
        recoveredFromOom: bool,
    ) -> tuple[HiTPrediction, ...]:
        batchSize = len(tensors)
        predictions: list[HiTPrediction] = []
        totalForwardNs = 0
        usedFp16Fallback = False
        for start in range(0, batchSize, chunkSize):
            chunkTensors = tensors[start : start + chunkSize]
            chunkSizes = imageSizes[start : start + chunkSize]
            try:
                chunk = self._inferDeviceBatchOnce(chunkTensors, chunkSizes, template)
            except self._torch.cuda.OutOfMemoryError:
                self._oomCount += 1
                self._recoverCudaOom()
                self._lastProfile = self._profileSnapshot(
                    preprocessNs=0,
                    hostToDeviceNs=0,
                    cudaForwardNs=totalForwardNs,
                    batchSize=batchSize,
                    fp16Fallback=usedFp16Fallback,
                )
                self._lastProfile.update(
                    {
                        "deviceInput": True,
                        "oomRecovered": False,
                        "fallbackChunkSize": chunkSize,
                    }
                )
                raise ModelError(
                    f"HiT CUDA out of memory for device chunk size {chunkSize}"
                ) from None
            predictions.extend(chunk)
            totalForwardNs += int(self._lastProfile.get("cudaForward", 0))
            usedFp16Fallback = usedFp16Fallback or bool(
                self._lastProfile.get("fp16Fallback", False)
            )
        self._lastProfile = self._profileSnapshot(
            preprocessNs=0,
            hostToDeviceNs=0,
            cudaForwardNs=totalForwardNs,
            batchSize=batchSize,
            fp16Fallback=usedFp16Fallback,
        )
        self._lastProfile.update(
            {
                "deviceInput": True,
                "oomRecovered": recoveredFromOom,
                "fallbackChunkSize": chunkSize,
            }
        )
        return tuple(predictions)

    def _inferDeviceBatchOnce(
        self,
        tensors: Sequence[Any],
        imageSizes: Sequence[tuple[int, int]],
        template: Any,
    ) -> tuple[HiTPrediction, ...]:
        batchSize = len(tensors)
        search = self._torch.stack(tensors, dim=0)
        batchTemplate = template.expand(batchSize, -1, -1, -1)
        if self._channelsLast:
            search = search.contiguous(memory_format=self._torch.channels_last)
            batchTemplate = batchTemplate.contiguous(memory_format=self._torch.channels_last)
        output, cudaForwardNs = self._forward(
            search, batchTemplate, useFp16=self._precision == "fp16"
        )
        outputsInvalid = not _outputsAreFinite(output, self._torch)
        if self._precision == "fp16" and outputsInvalid:
            self._fp16FallbackCount += 1
            output, fallbackForwardNs = self._forward(search, batchTemplate, useFp16=False)
            cudaForwardNs += fallbackForwardNs
            if not _outputsAreFinite(output, self._torch):
                raise ModelError("HiT FP32 fallback returned non-finite outputs")
        elif outputsInvalid:
            raise ModelError("HiT device inference returned non-finite outputs")
        self._lastProfile = self._profileSnapshot(
            preprocessNs=0,
            hostToDeviceNs=0,
            cudaForwardNs=cudaForwardNs,
            batchSize=batchSize,
            fp16Fallback=bool(self._precision == "fp16" and outputsInvalid),
        )
        self._lastProfile["deviceInput"] = True
        boxes = output["predBoxes"]
        if boxes.numel() == 0 or boxes.shape[0] != batchSize:
            raise ModelError("HiT returned an invalid device prediction batch")
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
                    imageWidth=size[0],
                    imageHeight=size[1],
                ),
                modelScore=float(presenceProbability * qualityProbability),
                appearanceScore=float(presenceProbability * qualityProbability),
                presenceLogit=presenceLogit,
                qualityLogit=qualityLogit,
                presenceProbability=presenceProbability,
                qualityProbability=qualityProbability,
                predictedIoU=qualityProbability,
                cornerScore=certainty,
            )
            for (
                boxRow,
                certainty,
                presenceLogit,
                qualityLogit,
                presenceProbability,
                qualityProbability,
                size,
            ) in zip(
                boxRows,
                certainties,
                presenceLogits,
                qualityLogits,
                presenceProbabilities,
                qualityProbabilities,
                imageSizes,
                strict=True,
            )
        )

    def _recoverCudaOom(self) -> None:
        """Release failed-batch temporaries without reusing the OOM traceback."""
        gc.collect()
        try:
            self._torch.cuda.empty_cache()
        except RuntimeError:
            # If CUDA reports a secondary asynchronous failure, the caller will
            # surface the original model error and close the runtime safely.
            pass

    def close(self) -> None:
        if self._closed:
            return
        try:
            # GPU Geometry and HiT inference enqueue asynchronous work.  Synchronize
            # before dropping the model and reusable buffers so the CUDA caching
            # allocator does not need to record events while the interpreter is
            # tearing down the context.  On Windows this otherwise makes a
            # subsequent evaluation process intermittently abort in CUDAEvent::record.
            self._torch.cuda.synchronize(self._device)
        except RuntimeError:
            # Preserve the original inference failure when a prior asynchronous
            # CUDA error has already poisoned the context.
            pass
        self._model = None
        self._cpuBuffers.clear()
        self._pinnedBuffers.clear()
        self._gpuBuffers.clear()
        self._stream = None
        self._closed = True
        self._torch.backends.cudnn.benchmark = self._previousCudnnBenchmark
        try:
            self._torch.cuda.empty_cache()
        except RuntimeError:
            # A prior asynchronous CUDA failure can poison cache cleanup. Closing must not mask
            # the original inference exception or abort interpreter shutdown.
            pass

    def _loadModel(self) -> Any:
        runtimeModel = _constructRuntimeModel(
            self._torch,
            weights=self._weights,
            hitRoot=self._hitRoot,
        )
        runtimeModel = runtimeModel.to(self._device).eval()
        if self._channelsLast:
            runtimeModel = runtimeModel.to(memory_format=self._torch.channels_last)
        return runtimeModel

    def _preprocess(self, rgb: NDArray[np.uint8]) -> Any:
        return self._preprocessBatch(np.ascontiguousarray(rgb)[None, ...])[0]

    def _validateDeviceRgb(self, value: Any) -> Any:
        if not self._torch.is_tensor(value):
            raise ProtocolError("HiT device input must be a torch tensor")
        if not _devicesMatch(value.device, self._device, self._torch):
            raise ProtocolError(
                f"HiT device input must be on {self._device}, actual={value.device}"
            )
        if value.ndim != 3 or tuple(value.shape[:1]) != (3,):
            raise ProtocolError(
                f"HiT device input must have shape [3,H,W], actual={tuple(value.shape)}"
            )
        if value.shape[1] != _SEARCH_SIZE or value.shape[2] != _SEARCH_SIZE:
            raise ProtocolError(
                f"HiT device input must be {_SEARCH_SIZE}x{_SEARCH_SIZE}, "
                f"actual={tuple(value.shape)}"
            )
        if value.dtype != self._torch.float32:
            raise ProtocolError(f"HiT device input must be float32, actual={value.dtype}")
        if not bool(self._torch.isfinite(value).all()):
            raise ProtocolError("HiT device input contains non-finite values")
        return value
    def _resizeBatch(self, rgbs: Sequence[NDArray[np.uint8]]) -> NDArray[np.uint8]:
        shape = (len(rgbs), _SEARCH_SIZE, _SEARCH_SIZE, 3)
        if self._reuseBuffers:
            resized = self._cpuBuffers.get(shape)
            if resized is None:
                resized = np.empty(shape, dtype=np.uint8)
                self._cpuBuffers[shape] = resized
            for index, rgb in enumerate(rgbs):
                resized[index] = _resizeRgb(rgb, _SEARCH_SIZE)
            return resized
        return np.stack([_resizeRgb(rgb, _SEARCH_SIZE) for rgb in rgbs])

    def _preprocessBatch(self, rgbs: NDArray[np.uint8]) -> tuple[Any, int]:
        contiguous = np.ascontiguousarray(rgbs)
        if self._pinnedMemory:
            host = self._pinnedBuffers.get(tuple(contiguous.shape))
            if host is None:
                host = self._torch.empty(
                    contiguous.shape,
                    dtype=self._torch.uint8,
                    pin_memory=True,
                )
                self._pinnedBuffers[tuple(contiguous.shape)] = host
            host.copy_(self._torch.from_numpy(contiguous))
        else:
            host = self._torch.from_numpy(contiguous)
        transferStartedNs = perf_counter_ns() if self._profileEnabled else None
        if self._reuseBuffers:
            deviceTensor = self._gpuBuffers.get(tuple(contiguous.shape))
            if deviceTensor is None:
                deviceTensor = self._torch.empty(
                    contiguous.shape, device=self._device, dtype=self._torch.uint8
                )
                self._gpuBuffers[tuple(contiguous.shape)] = deviceTensor
            deviceTensor.copy_(host, non_blocking=self._nonBlocking)
            tensor = deviceTensor
        else:
            tensor = host.to(device=self._device, non_blocking=self._nonBlocking)
        if self._nonBlocking and self._profileEnabled:
            self._torch.cuda.current_stream().synchronize()
        hostToDeviceNs = (
            perf_counter_ns() - transferStartedNs if transferStartedNs is not None else 0
        )
        normalized = tensor.permute(0, 3, 1, 2).to(dtype=self._torch.float32)
        normalized.div_(255.0).sub_(self._mean).div_(self._std)
        return normalized, hostToDeviceNs

    def _forward(self, search: Any, template: Any, *, useFp16: bool) -> tuple[dict[str, Any], int]:
        autocast = (
            self._torch.autocast(device_type="cuda", dtype=self._torch.float16)
            if useFp16
            else nullcontext()
        )
        # Geometry and HiT are separate owners of CUDA tensors. Pin every model
        # call to the shared default stream and make both handoff boundaries
        # explicit; otherwise a thread-local stream change can mismatch cuBLAS or
        # cuDNN handles with tensors created by GPU Geometry.
        self._stream.synchronize()
        startEvent = endEvent = None
        with self._torch.cuda.stream(self._stream):
            if self._profileEnabled:
                startEvent = self._torch.cuda.Event(enable_timing=True)
                endEvent = self._torch.cuda.Event(enable_timing=True)
                startEvent.record(self._stream)
            with self._torch.inference_mode(), autocast:
                output = self._model(template, search)
            if endEvent is not None:
                endEvent.record(self._stream)
        self._stream.synchronize()
        elapsedNs = 0
        if startEvent is not None and endEvent is not None:
            elapsedNs = int(startEvent.elapsed_time(endEvent) * 1_000_000.0)
        boxes = output.get("predBoxes")
        if boxes is None or boxes.numel() == 0:
            raise ModelError("HiT returned no predicted boxes")
        return output, elapsedNs

    def _profileSnapshot(
        self,
        *,
        preprocessNs: int,
        hostToDeviceNs: int,
        cudaForwardNs: int,
        batchSize: int,
        fp16Fallback: bool,
    ) -> dict[str, int | float | bool | str]:
        snapshot: dict[str, int | float | bool | str] = {
            "preprocess": preprocessNs,
            "hostToDevice": hostToDeviceNs,
            "cudaForward": cudaForwardNs,
            "batchSize": batchSize,
            "fp16Fallback": fp16Fallback,
            "fp16FallbackCount": self._fp16FallbackCount,
            "oomCount": self._oomCount,
            "precision": self._precision,
            "cudnnBenchmark": self._benchmark,
            "channelsLast": self._channelsLast,
            "reuseBuffers": self._reuseBuffers,
            "pinnedMemory": self._pinnedMemory,
            "nonBlocking": self._nonBlocking,
        }
        if self._profileEnabled:
            snapshot["maxMemoryAllocatedBytes"] = int(
                self._torch.cuda.max_memory_allocated(self._device)
            )
            snapshot["maxMemoryReservedBytes"] = int(
                self._torch.cuda.max_memory_reserved(self._device)
            )
        return snapshot

    def _requireOpen(self) -> None:
        if self._closed:
            raise ProtocolError("HiT session is closed")


def _devicesMatch(actual: Any, expected: Any, torchModule: Any) -> bool:
    """Compare devices while treating ``cuda`` as the current CUDA device."""
    if actual.type != expected.type:
        return False
    if actual.type != "cuda":
        return actual == expected
    currentIndex = int(torchModule.cuda.current_device())
    actualIndex = currentIndex if actual.index is None else int(actual.index)
    expectedIndex = currentIndex if expected.index is None else int(expected.index)
    return actualIndex == expectedIndex


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


def _sampleTargetDevice(
    rgb: Any,
    bbox: BBoxXYWH,
    factor: float,
    outputSize: int,
    torch: ModuleType,
) -> Any:
    """Crop a normalized [C,H,W] tensor without moving image data to the host."""
    import torch.nn.functional as functional

    height, width = int(rgb.shape[1]), int(rgb.shape[2])
    cropSize = max(1, int(math.ceil(math.sqrt(bbox.widthPx * bbox.heightPx) * factor)))
    x1 = int(round(bbox.xPx + 0.5 * bbox.widthPx - 0.5 * cropSize))
    y1 = int(round(bbox.yPx + 0.5 * bbox.heightPx - 0.5 * cropSize))
    x2, y2 = x1 + cropSize, y1 + cropSize
    left, right = max(0, -x1), max(x2 - width, 0)
    top, bottom = max(0, -y1), max(y2 - height, 0)
    clipped = rgb[:, max(0, y1) : min(height, y2), max(0, x1) : min(width, x2)]
    if clipped.numel() == 0:
        raise ProtocolError("HiT device template crop has no pixels")
    if left or right or top or bottom:
        clipped = functional.pad(clipped.unsqueeze(0), (left, right, top, bottom)).squeeze(0)
    return functional.interpolate(
        clipped.unsqueeze(0),
        size=(outputSize, outputSize),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0).to(dtype=torch.float32)


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


def _envFlag(name: str) -> bool:
    value = os.environ.get(name, "0").strip().lower()
    if value not in {"0", "1", "false", "true"}:
        raise ModelError(f"{name} must be 0/1/false/true")
    return value in {"1", "true"}


__all__ = ["PyTorchHiTSession", "validateHiTCheckpoint"]
