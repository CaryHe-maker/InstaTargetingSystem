"""Executable staged HiT-Small training backend."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from collections.abc import Mapping, Sequence
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.optim import AdamW
from torch.utils.data import DataLoader

from instatarget.core.config import TrainingConfig, loadTrainingConfig
from instatarget.eval.calibration_metrics import (
    brierScore,
    expectedCalibrationError,
    prAuc,
    rocAuc,
)
from instatarget.training.augment import LocalViewAugmenter
from instatarget.training.dataset import ManifestPairDataset, TrainingPair
from instatarget.training.losses import alignedBoxIou, computeTrainingLoss
from instatarget.training.model import (
    HiTTrainingModel,
    buildTrainingModel,
    configureTrainingStage,
    keepFrozenNormalizationInEval,
)

_MEAN = (0.485, 0.456, 0.406)
_STD = (0.229, 0.224, 0.225)


def train(config: TrainingConfig) -> Path:
    if not torch.cuda.is_available():
        raise RuntimeError("HiT-Small training requires a CUDA-capable PyTorch runtime")
    _seedEverything(config.optimization.seed)
    device = torch.device("cuda")
    trainDataset = ManifestPairDataset(
        config.data,
        config.data.trainSplit,
        seed=config.optimization.seed,
        augmenter=LocalViewAugmenter(),
    )
    validationDataset = ManifestPairDataset(
        config.data,
        config.data.validationSplit,
        seed=config.optimization.seed + 1,
    )
    trainLoader = _dataLoader(trainDataset, config, shuffle=True)
    validationLoader = _dataLoader(validationDataset, config, shuffle=False)

    model = buildTrainingModel(config).to(device)
    parameterGroups, reports = configureTrainingStage(model, config)
    optimizer = AdamW(
        parameterGroups,
        weight_decay=config.optimization.weightDecay,
    )
    totalSteps = _totalOptimizerSteps(config, len(trainLoader))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=_learningRateSchedule(config, totalSteps)
    )
    scaler = torch.amp.GradScaler("cuda", enabled=config.optimization.precision == "fp16")
    checkpointDir = config.runtime.checkpointDir
    checkpointDir.mkdir(parents=True, exist_ok=True)
    _writeRunMetadata(config, reports, checkpointDir)

    epoch = 0
    globalStep = 0
    bestValidation = math.inf
    staleValidations = 0
    if config.runtime.resume is not None:
        epoch, globalStep, bestValidation, staleValidations = _resume(
            config.runtime.resume, model, optimizer, scheduler, scaler, config
        )

    optimizer.zero_grad(set_to_none=True)
    while epoch < config.optimization.epochs and globalStep < totalSteps:
        trainDataset.setEpoch(epoch)
        model.train()
        keepFrozenNormalizationInEval(model)
        for batchIndex, batch in enumerate(trainLoader):
            tensors = _moveBatch(batch, device)
            with _autocast(config):
                output = model(tensors["template"], tensors["search"])
                losses = computeTrainingLoss(output, tensors, config.loss)
                scaledLoss = losses["total"] / config.optimization.gradientAccumulation
            scaler.scale(scaledLoss).backward()
            isBoundary = (batchIndex + 1) % config.optimization.gradientAccumulation == 0
            isLastBatch = batchIndex + 1 == len(trainLoader)
            if not (isBoundary or isLastBatch):
                continue
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                (parameter for parameter in model.parameters() if parameter.requires_grad),
                config.optimization.gradientClipNorm,
            )
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            globalStep += 1
            if globalStep % config.runtime.logEverySteps == 0:
                _logLoss(globalStep, epoch, losses, optimizer)
            if globalStep % config.runtime.validateEverySteps == 0:
                metrics = validate(model, validationLoader, config, device)
                print(json.dumps({"step": globalStep, "validation": metrics}, sort_keys=True))
                improved = metrics["loss"] < bestValidation
                if improved:
                    bestValidation = metrics["loss"]
                    staleValidations = 0
                else:
                    staleValidations += 1
                latest = checkpointDir / "latest.pth"
                _saveCheckpoint(
                    latest,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    config,
                    epoch,
                    globalStep,
                    bestValidation,
                    staleValidations,
                    metrics,
                )
                if improved:
                    _saveCheckpoint(
                        checkpointDir / "best.pth",
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        config,
                        epoch,
                        globalStep,
                        bestValidation,
                        staleValidations,
                        metrics,
                    )
                if staleValidations >= config.runtime.earlyStoppingPatience:
                    return checkpointDir / "best.pth"
                model.train()
                keepFrozenNormalizationInEval(model)
            if globalStep >= totalSteps:
                break
        epoch += 1

    finalMetrics = validate(model, validationLoader, config, device)
    finalPath = checkpointDir / "final.pth"
    _saveCheckpoint(
        finalPath,
        model,
        optimizer,
        scheduler,
        scaler,
        config,
        epoch,
        globalStep,
        min(bestValidation, finalMetrics["loss"]),
        staleValidations,
        finalMetrics,
    )
    return finalPath


@torch.inference_mode()
def validate(
    model: HiTTrainingModel,
    loader: DataLoader[dict[str, Any]],
    config: TrainingConfig,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    probabilities: list[float] = []
    labels: list[float] = []
    positiveIous: list[float] = []
    batches = 0
    for batch in loader:
        tensors = _moveBatch(batch, device)
        with _autocast(config):
            output = model(tensors["template"], tensors["search"])
            losses = computeTrainingLoss(output, tensors, config.loss)
        batches += 1
        for name in ("total", "presence", "bboxL1", "bboxGiou", "quality"):
            totals[name] = totals.get(name, 0.0) + float(losses[name].detach().cpu())
        probability = (
            output["presenceProbability"] * output["qualityProbability"]
        ).float()
        probabilities.extend(probability.cpu().tolist())
        labels.extend(tensors["present"].float().cpu().tolist())
        positive = tensors["present"].bool()
        if positive.any():
            positiveIous.extend(
                alignedBoxIou(
                    output["predBoxes"].reshape(-1, 4)[positive],
                    tensors["boxes"][positive],
                )
                .float()
                .cpu()
                .tolist()
            )
    if batches == 0:
        raise RuntimeError("validation loader is empty")
    return {
        "loss": totals["total"] / batches,
        "presenceLoss": totals["presence"] / batches,
        "bboxL1": totals["bboxL1"] / batches,
        "bboxGiou": totals["bboxGiou"] / batches,
        "qualityLoss": totals["quality"] / batches,
        "meanPositiveIoU": float(np.mean(positiveIous)) if positiveIous else 0.0,
        "brier": brierScore(probabilities, labels),
        "ece": expectedCalibrationError(probabilities, labels),
        "prAuc": prAuc(probabilities, labels),
        "rocAuc": rocAuc(probabilities, labels),
    }


def collateTrainingPairs(items: Sequence[TrainingPair]) -> dict[str, Any]:
    return {
        "template": _normalizeImages(tuple(item.templateRgb for item in items)),
        "search": _normalizeImages(tuple(item.searchRgb for item in items)),
        "boxes": torch.from_numpy(np.stack([item.targetBoxCxCyWh for item in items])),
        "present": torch.tensor([item.present for item in items], dtype=torch.bool),
        "labelQuality": torch.tensor(
            [item.labelQuality for item in items], dtype=torch.float32
        ),
        "sequenceId": tuple(item.sequenceId for item in items),
        "difficultType": tuple(item.difficultType for item in items),
    }


def _normalizeImages(images: Sequence[np.ndarray]) -> Tensor:
    array = np.ascontiguousarray(np.stack(images))
    tensor = torch.from_numpy(array).permute(0, 3, 1, 2).float().div_(255.0)
    mean = torch.tensor(_MEAN, dtype=tensor.dtype).view(1, 3, 1, 1)
    std = torch.tensor(_STD, dtype=tensor.dtype).view(1, 3, 1, 1)
    return (tensor - mean) / std


def _dataLoader(
    dataset: ManifestPairDataset, config: TrainingConfig, *, shuffle: bool
) -> DataLoader[dict[str, Any]]:
    generator = torch.Generator().manual_seed(config.optimization.seed)
    return DataLoader(
        dataset,
        batch_size=config.optimization.batchSize,
        shuffle=shuffle,
        num_workers=config.optimization.workers,
        pin_memory=True,
        persistent_workers=config.optimization.workers > 0,
        collate_fn=collateTrainingPairs,
        worker_init_fn=_seedWorker,
        generator=generator,
    )


def _moveBatch(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def _autocast(config: TrainingConfig) -> Any:
    precision = config.optimization.precision
    if precision == "fp32":
        return nullcontext()
    dtype = torch.float16 if precision == "fp16" else torch.bfloat16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _totalOptimizerSteps(config: TrainingConfig, batchesPerEpoch: int) -> int:
    available = math.ceil(batchesPerEpoch / config.optimization.gradientAccumulation)
    available *= config.optimization.epochs
    return (
        min(available, config.optimization.maxSteps)
        if config.optimization.maxSteps
        else available
    )


def _learningRateSchedule(config: TrainingConfig, totalSteps: int) -> Any:
    warmup = config.optimization.warmupSteps

    def schedule(step: int) -> float:
        if warmup > 0 and step < warmup:
            return max(1e-8, step / warmup)
        if config.optimization.scheduler == "constant":
            return 1.0
        progress = (step - warmup) / max(1, totalSteps - warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(max(progress, 0.0), 1.0)))

    return schedule


def _saveCheckpoint(
    path: Path,
    model: HiTTrainingModel,
    optimizer: AdamW,
    scheduler: Any,
    scaler: Any,
    config: TrainingConfig,
    epoch: int,
    globalStep: int,
    bestValidation: float,
    staleValidations: int,
    metrics: Mapping[str, float],
) -> None:
    payload = {
        "format": "instatarget.hit.training.v1",
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "globalStep": globalStep,
        "bestValidation": bestValidation,
        "staleValidations": staleValidations,
        "metrics": dict(metrics),
        "trainingConfig": _jsonableConfig(config),
        "manifestSha256": _sha256(config.data.manifest),
        "initialWeightsSha256": _sha256(config.model.initialWeights),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _resume(
    path: Path,
    model: HiTTrainingModel,
    optimizer: AdamW,
    scheduler: Any,
    scaler: Any,
    config: TrainingConfig,
) -> tuple[int, int, float, int]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    savedConfig = checkpoint.get("trainingConfig", {})
    if savedConfig.get("model", {}).get("stage") != config.model.stage:
        raise RuntimeError("resume checkpoint training stage does not match configuration")
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint.get("scaler", {}))
    return (
        int(checkpoint["epoch"]),
        int(checkpoint["globalStep"]),
        float(checkpoint["bestValidation"]),
        int(checkpoint["staleValidations"]),
    )


def _writeRunMetadata(config: TrainingConfig, reports: Sequence[Any], output: Path) -> None:
    payload = {
        "config": _jsonableConfig(config),
        "manifestSha256": _sha256(config.data.manifest),
        "initialWeightsSha256": _sha256(config.model.initialWeights),
        "parameterGroups": [asdict(report) for report in reports],
    }
    destination = output / "run_metadata.json"
    temporary = destination.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, destination)
    print(json.dumps({"parameterGroups": payload["parameterGroups"]}, sort_keys=True))


def _jsonableConfig(config: TrainingConfig) -> dict[str, Any]:
    def convert(value: Any) -> Any:
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {key: convert(item) for key, item in value.items()}
        return value

    return convert(asdict(config))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _seedEverything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _seedWorker(workerId: int) -> None:
    del workerId
    seed = torch.initial_seed() % (2**32)
    random.seed(seed)
    np.random.seed(seed)


def _logLoss(step: int, epoch: int, losses: Mapping[str, Tensor], optimizer: AdamW) -> None:
    print(
        json.dumps(
            {
                "step": step,
                "epoch": epoch,
                "loss": {name: float(value.detach().cpu()) for name, value in losses.items()},
                "learningRates": {
                    str(group.get("name", index)): group["lr"]
                    for index, group in enumerate(optimizer.param_groups)
                },
            },
            sort_keys=True,
        )
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the InstaTarget HiT backend")
    parser.add_argument("--config", default="configs/train_backend.yaml")
    arguments = parser.parse_args(argv)
    checkpoint = train(loadTrainingConfig(arguments.config))
    print(f"training complete: {checkpoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["collateTrainingPairs", "main", "train", "validate"]
