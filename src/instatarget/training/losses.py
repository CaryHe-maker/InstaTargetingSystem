"""Masked localization, presence, and quality losses for HiT training."""

from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import Tensor
from torch.nn import functional as F

from instatarget.core.config import TrainingLossConfig


def computeTrainingLoss(
    outputs: Mapping[str, Tensor],
    targets: Mapping[str, Tensor],
    config: TrainingLossConfig,
) -> dict[str, Tensor]:
    predBoxes = outputs["predBoxes"].reshape(-1, 4)
    presenceLogit = outputs["presenceLogit"].reshape(-1)
    qualityLogit = outputs["qualityLogit"].reshape(-1)
    targetBoxes = targets["boxes"].reshape(-1, 4).to(predBoxes)
    present = targets["present"].reshape(-1).to(device=predBoxes.device, dtype=torch.bool)
    sampleWeights = targets.get("labelQuality", torch.ones_like(presenceLogit)).reshape(-1)
    sampleWeights = sampleWeights.to(presenceLogit).clamp(0.0, 1.0)
    if not (
        predBoxes.shape[0]
        == presenceLogit.shape[0]
        == qualityLogit.shape[0]
        == targetBoxes.shape[0]
        == present.shape[0]
    ):
        raise ValueError("training output and target batch sizes must match")

    presenceTargets = present.to(presenceLogit.dtype)
    presencePerItem = F.binary_cross_entropy_with_logits(
        presenceLogit, presenceTargets, reduction="none"
    )
    if config.presenceFocalGamma > 0.0:
        probability = torch.sigmoid(presenceLogit)
        targetProbability = torch.where(present, probability, 1.0 - probability)
        presencePerItem *= (1.0 - targetProbability).pow(config.presenceFocalGamma)
    presenceLoss = _weightedMean(presencePerItem, sampleWeights)

    if present.any():
        positivePred = predBoxes[present]
        positiveTarget = targetBoxes[present]
        positiveWeights = sampleWeights[present]
        l1PerItem = F.l1_loss(positivePred, positiveTarget, reduction="none").mean(dim=1)
        l1Loss = _weightedMean(l1PerItem, positiveWeights)
        giou = alignedGeneralizedBoxIou(positivePred, positiveTarget)
        giouLoss = _weightedMean(1.0 - giou, positiveWeights)
        qualityTargets = torch.zeros_like(qualityLogit)
        qualityTargets[present] = alignedBoxIou(positivePred, positiveTarget).detach()
    else:
        zero = predBoxes.sum() * 0.0
        l1Loss = zero
        giouLoss = zero
        qualityTargets = torch.zeros_like(qualityLogit)

    qualityWeights = sampleWeights * torch.where(
        present,
        torch.ones_like(sampleWeights),
        torch.full_like(sampleWeights, config.qualityNegativeWeight),
    )
    qualityLoss = _weightedMean(
        F.binary_cross_entropy_with_logits(qualityLogit, qualityTargets, reduction="none"),
        qualityWeights,
    )
    total = (
        config.presenceWeight * presenceLoss
        + config.l1Weight * l1Loss
        + config.giouWeight * giouLoss
        + config.qualityWeight * qualityLoss
    )
    return {
        "total": total,
        "presence": presenceLoss,
        "bboxL1": l1Loss,
        "bboxGiou": giouLoss,
        "quality": qualityLoss,
        "positiveCount": present.sum().to(predBoxes.dtype),
        "negativeCount": (~present).sum().to(predBoxes.dtype),
        "qualityTargetMean": qualityTargets.mean(),
    }


def alignedBoxIou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    first = _cxcywhToXyxy(boxes1)
    second = _cxcywhToXyxy(boxes2)
    intersectionMin = torch.maximum(first[:, :2], second[:, :2])
    intersectionMax = torch.minimum(first[:, 2:], second[:, 2:])
    intersection = (intersectionMax - intersectionMin).clamp_min(0.0).prod(dim=1)
    area1 = (first[:, 2:] - first[:, :2]).clamp_min(0.0).prod(dim=1)
    area2 = (second[:, 2:] - second[:, :2]).clamp_min(0.0).prod(dim=1)
    return intersection / (area1 + area2 - intersection).clamp_min(1e-8)


def alignedGeneralizedBoxIou(boxes1: Tensor, boxes2: Tensor) -> Tensor:
    first = _cxcywhToXyxy(boxes1)
    second = _cxcywhToXyxy(boxes2)
    iou = alignedBoxIou(boxes1, boxes2)
    enclosureMin = torch.minimum(first[:, :2], second[:, :2])
    enclosureMax = torch.maximum(first[:, 2:], second[:, 2:])
    enclosureArea = (enclosureMax - enclosureMin).clamp_min(0.0).prod(dim=1)
    intersectionMin = torch.maximum(first[:, :2], second[:, :2])
    intersectionMax = torch.minimum(first[:, 2:], second[:, 2:])
    intersection = (intersectionMax - intersectionMin).clamp_min(0.0).prod(dim=1)
    area1 = (first[:, 2:] - first[:, :2]).clamp_min(0.0).prod(dim=1)
    area2 = (second[:, 2:] - second[:, :2]).clamp_min(0.0).prod(dim=1)
    union = area1 + area2 - intersection
    return iou - (enclosureArea - union) / enclosureArea.clamp_min(1e-8)


def _cxcywhToXyxy(boxes: Tensor) -> Tensor:
    center = boxes[:, :2]
    halfSize = boxes[:, 2:].clamp_min(0.0) / 2.0
    return torch.cat((center - halfSize, center + halfSize), dim=1)


def _weightedMean(values: Tensor, weights: Tensor) -> Tensor:
    return (values * weights).sum() / weights.sum().clamp_min(1e-8)


__all__ = ["alignedBoxIou", "alignedGeneralizedBoxIou", "computeTrainingLoss"]
