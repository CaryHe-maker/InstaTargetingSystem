"""Dependency-free binary ranking and calibration metrics."""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


def brierScore(probabilities: ArrayLike, labels: ArrayLike) -> float:
    probability, target = _validated(probabilities, labels)
    return float(np.mean((probability - target) ** 2))


def expectedCalibrationError(
    probabilities: ArrayLike, labels: ArrayLike, bins: int = 15
) -> float:
    probability, target = _validated(probabilities, labels)
    if bins <= 0:
        raise ValueError("bins must be positive")
    edges = np.linspace(0.0, 1.0, bins + 1)
    total = probability.size
    result = 0.0
    for index in range(bins):
        upperInclusive = index == bins - 1
        mask = (probability >= edges[index]) & (
            probability <= edges[index + 1]
            if upperInclusive
            else probability < edges[index + 1]
        )
        if np.any(mask):
            result += float(np.mean(mask)) * abs(
                float(np.mean(probability[mask])) - float(np.mean(target[mask]))
            )
    return result if total else 0.0


def rocAuc(probabilities: ArrayLike, labels: ArrayLike) -> float:
    probability, target = _validated(probabilities, labels)
    positive = int(np.sum(target == 1.0))
    negative = int(np.sum(target == 0.0))
    if positive == 0 or negative == 0:
        return float("nan")
    order = np.argsort(probability, kind="stable")
    ranks = np.empty_like(order, dtype=np.float64)
    sortedProbability = probability[order]
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and sortedProbability[end] == sortedProbability[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    numerator = np.sum(ranks[target == 1.0]) - positive * (positive + 1) / 2
    return float(numerator / (positive * negative))


def prAuc(probabilities: ArrayLike, labels: ArrayLike) -> float:
    probability, target = _validated(probabilities, labels)
    positives = float(np.sum(target))
    if positives == 0.0:
        return float("nan")
    order = np.argsort(-probability, kind="stable")
    sortedLabels = target[order]
    truePositive = np.cumsum(sortedLabels)
    falsePositive = np.cumsum(1.0 - sortedLabels)
    recall = truePositive / positives
    precision = truePositive / np.maximum(truePositive + falsePositive, 1.0)
    recall = np.concatenate(([0.0], recall))
    precision = np.concatenate(([1.0], precision))
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def _validated(probabilities: ArrayLike, labels: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    probability = np.asarray(probabilities, dtype=np.float64).reshape(-1)
    target = np.asarray(labels, dtype=np.float64).reshape(-1)
    if probability.size == 0 or probability.shape != target.shape:
        raise ValueError("probabilities and labels must be non-empty and have equal shape")
    if not np.isfinite(probability).all() or np.any((probability < 0.0) | (probability > 1.0)):
        raise ValueError("probabilities must be finite and in [0, 1]")
    if not np.isin(target, (0.0, 1.0)).all():
        raise ValueError("labels must be binary")
    return probability, target


__all__ = ["brierScore", "expectedCalibrationError", "prAuc", "rocAuc"]
