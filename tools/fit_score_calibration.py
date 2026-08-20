"""Fit the Stage 3 score artifact from canonical calibration-split candidates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from instatarget.controller.score_calibration import CALIBRATION_FORMAT, sha256File
from instatarget.eval.calibration_metrics import (
    brierScore,
    expectedCalibrationError,
    prAuc,
    rocAuc,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", required=True, type=Path, nargs="+")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(r"E:\NewDownload\train"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    datasetRoot = args.dataset_root.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    if not manifest.is_relative_to(datasetRoot):
        raise RuntimeError(f"manifest must be inside canonical dataset root: {datasetRoot}")
    rows = _loadRows(args.candidates)
    if any(row.get("split", "calibration") != "calibration" for row in rows):
        raise RuntimeError("score fitting accepts calibration split rows only")

    raw = np.asarray([float(row["modelScore"]) for row in rows], dtype=np.float64)
    motion = np.asarray([float(row["motionProbability"]) for row in rows], dtype=np.float64)
    labels = np.asarray([bool(row["hitAt0.5"]) for row in rows], dtype=np.float64)
    targetPresent = np.asarray([bool(row["targetPresent"]) for row in rows])
    sequenceIds = {str(row["sequenceId"]) for row in rows}
    if len(sequenceIds) < 2:
        raise RuntimeError("calibration requires at least two sequences")
    if labels.min() == labels.max():
        raise RuntimeError("calibration requires both hit and miss candidates")
    if not np.isfinite(raw).all() or not np.isfinite(motion).all():
        raise RuntimeError("candidate scores must be finite")

    alpha, beta, intercept = _fitBeta(raw, labels)
    calibrated = _beta(raw, alpha, beta, intercept)
    appearanceWeight, candidateThreshold, fusionThreshold = _selectOperatingPoint(
        calibrated,
        motion,
        labels,
        targetPresent,
    )
    artifact = {
        "format": CALIBRATION_FORMAT,
        "checkpointSha256": sha256File(args.checkpoint),
        "manifestSha256": sha256File(manifest),
        "split": "calibration",
        "appearanceInput": "presence_quality_product",
        "appearance": {
            "method": "beta",
            "alpha": alpha,
            "beta": beta,
            "intercept": intercept,
        },
        "singleScore": {
            "appearanceWeight": appearanceWeight,
            "motionWeight": 1.0 - appearanceWeight,
        },
        "thresholds": {
            "candidateMinScore": candidateThreshold,
            "fusionSourceMinConfidence": fusionThreshold,
        },
        "fit": {
            "sampleCount": len(rows),
            "positiveCount": int(labels.sum()),
            "negativeCount": int((1.0 - labels).sum()),
            "sequenceCount": len(sequenceIds),
            "rawBrier": brierScore(raw, labels),
            "calibratedBrier": brierScore(calibrated, labels),
            "rawEce": expectedCalibrationError(raw, labels),
            "calibratedEce": expectedCalibrationError(calibrated, labels),
            "prAuc": prAuc(calibrated, labels),
            "rocAuc": rocAuc(calibrated, labels),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(artifact, indent=2))
    return 0


def _loadRows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    required = {
        "sequenceId",
        "modelScore",
        "motionProbability",
        "hitAt0.5",
        "targetPresent",
    }
    for path in paths:
        with path.expanduser().resolve().open("r", encoding="utf-8") as stream:
            for line in stream:
                if not line.strip():
                    continue
                row = json.loads(line)
                missing = sorted(required - set(row))
                if missing:
                    raise RuntimeError(f"candidate row fields missing in {path}: {missing}")
                rows.append(row)
    if not rows:
        raise RuntimeError("candidate inputs are empty")
    return rows


def _fitBeta(probabilities: np.ndarray, labels: np.ndarray) -> tuple[float, float, float]:
    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    logP = np.log(clipped)
    negativeLogP = -np.log1p(-clipped)

    def objective(parameters: np.ndarray) -> float:
        alpha = np.exp(parameters[0])
        beta = np.exp(parameters[1])
        logits = parameters[2] + alpha * logP + beta * negativeLogP
        loss = np.logaddexp(0.0, logits) - labels * logits
        return float(np.mean(loss) + 1e-6 * (alpha * alpha + beta * beta))

    result = minimize(
        objective,
        np.asarray((0.0, 0.0, 0.0), dtype=np.float64),
        method="L-BFGS-B",
        bounds=((-8.0, 8.0), (-8.0, 8.0), (-30.0, 30.0)),
    )
    if not result.success or not np.isfinite(result.fun):
        raise RuntimeError(f"beta calibration fit failed: {result.message}")
    return float(np.exp(result.x[0])), float(np.exp(result.x[1])), float(result.x[2])


def _beta(
    probabilities: np.ndarray,
    alpha: float,
    beta: float,
    intercept: float,
) -> np.ndarray:
    clipped = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    return expit(intercept + alpha * np.log(clipped) - beta * np.log1p(-clipped))


def _selectOperatingPoint(
    appearance: np.ndarray,
    motion: np.ndarray,
    labels: np.ndarray,
    targetPresent: np.ndarray,
) -> tuple[float, float, float]:
    absent = ~targetPresent
    best: tuple[float, float, float, float, float, float] | None = None
    for appearanceWeight in (0.50, 0.60, 0.70, 0.80, 0.90):
        scores = appearanceWeight * appearance + (1.0 - appearanceWeight) * motion
        thresholds = np.unique(
            np.concatenate(
                (
                    np.quantile(scores, np.linspace(0.05, 0.95, 91)),
                    np.asarray((0.40,), dtype=np.float64),
                )
            )
        )
        for threshold in thresholds:
            accepted = scores >= threshold
            falsePositiveRate = float(np.mean(accepted[absent])) if np.any(absent) else 0.0
            truePositiveRate = float(np.mean(accepted[labels == 1.0]))
            precisionDenominator = int(np.sum(accepted))
            precision = (
                float(np.sum(labels[accepted])) / precisionDenominator
                if precisionDenominator
                else 0.0
            )
            feasible = falsePositiveRate <= 0.05
            objective = (1.0 if feasible else 0.0, truePositiveRate, precision, -falsePositiveRate)
            candidate = (*objective, appearanceWeight, float(threshold))
            if best is None or candidate[:4] > best[:4]:
                best = candidate
    if best is None:
        raise RuntimeError("cannot select a calibration operating point")
    appearanceWeight = best[4]
    candidateThreshold = best[5]
    scores = appearanceWeight * appearance + (1.0 - appearanceWeight) * motion
    absentScores = scores[absent]
    fusionThreshold = (
        float(np.quantile(absentScores, 0.99)) if absentScores.size else candidateThreshold
    )
    fusionThreshold = max(candidateThreshold, min(0.99, fusionThreshold))
    return (
        round(float(appearanceWeight), 6),
        round(float(candidateThreshold), 6),
        round(float(fusionThreshold), 6),
    )


if __name__ == "__main__":
    raise SystemExit(main())
