"""Collect Stage 3 calibration candidates from canonical labeled views."""

from __future__ import annotations

import argparse
import json
from dataclasses import replace
from math import pi
from pathlib import Path

import numpy as np

from instatarget.core.config import loadConfig
from instatarget.core.types import BBoxXYWH, BFoV, ViewSpec
from instatarget.eval.otb_metrics import bboxIoU, circularBBoxIoU
from instatarget.geometry.projection_math import makeSphericalPoint
from instatarget.geometry.spherical_geometry import SphericalGeometryImpl
from instatarget.tracker.pytorch_hit_session import PyTorchHiTSession
from instatarget.training.dataset import (
    ManifestRecord,
    VideoFrameDecoder,
    _bfovToNormalizedLocal,
    _recordBfov,
    loadManifest,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frames-per-sequence", type=int, default=200)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(r"E:\NewDownload\train"),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.frames_per_sequence <= 0:
        raise ValueError("--frames-per-sequence must be positive")
    datasetRoot = args.dataset_root.expanduser().resolve()
    manifest = args.manifest.expanduser().resolve()
    if not manifest.is_relative_to(datasetRoot):
        raise RuntimeError(f"manifest must be inside canonical dataset root: {datasetRoot}")
    records = tuple(record for record in loadManifest(manifest) if record.split == "calibration")
    outside = [
        record.videoPath
        for record in records
        if not record.videoPath.is_relative_to(datasetRoot)
    ]
    if outside:
        raise RuntimeError(f"calibration video is outside canonical dataset root: {outside[0]}")

    grouped: dict[str, list[ManifestRecord]] = {}
    for record in records:
        grouped.setdefault(record.sequenceId, []).append(record)
    selected = {
        sequenceId: _sampleSequence(sequenceRecords, args.frames_per_sequence)
        for sequenceId, sequenceRecords in grouped.items()
    }
    config = loadConfig(args.config)
    modelConfig = replace(config.model, weights=args.weights.expanduser().resolve())
    geometry = SphericalGeometryImpl(
        boundarySamplesPerEdge=config.geometry.boundarySamplesPerEdge
    )
    session = PyTorchHiTSession(modelConfig)
    decoder = VideoFrameDecoder(cacheSize=2)
    rows: list[dict[str, object]] = []
    try:
        for sequenceId, sequenceRecords in sorted(selected.items()):
            templateRecord = next(
                record
                for record in sorted(grouped[sequenceId], key=lambda item: item.frameIndex)
                if record.visible and record.bfov is not None
            )
            templateTarget = _recordBfov(templateRecord, geometry)
            templateSpec = _templateSpec(templateTarget)
            templateFrame = decoder.read(templateRecord)
            templateView = geometry.cropViews(templateFrame, (templateSpec,))[0]
            templateBox = _normalizedToPixels(
                _requireTargetBox(templateTarget, templateSpec),
                templateSpec.outputWidthPx,
                templateSpec.outputHeightPx,
            )
            template = session.encodeTemplate(templateView.rgb, templateBox)
            for record in sequenceRecords:
                if not record.visible or record.bfov is None or record.bbox is None:
                    continue
                frame = decoder.read(record)
                specs = _calibrationSpecs(record.bfov, config.geometry.viewWidthPx)
                views = geometry.cropViews(frame, specs)
                predictions = session.inferBatch(
                    tuple(view.rgb for view in views),
                    (template,),
                )
                for view, prediction in zip(views, predictions, strict=True):
                    targetNormalized = _bfovToNormalizedLocal(record.bfov, view.spec)
                    targetBox = (
                        _normalizedToPixels(
                            targetNormalized,
                            view.spec.outputWidthPx,
                            view.spec.outputHeightPx,
                        )
                        if targetNormalized is not None
                        else None
                    )
                    localIoU = bboxIoU(prediction.bbox, targetBox) if targetBox else 0.0
                    projection = geometry.projectLocalBoxBoundary(
                        prediction.bbox,
                        view.spec,
                        record.width,
                        record.height,
                    )
                    rows.append(
                        {
                            "sequenceId": record.sequenceId,
                            "split": "calibration",
                            "frameIndex": record.frameIndex,
                            "viewId": view.spec.viewId,
                            "targetPresent": targetBox is not None,
                            "targetLocalBoxCxCyWh": (
                                targetNormalized.tolist()
                                if targetNormalized is not None
                                else None
                            ),
                            "localIoU": localIoU,
                            "circularErpIoU": circularBBoxIoU(
                                projection.bbox,
                                record.bbox,
                                record.width,
                            ),
                            "hitAt0.5": bool(targetBox is not None and localIoU >= 0.5),
                            "modelScore": prediction.modelScore,
                            "presenceLogit": prediction.presenceLogit,
                            "qualityLogit": prediction.qualityLogit,
                            "presenceProbability": prediction.presenceProbability,
                            "qualityProbability": prediction.qualityProbability,
                            "predictedIoU": prediction.predictedIoU,
                            "cornerScore": prediction.cornerScore,
                            "motionProbability": _viewMotionScore(view.spec, record.bfov),
                            "viewPitchDeg": float(np.degrees(view.spec.bfov.center.pitchRad)),
                            "viewFovDeg": float(np.degrees(view.spec.bfov.horizontalFovRad)),
                            "targetPitchDeg": float(np.degrees(record.bfov.center.pitchRad)),
                            "difficultType": record.difficultType,
                        }
                    )
            print(f"collected {sequenceId}: frames={len(sequenceRecords)}")
    finally:
        decoder.close()
        session.close()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        "".join(json.dumps(row, separators=(",", ":")) + "\n" for row in rows),
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps({"sequenceCount": len(selected), "candidateCount": len(rows)}, indent=2))
    return 0


def _sampleSequence(records: list[ManifestRecord], count: int) -> tuple[ManifestRecord, ...]:
    ordered = tuple(sorted(records, key=lambda item: item.frameIndex))
    if len(ordered) <= count:
        return ordered
    indices = np.linspace(0, len(ordered) - 1, count, dtype=np.int64)
    return tuple(ordered[int(index)] for index in np.unique(indices))


def _templateSpec(target: BFoV) -> ViewSpec:
    return ViewSpec(
        viewId=0,
        bfov=BFoV(
            center=target.center,
            horizontalFovRad=min(2.0 * pi / 3.0, max(pi / 12.0, 2.0 * target.horizontalFovRad)),
            verticalFovRad=min(2.0 * pi / 3.0, max(pi / 12.0, 2.0 * target.verticalFovRad)),
        ),
        outputWidthPx=128,
        outputHeightPx=128,
    )


def _calibrationSpecs(target: BFoV, outputSize: int) -> tuple[ViewSpec, ...]:
    fov = min(
        2.0 * pi / 3.0,
        max(pi / 6.0, 2.0 * max(target.horizontalFovRad, target.verticalFovRad)),
    )
    offsets = ((0.0, 0.0), (0.28, 0.0), (0.0, 0.28), (1.20, 0.0))
    return tuple(
        ViewSpec(
            viewId=viewId,
            bfov=BFoV(
                center=makeSphericalPoint(
                    target.center.yawRad + yawRatio * fov,
                    target.center.pitchRad + pitchRatio * fov,
                ),
                horizontalFovRad=fov,
                verticalFovRad=fov,
            ),
            outputWidthPx=outputSize,
            outputHeightPx=outputSize,
        )
        for viewId, (yawRatio, pitchRatio) in enumerate(offsets)
    )


def _requireTargetBox(target: BFoV, spec: ViewSpec) -> np.ndarray:
    normalized = _bfovToNormalizedLocal(target, spec)
    if normalized is None:
        raise RuntimeError("template target is not visible in its own template view")
    return normalized


def _normalizedToPixels(box: np.ndarray, width: int, height: int) -> BBoxXYWH:
    cx, cy, boxWidth, boxHeight = (float(value) for value in box)
    return BBoxXYWH(
        (cx - boxWidth / 2.0) * width,
        (cy - boxHeight / 2.0) * height,
        boxWidth * width,
        boxHeight * height,
    )


def _viewMotionScore(spec: ViewSpec, target: BFoV) -> float:
    dot = float(
        np.clip(
            spec.bfov.center.x * target.center.x
            + spec.bfov.center.y * target.center.y
            + spec.bfov.center.z * target.center.z,
            -1.0,
            1.0,
        )
    )
    angle = float(np.arccos(dot))
    return float(np.clip(1.0 - angle / (pi / 6.0) * 0.1, 0.0, 1.0))


if __name__ == "__main__":
    raise SystemExit(main())
