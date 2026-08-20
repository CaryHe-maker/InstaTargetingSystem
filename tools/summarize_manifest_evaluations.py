"""Aggregate compatible manifest-controller evaluations across sequences."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from instatarget.eval.otb_metrics import auc, trackingLossRate


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", required=True, type=Path, nargs="+")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--allow-mixed-splits",
        action="store_true",
        help="Allow aggregating sequences from multiple manifest splits.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in args.inputs]
    if not reports:
        raise RuntimeError("at least one evaluation report is required")
    splits = {report["summary"]["split"] for report in reports}
    weights = {report["summary"]["weights"] for report in reports}
    sequences = [report["summary"]["sequence"] for report in reports]
    if len(weights) != 1:
        raise RuntimeError("all evaluations must use one checkpoint")
    if len(splits) != 1 and not args.allow_mixed_splits:
        raise RuntimeError("all evaluations must use one split unless --allow-mixed-splits is set")
    if len(sequences) != len(set(sequences)):
        raise RuntimeError("evaluation sequence IDs must be unique")

    frameRows: list[dict[str, Any]] = []
    statusCounts: Counter[str] = Counter()
    sourceCounts: Counter[str] = Counter()
    latency: list[float] = []
    for path, report in zip(args.inputs, reports, strict=True):
        frameRows.extend(report["frameMetrics"])
        statusCounts.update(report["summary"]["statusCounts"])
        sourceCounts.update(report["summary"]["resultSourceCounts"])
        timingPath = path.with_name(f"{path.stem}.timings.jsonl")
        with timingPath.open("r", encoding="utf-8") as stream:
            timingRows = [json.loads(line) for line in stream if line.strip()]
        expected = int(report["summary"]["frameCount"])
        if len(timingRows) != expected:
            raise RuntimeError(f"timing row count mismatch for {path}")
        indices = [int(row["frameIndex"]) for row in timingRows]
        if indices != list(range(expected)):
            raise RuntimeError(f"non-contiguous frameIndex in {timingPath}")
        latency.extend(float(row["totalProcessingMs"]) for row in timingRows[1:])

    visible = [row for row in frameRows if row["visible"]]
    absent = [row for row in frameRows if not row["visible"]]
    erpIous = [float(row["circularErpIoU"]) for row in visible]
    sphericalIous = [float(row["sphericalIoU"]) for row in visible]
    centerErrors = [float(row["centerErrorDeg"]) for row in visible]
    widthErrors = [float(row["widthRelativeError"]) for row in visible]
    heightErrors = [float(row["heightRelativeError"]) for row in visible]
    summary = {
        "format": "instatarget.manifest-controller-aggregate.v1",
        "split": next(iter(splits)) if len(splits) == 1 else "mixed",
        "weights": next(iter(weights)),
        "sequenceIds": sequences,
        "sequenceCount": len(sequences),
        "evaluatedVisibleFrames": len(visible),
        "absentFrames": len(absent),
        "circularErpMeanIoU": _mean(erpIous),
        "successAUC": auc(erpIous),
        "successRateAt0.5": _mean([value > 0.5 for value in erpIous]),
        "lostFrameCount": int(np.sum(np.asarray(erpIous) <= 1e-12)),
        "trackingLossRate": trackingLossRate(erpIous),
        "sphericalMeanIoU": _mean(sphericalIous),
        "centerErrorP50Deg": _percentile(centerErrors, 50),
        "centerErrorP95Deg": _percentile(centerErrors, 95),
        "widthRelativeErrorP50": _percentile(widthErrors, 50),
        "widthRelativeErrorP95": _percentile(widthErrors, 95),
        "heightRelativeErrorP50": _percentile(heightErrors, 50),
        "heightRelativeErrorP95": _percentile(heightErrors, 95),
        "absentFalsePositiveRate": _mean([row["valid"] for row in absent]),
        "latencyP50Ms": _percentile(latency, 50),
        "latencyP95Ms": _percentile(latency, 95),
        "latencyP99Ms": _percentile(latency, 99),
        "statusCounts": dict(statusCounts),
        "resultSourceCounts": dict(sourceCounts),
        "validRate": _mean(
            [
                float(row["valid"])
                for report in reports
                for row in report["frames"][1:]
            ]
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(summary, indent=2))
    return 0


def _mean(values: list[Any]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else 0.0


def _percentile(values: list[float], percentile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), percentile)) if values else 0.0


if __name__ == "__main__":
    raise SystemExit(main())
