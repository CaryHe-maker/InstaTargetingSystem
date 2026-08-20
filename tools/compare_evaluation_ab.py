"""Compare two manifest evaluation artifacts against release gates."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

_CANDIDATE_SCALARS = (
    "modelScore",
    "presenceLogit",
    "qualityLogit",
    "presenceProbability",
    "qualityProbability",
    "predictedIoU",
    "cornerScore",
    "appearanceProbability",
    "motionProbability",
    "singleScore",
)
_CANDIDATE_STRUCTURES = ("localBBox", "projectedBBox", "projectedBFoV")
_FINAL_STRUCTURES = ("bbox", "bfov")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--fp16-gates", action="store_true")
    parser.add_argument(
        "--min-p95-improvement",
        type=float,
        default=0.05,
        help="Minimum relative P95 reduction required by --fp16-gates (default: 5%%).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if not 0.0 <= args.min_p95_improvement < 1.0:
        raise ValueError("--min-p95-improvement must be in [0, 1)")
    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    candidate = json.loads(args.candidate.read_text(encoding="utf-8"))
    _requireComparableReports(baseline, candidate)

    baselineCandidates = _indexCandidates(_readCandidates(args.baseline))
    candidateCandidates = _indexCandidates(_readCandidates(args.candidate))
    if baselineCandidates.keys() != candidateCandidates.keys():
        missing = sorted(baselineCandidates.keys() - candidateCandidates.keys())[:3]
        extra = sorted(candidateCandidates.keys() - baselineCandidates.keys())[:3]
        raise RuntimeError(f"candidate keys changed: missing={missing}, extra={extra}")

    candidateDeltas = {
        field: _maxScalarDelta(baselineCandidates, candidateCandidates, field)
        for field in _CANDIDATE_SCALARS
    }
    candidateStructureDeltas = {
        field: _maxStructureDelta(baselineCandidates, candidateCandidates, field)
        for field in _CANDIDATE_STRUCTURES
        if _allHaveField(baselineCandidates, candidateCandidates, field)
    }

    baselineFrames = _indexFrames(baseline["frames"])
    candidateFrames = _indexFrames(candidate["frames"])
    if baselineFrames.keys() != candidateFrames.keys():
        raise RuntimeError("final TrackResult frame indexes changed")
    finalStructureDeltas = {
        field: _maxStructureDelta(baselineFrames, candidateFrames, field)
        for field in _FINAL_STRUCTURES
        if _allHaveField(baselineFrames, candidateFrames, field)
    }
    finalConfidenceDelta = _maxScalarDelta(
        baselineFrames,
        candidateFrames,
        "confidence",
    )
    finalDiscreteEqual = all(
        left[field] == candidateFrames[key][field]
        for key, left in baselineFrames.items()
        for field in ("valid", "status", "resultSource")
    )

    nonFinite = _nonFinitePaths(candidate)
    nonFinite.extend(_nonFinitePaths(list(candidateCandidates.values()), root="candidates"))
    leftSummary = baseline["summary"]
    rightSummary = candidate["summary"]
    meanIoURelativeDrop = _relativeDrop(
        float(leftSummary["circularErpMeanIoU"]),
        float(rightSummary["circularErpMeanIoU"]),
    )
    successDrop = float(leftSummary["successRateAt0.5"]) - float(
        rightSummary["successRateAt0.5"]
    )
    baselineP95 = float(leftSummary["latencyP95Ms"])
    candidateP95 = float(rightSummary["latencyP95Ms"])
    p95Improvement = (baselineP95 - candidateP95) / max(baselineP95, 1e-12)

    numericDeltas = (
        list(candidateDeltas.values())
        + list(candidateStructureDeltas.values())
        + list(finalStructureDeltas.values())
        + [finalConfidenceDelta]
    )
    passed = not nonFinite and all(math.isfinite(value) for value in numericDeltas)
    if args.fp16_gates:
        passed = passed and meanIoURelativeDrop <= 0.005 and successDrop <= 0.005
        passed = passed and float(rightSummary["trackingLossRate"]) <= float(
            leftSummary["trackingLossRate"]
        )
        passed = passed and float(rightSummary["absentFalsePositiveRate"]) <= float(
            leftSummary["absentFalsePositiveRate"]
        )
        passed = passed and p95Improvement >= args.min_p95_improvement
    else:
        passed = passed and finalDiscreteEqual and all(value == 0.0 for value in numericDeltas)

    result: dict[str, Any] = {
        "passed": bool(passed),
        "candidateOutputMaxAbsDeltas": candidateDeltas,
        "candidateStructureMaxAbsDeltas": candidateStructureDeltas,
        "finalTrackResultMaxAbsDeltas": {
            **finalStructureDeltas,
            "confidence": finalConfidenceDelta,
        },
        "finalTrackResultDiscreteEqual": finalDiscreteEqual,
        "nonFiniteCount": len(nonFinite),
        "nonFiniteExamples": nonFinite[:10],
        "meanIoURelativeDrop": meanIoURelativeDrop,
        "successRateAbsoluteDrop": successDrop,
        "trackingLossRateDelta": float(rightSummary["trackingLossRate"])
        - float(leftSummary["trackingLossRate"]),
        "absentFalsePositiveRateDelta": float(rightSummary["absentFalsePositiveRate"])
        - float(leftSummary["absentFalsePositiveRate"]),
        "baselineP95Ms": baselineP95,
        "candidateP95Ms": candidateP95,
        "p95RelativeImprovement": p95Improvement,
        "requiredP95RelativeImprovement": (
            args.min_p95_improvement if args.fp16_gates else None
        ),
    }
    print(json.dumps(result, indent=2))
    return 0 if passed else 1


def _requireComparableReports(left: dict[str, Any], right: dict[str, Any]) -> None:
    for field in ("sequence", "split", "frameCount"):
        if left["summary"].get(field) != right["summary"].get(field):
            raise RuntimeError(f"report summary differs for {field}")


def _readCandidates(report: Path) -> list[dict[str, Any]]:
    path = report.with_name(f"{report.stem}.candidates.jsonl")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _indexCandidates(rows: list[dict[str, Any]]) -> dict[tuple[int, int, int], dict[str, Any]]:
    result = {
        (
            int(row["frameIndex"]),
            int(row.get("roundIndex", 0)),
            int(row["viewId"]),
        ): row
        for row in rows
    }
    if len(result) != len(rows):
        raise RuntimeError("candidate artifact contains duplicate frame/round/view keys")
    return result


def _indexFrames(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    result = {int(row["frameIndex"]): row for row in rows}
    if len(result) != len(rows):
        raise RuntimeError("report contains duplicate final frame indexes")
    return result


def _allHaveField(
    left: dict[Any, dict[str, Any]],
    right: dict[Any, dict[str, Any]],
    field: str,
) -> bool:
    return all(field in row for row in left.values()) and all(
        field in row for row in right.values()
    )


def _maxScalarDelta(
    left: dict[Any, dict[str, Any]],
    right: dict[Any, dict[str, Any]],
    field: str,
) -> float:
    return max(
        (abs(float(row[field]) - float(right[key][field])) for key, row in left.items()),
        default=0.0,
    )


def _maxStructureDelta(
    left: dict[Any, dict[str, Any]],
    right: dict[Any, dict[str, Any]],
    field: str,
) -> float:
    return max(
        (
            _maxFlatDelta(_flattenNumeric(row[field]), _flattenNumeric(right[key][field]))
            for key, row in left.items()
        ),
        default=0.0,
    )


def _flattenNumeric(value: Any, prefix: str = "") -> dict[str, float]:
    if isinstance(value, dict):
        result: dict[str, float] = {}
        for key, child in value.items():
            childPrefix = f"{prefix}.{key}" if prefix else str(key)
            result.update(_flattenNumeric(child, childPrefix))
        return result
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return {}
    return {prefix: float(value)}


def _maxFlatDelta(left: dict[str, float], right: dict[str, float]) -> float:
    if left.keys() != right.keys():
        raise RuntimeError("numeric structure fields changed")
    return max((abs(value - right[key]) for key, value in left.items()), default=0.0)


def _nonFinitePaths(value: Any, *, root: str = "report") -> list[str]:
    invalid: list[str] = []

    def visit(item: Any, path: str) -> None:
        if isinstance(item, dict):
            for key, child in item.items():
                visit(child, f"{path}.{key}")
        elif isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]")
        elif isinstance(item, (int, float)) and not isinstance(item, bool):
            if not math.isfinite(float(item)):
                invalid.append(path)

    visit(value, root)
    return invalid


def _relativeDrop(baseline: float, candidate: float) -> float:
    return (baseline - candidate) / max(abs(baseline), 1e-12)


if __name__ == "__main__":
    raise SystemExit(main())
