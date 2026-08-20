"""Run and aggregate the isolated LOST experiment matrix."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from eval_lost_experiment import POLICIES, VIEW_STRATEGIES
from eval_lost_experiment import main as evaluateLost
from summarize_manifest_evaluations import main as summarizeEvaluations

DEFAULT_SEQUENCES = (
    "train_real/seq_0005",
    "train_sim/seq_0010",
    "train_sim/seq_0017",
    "train_sim/seq_0075",
    "train_sim/seq_0045",
)
REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path(r"E:\NewDownload\train\manifest.jsonl"),
    )
    parser.add_argument("--dataset-root", type=Path, default=Path(r"E:\NewDownload\train"))
    parser.add_argument(
        "--config",
        type=Path,
        default=REPOSITORY_ROOT / "configs" / "RGBonly.yaml",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=REPOSITORY_ROOT / "models" / "hit_small_stage3.pth",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--sequence", nargs="+", default=list(DEFAULT_SEQUENCES))
    parser.add_argument("--policy", nargs="+", choices=POLICIES, default=list(POLICIES))
    parser.add_argument(
        "--view-strategy",
        nargs="+",
        choices=VIEW_STRATEGIES,
        default=list(VIEW_STRATEGIES),
    )
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--spherical-samples-yaw", type=int, default=128)
    parser.add_argument("--spherical-samples-pitch", type=int, default=64)
    parser.add_argument(
        "--progress-interval-seconds",
        type=float,
        default=60.0,
        help="print in-sequence metrics at this interval; use 0 to disable",
    )
    parser.add_argument("--no-resume", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    outputRoot = args.output_dir.expanduser().resolve()
    combinations = [
        (policy, strategy)
        for policy in args.policy
        for strategy in args.view_strategy
    ]
    total = len(combinations) * len(args.sequence)
    completed = 0
    for policy, strategy in combinations:
        combinationRoot = outputRoot / f"{policy}__{strategy}"
        evaluationsRoot = combinationRoot / "evaluations"
        evaluationsRoot.mkdir(parents=True, exist_ok=True)
        reports: list[Path] = []
        for sequence in args.sequence:
            completed += 1
            safeName = sequence.replace("/", "__").replace("\\", "__")
            reportPath = evaluationsRoot / f"{safeName}.json"
            if not args.no_resume and _isComplete(reportPath, policy, strategy, args.max_frames):
                print(f"[{completed}/{total}] resume {policy} {strategy} {sequence}", flush=True)
            else:
                print(f"[{completed}/{total}] run {policy} {strategy} {sequence}", flush=True)
                command = [
                    "--manifest",
                    str(args.manifest.expanduser().resolve()),
                    "--dataset-root",
                    str(args.dataset_root.expanduser().resolve()),
                    "--config",
                    str(args.config.expanduser().resolve()),
                    "--weights",
                    str(args.weights.expanduser().resolve()),
                    "--split",
                    "validation",
                    "--sequence",
                    sequence,
                    "--output",
                    str(reportPath),
                    "--policy",
                    policy,
                    "--view-strategy",
                    strategy,
                    "--quiet",
                    "--spherical-samples-yaw",
                    str(args.spherical_samples_yaw),
                    "--spherical-samples-pitch",
                    str(args.spherical_samples_pitch),
                    "--progress-interval-seconds",
                    str(args.progress_interval_seconds),
                ]
                if args.max_frames is not None:
                    command.extend(("--max-frames", str(args.max_frames)))
                code = evaluateLost(command)
                if code != 0:
                    raise RuntimeError(
                        f"LOST evaluation failed: policy={policy}, strategy={strategy}, "
                        f"sequence={sequence}, code={code}"
                    )
            _printSequenceResult(completed, total, reportPath)
            reports.append(reportPath)
        aggregatePath = combinationRoot / "aggregate.json"
        summarizeEvaluations(
            [
                "--inputs",
                *[str(path) for path in reports],
                "--output",
                str(aggregatePath),
            ]
        )
        aggregate = json.loads(aggregatePath.read_text(encoding="utf-8"))
        aggregate["lostExperiment"] = _aggregateLost(reports, policy, strategy)
        aggregatePath.write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
        _printCombinationResult(policy, strategy, aggregate)
    _writeMatrix(outputRoot, combinations)
    return 0


def _isComplete(
    reportPath: Path,
    policy: str,
    strategy: str,
    maxFrames: int | None,
) -> bool:
    candidates = reportPath.with_name(f"{reportPath.stem}.candidates.jsonl")
    timings = reportPath.with_name(f"{reportPath.stem}.timings.jsonl")
    if not reportPath.is_file() or not candidates.is_file() or not timings.is_file():
        return False
    try:
        report = json.loads(reportPath.read_text(encoding="utf-8"))
        experiment = report["experiment"]
        frameCount = int(report["summary"]["frameCount"])
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False
    return bool(
        experiment.get("policy") == policy
        and experiment.get("viewStrategy") == strategy
        and (maxFrames is None or frameCount == maxFrames)
    )


def _printSequenceResult(completed: int, total: int, reportPath: Path) -> None:
    summary = json.loads(reportPath.read_text(encoding="utf-8"))["summary"]
    print(
        f"[{completed}/{total}] result {summary['sequence']} "
        f"mean_iou={float(summary['circularErpMeanIoU']):.6f} "
        f"loss_rate={float(summary['trackingLossRate']):.6f} "
        f"lost_frames={int(summary['lostSearchFrameCount'])} "
        f"lost_iou={float(summary['lostSearchMeanIoU']):.6f} "
        f"rollbacks={int(summary['rollbackEventCount'])} "
        f"replay_rate={float(summary['rollbackFrameExecutionRate']):.4f}",
        flush=True,
    )


def _printCombinationResult(policy: str, strategy: str, aggregate: dict[str, Any]) -> None:
    lost = aggregate["lostExperiment"]
    print(
        f"[aggregate] {policy}/{strategy} "
        f"mean_iou={float(aggregate['circularErpMeanIoU']):.6f} "
        f"loss_rate={float(aggregate['trackingLossRate']):.6f} "
        f"lost_iou={float(lost['lostSearchMeanIoU']):.6f} "
        f"rollbacks={int(lost['rollbackEventCount'])} "
        f"replay_rate={float(lost['rollbackFrameExecutionRate']):.4f} "
        f"absent_fpr={float(aggregate['absentFalsePositiveRate']):.6f}",
        flush=True,
    )


def _aggregateLost(reports: list[Path], policy: str, strategy: str) -> dict[str, Any]:
    summaries = [json.loads(path.read_text(encoding="utf-8"))["summary"] for path in reports]
    logicalFrames = sum(max(0, int(item["frameCount"]) - 1) for item in summaries)
    lostVisible = sum(int(item["lostSearchVisibleFrameCount"]) for item in summaries)
    lostIouSum = sum(
        float(item["lostSearchMeanIoU"]) * int(item["lostSearchVisibleFrameCount"])
        for item in summaries
    )
    lostZeroSum = sum(
        float(item["lostSearchZeroIoURate"]) * int(item["lostSearchVisibleFrameCount"])
        for item in summaries
    )
    lostSuccessSum = sum(
        float(item["lostSearchSuccessRateAt0.5"])
        * int(item["lostSearchVisibleFrameCount"])
        for item in summaries
    )
    replayed = sum(int(item["replayedFrameExecutions"]) for item in summaries)
    return {
        "policy": policy,
        "viewStrategy": strategy,
        "logicalFrameCount": logicalFrames,
        "lostSearchFrameCount": sum(int(item["lostSearchFrameCount"]) for item in summaries),
        "lostSearchVisibleFrameCount": lostVisible,
        "lostSearchMeanIoU": lostIouSum / lostVisible if lostVisible else 0.0,
        "lostSearchZeroIoURate": lostZeroSum / lostVisible if lostVisible else 0.0,
        "lostSearchSuccessRateAt0.5": lostSuccessSum / lostVisible if lostVisible else 0.0,
        "finalLostStatusFrameCount": sum(
            int(item["finalLostStatusFrameCount"]) for item in summaries
        ),
        "rollbackEventCount": sum(int(item["rollbackEventCount"]) for item in summaries),
        "replayedFrameExecutions": replayed,
        "rollbackFrameExecutionRate": replayed / logicalFrames if logicalFrames else 0.0,
        "executedAverageViewsPerFrame": _weightedMean(
            summaries, "executedAverageViewsPerFrame"
        ),
        "executedAverageForwardsPerFrame": _weightedMean(
            summaries, "executedAverageForwardsPerFrame"
        ),
        "perSequence": {
            str(item["sequence"]): {
                "frameCount": int(item["frameCount"]),
                "meanIoU": float(item["circularErpMeanIoU"]),
                "trackingLossRate": float(item["trackingLossRate"]),
                "lostSearchFrameCount": int(item["lostSearchFrameCount"]),
                "lostSearchMeanIoU": float(item["lostSearchMeanIoU"]),
                "rollbackEventCount": int(item["rollbackEventCount"]),
            }
            for item in summaries
        },
    }


def _weightedMean(summaries: list[dict[str, Any]], field: str) -> float:
    weights = [max(0, int(item["frameCount"]) - 1) for item in summaries]
    if not any(weights):
        return 0.0
    return float(
        np.average(
            np.asarray([float(item[field]) for item in summaries], dtype=np.float64),
            weights=np.asarray(weights, dtype=np.float64),
        )
    )


def _writeMatrix(outputRoot: Path, combinations: list[tuple[str, str]]) -> None:
    rows = []
    for policy, strategy in combinations:
        path = outputRoot / f"{policy}__{strategy}" / "aggregate.json"
        if not path.is_file():
            continue
        aggregate = json.loads(path.read_text(encoding="utf-8"))
        lost = aggregate["lostExperiment"]
        rows.append(
            {
                "policy": policy,
                "viewStrategy": strategy,
                "circularErpMeanIoU": aggregate["circularErpMeanIoU"],
                "successRateAt0.5": aggregate["successRateAt0.5"],
                "trackingLossRate": aggregate["trackingLossRate"],
                "lostFrameCount": aggregate["lostFrameCount"],
                "sphericalMeanIoU": aggregate["sphericalMeanIoU"],
                "absentFalsePositiveRate": aggregate["absentFalsePositiveRate"],
                "latencyP95Ms": aggregate["latencyP95Ms"],
                **{
                    key: lost[key]
                    for key in (
                        "lostSearchFrameCount",
                        "lostSearchMeanIoU",
                        "lostSearchZeroIoURate",
                        "rollbackEventCount",
                        "rollbackFrameExecutionRate",
                        "executedAverageViewsPerFrame",
                        "executedAverageForwardsPerFrame",
                    )
                },
            }
        )
    (outputRoot / "matrix.json").write_text(
        json.dumps(rows, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    raise SystemExit(main())
