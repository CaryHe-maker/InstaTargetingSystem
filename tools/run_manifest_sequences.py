"""Run manifest-backed tracking for selected or all sequences.

This is the user-facing batch wrapper around ``eval_manifest_controller``.  It
keeps one runtime per sequence and writes both JSON reports and a compact
competition-style ``result.txt`` summary under the artifact directory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from eval_manifest_controller import main as evaluateSequence
from summarize_manifest_evaluations import main as summarizeEvaluations

from instatarget.training.dataset import loadManifest

DEFAULT_DATASET_ROOT = Path(r"E:\NewDownload\train")
DEFAULT_MANIFEST = DEFAULT_DATASET_ROOT / "manifest.jsonl"
DEFAULT_CONFIG = Path("configs/RGBonly.yaml")
DEFAULT_WEIGHTS = Path("models/hit_small_stage3.pth")


def buildParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument(
        "--split",
        choices=("train", "validation", "calibration", "holdout", "all"),
        default="train",
        help="Manifest split to run; use all for every split (holdout needs --allow-holdout).",
    )
    selected = parser.add_mutually_exclusive_group(required=True)
    selected.add_argument("--sequence", nargs="+", help="One or more sequence IDs.")
    selected.add_argument("--all", action="store_true", help="Run every sequence in --split.")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Write midVisual/backend_box and midVisual/geometry_box; evaluation remains enabled.",
    )
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--precision", choices=("fp32", "fp16"))
    parser.add_argument("--cudnn-benchmark", action="store_true")
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--reuse-buffers", action="store_true")
    parser.add_argument("--pinned-nonblocking", action="store_true")
    parser.add_argument("--spherical-samples-yaw", type=int, default=128)
    parser.add_argument("--spherical-samples-pitch", type=int, default=64)
    parser.add_argument("--allow-holdout", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = buildParser().parse_args(argv)
    if args.split == "holdout" and not args.allow_holdout:
        raise RuntimeError("holdout evaluation requires --allow-holdout after model freeze")
    datasetRoot = args.dataset_root.expanduser().resolve()
    manifestPath = args.manifest.expanduser().resolve()
    if not manifestPath.is_relative_to(datasetRoot):
        raise RuntimeError(
            f"manifest must be inside the canonical dataset root {datasetRoot}: {manifestPath}"
        )
    records = loadManifest(manifestPath)
    available = sorted(
        {
            record.sequenceId
            for record in records
            if args.split == "all" or record.split == args.split
        }
    )
    if args.sequence:
        unknown = sorted(set(args.sequence) - set(available))
        if unknown:
            raise RuntimeError(
                f"sequence(s) not present in split={args.split}: {', '.join(unknown)}"
            )
        sequenceIds = list(dict.fromkeys(args.sequence))
    else:
        sequenceIds = available
    if not sequenceIds:
        raise RuntimeError(f"manifest has no sequences in split={args.split}")

    splitBySequence = {
        sequenceId: next(
            record.split
            for record in records
            if record.sequenceId == sequenceId
            and (args.split == "all" or record.split == args.split)
        )
        for sequenceId in sequenceIds
    }
    if any(split == "holdout" for split in splitBySequence.values()) and not args.allow_holdout:
        raise RuntimeError(
            "the selected sequences include holdout; rerun after model freeze with --allow-holdout"
        )

    outputRoot = args.output_dir.expanduser().resolve()
    reportsRoot = outputRoot / "evaluations"
    reportsRoot.mkdir(parents=True, exist_ok=True)
    reportPaths: list[Path] = []
    for index, sequenceId in enumerate(sequenceIds, 1):
        safeName = sequenceId.replace("/", "__").replace("\\", "__")
        reportPath = reportsRoot / f"{safeName}.json"
        command = [
            "--manifest",
            str(manifestPath),
            "--dataset-root",
            str(datasetRoot),
            "--config",
            str(args.config.expanduser().resolve()),
            "--weights",
            str(args.weights.expanduser().resolve()),
            "--split",
            splitBySequence[sequenceId],
            "--sequence",
            sequenceId,
            "--output",
            str(reportPath),
            "--spherical-samples-yaw",
            str(args.spherical_samples_yaw),
            "--spherical-samples-pitch",
            str(args.spherical_samples_pitch),
        ]
        if args.max_frames is not None:
            command.extend(("--max-frames", str(args.max_frames)))
        if args.profile:
            command.append("--profile")
        if args.precision is not None:
            command.extend(("--precision", args.precision))
        for enabled, option in (
            (args.cudnn_benchmark, "--cudnn-benchmark"),
            (args.channels_last, "--channels-last"),
            (args.reuse_buffers, "--reuse-buffers"),
            (args.pinned_nonblocking, "--pinned-nonblocking"),
        ):
            if enabled:
                command.append(option)
        if args.allow_holdout:
            command.append("--allow-holdout")
        if args.visualize:
            command.extend(
                (
                    "--visual-output-root",
                    str(outputRoot / "midVisual"),
                    "--result-visual-root",
                    str(outputRoot / "resultVisual"),
                )
            )
        print(f"[{index}/{len(sequenceIds)}] {sequenceId}", flush=True)
        code = evaluateSequence(command)
        if code != 0:
            raise RuntimeError(f"evaluation failed for {sequenceId} with exit code {code}")
        reportPaths.append(reportPath)

    aggregatePath = outputRoot / "aggregate.json"
    summaryArguments = [
        "--inputs",
        *[str(path) for path in reportPaths],
        "--output",
        str(aggregatePath),
    ]
    if args.split == "all":
        summaryArguments.append("--allow-mixed-splits")
    summarizeEvaluations(summaryArguments)
    aggregate = json.loads(aggregatePath.read_text(encoding="utf-8"))
    resultPath = outputRoot / "result.txt"
    resultPath.write_text(_formatResult(aggregate, reportPaths), encoding="utf-8")
    print(f"result.txt: {resultPath}")
    return 0


def _formatResult(summary: dict[str, Any], reportPaths: list[Path]) -> str:
    timingRows: list[dict[str, Any]] = []
    for reportPath in reportPaths:
        timingPath = reportPath.with_name(f"{reportPath.stem}.timings.jsonl")
        with timingPath.open("r", encoding="utf-8") as stream:
            timingRows.extend(json.loads(line) for line in stream if line.strip())
    evaluatedTiming = [
        float(row["totalProcessingMs"]) for row in timingRows if int(row["frameIndex"]) != 0
    ]
    average = sum(evaluatedTiming) / len(evaluatedTiming) if evaluatedTiming else 0.0
    lines = [
        "InstaTarget manifest competition-style evaluation",
        f"split: {summary['split']}",
        f"sequence_count: {summary['sequenceCount']}",
        f"evaluated_visible_frames: {summary['evaluatedVisibleFrames']}",
        f"mean_iou (final IoU): {summary['circularErpMeanIoU']:.6f}",
        f"success_auc (AUC): {summary['successAUC']:.6f}",
        f"success_rate@0.5 (Success Rate): {summary['successRateAt0.5']:.6f}",
        f"tracking_loss_rate: {summary['trackingLossRate']:.6f}",
        f"lost_frame_count: {summary['lostFrameCount']}",
        f"per_frame_runtime_ms_mean: {average:.6f}",
        f"per_frame_runtime_ms_p50: {summary['latencyP50Ms']:.6f}",
        f"per_frame_runtime_ms_p95: {summary['latencyP95Ms']:.6f}",
        f"per_frame_runtime_ms_p99: {summary['latencyP99Ms']:.6f}",
        f"valid_rate: {summary.get('validRate', 0.0):.6f}",
        "",
        "Per-frame timing rows are in evaluations/*.timings.jsonl.",
    ]
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    raise SystemExit(main())
