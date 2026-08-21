"""Resumable, failure-tolerant runner for ``docs/TryingPlan.md``.

The runner deliberately keeps the single-sequence evaluator as the execution
kernel.  Every task gets a fresh runtime in that evaluator process, while this
parent owns task identity, atomic completion markers, retries, and aggregation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    # Direct ``python tools/run_trying_plan.py`` execution.
    from eval_manifest_controller import EXPERIMENT_VARIANTS, main as evaluate
except ModuleNotFoundError:  # pragma: no cover - import-as-module convenience
    from tools.eval_manifest_controller import EXPERIMENT_VARIANTS, main as evaluate
from instatarget.core.config import loadConfig
from instatarget.training.dataset import loadManifest

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SEQUENCES = (
    "train_sim/seq_0045",
    "train_sim/seq_0017",
    "train_real/seq_0018",
    "train_real/seq_0026",
    "train_real/seq_0037",
    "train_real/seq_0005",
    "train_real/seq_0036",
    "train_sim/seq_0078",
    "train_sim/seq_0010",
    "train_sim/seq_0048",
    "train_sim/seq_0076",
    "train_sim/seq_0075",
    "train_sim/seq_0036",
    "train_sim/seq_0052",
    "train_sim/seq_0058",
)
VARIANTS = tuple(item for item in EXPERIMENT_VARIANTS if item != "shared_control_production")
METRIC_FIELDS = (
    "circularErpMeanIoU",
    "sphericalMeanIoU",
    "successRateAt0.5",
    "trackingLossRate",
    "absentFalsePositiveRate",
    "meanWidthRelativeError",
    "meanHeightRelativeError",
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path(r"E:\NewDownload\train\manifest.jsonl"))
    parser.add_argument("--dataset-root", type=Path, default=Path(r"E:\NewDownload\train"))
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "configs" / "RGBonly.yaml")
    parser.add_argument("--weights", type=Path, default=REPOSITORY_ROOT / "models" / "hit_small_stage3.pth")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--reuse-baseline-root",
        type=Path,
        help="existing TryingPlan output root whose shared_control/production artifacts are reused",
    )
    parser.add_argument("--sequence", nargs="+", choices=SEQUENCES, default=list(SEQUENCES))
    parser.add_argument(
        "--allow-custom-sequences",
        action="store_true",
        help="allow a short-test sequence subset; the production plan always uses all 15 fixed sequences",
    )
    parser.add_argument("--variant", nargs="+", choices=("shared_control_production", *VARIANTS))
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--retry-count", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--only", nargs="*", help="run only these variant names; useful for short tests")
    parser.add_argument("--stop-after", type=int, help="stop after this many task attempts")
    parser.add_argument("--progress-interval-seconds", type=float, default=60.0)
    parser.add_argument(
        "--min-free-gb",
        type=float,
        default=1.0,
        help="fail before inference when the output volume has less free space",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="write/validate the run plan without starting model inference",
    )
    return parser


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _task_hash(args: argparse.Namespace, variant: str, sequence: str) -> str:
    payload = {
        "manifest": str(args.manifest.expanduser().resolve()),
        "manifestSha256": _sha256(args.manifest.expanduser().resolve()),
        "datasetRoot": str(args.dataset_root.expanduser().resolve()),
        "config": str(args.config.expanduser().resolve()),
        "configSha256": _sha256(args.config.expanduser().resolve()),
        "weights": str(args.weights.expanduser().resolve()),
        "weightsSha256": _sha256(args.weights.expanduser().resolve()),
        "split": "validation",
        "variant": variant,
        "sequence": sequence,
        "maxFrames": args.max_frames,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def _safe(sequence: str) -> str:
    return sequence.replace("/", "__").replace("\\", "__")


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _append_event(root: Path, event: dict[str, Any]) -> None:
    path = root / "events.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"time": _now(), **event}, separators=(",", ":")) + "\n")


def _finite(value: Any) -> bool:
    if isinstance(value, float):
        return value == value and abs(value) != float("inf")
    if isinstance(value, dict):
        return all(_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_finite(item) for item in value)
    return True


def _artifact_complete(path: Path) -> bool:
    required = ("report.json", "report.candidates.jsonl", "report.timings.jsonl")
    if not all((path / name).is_file() for name in required):
        return False
    try:
        report = json.loads((path / "report.json").read_text(encoding="utf-8"))
        summary = report["summary"]
        frame_count = int(summary["frameCount"])
        if frame_count < 2 or not _finite(report):
            return False
        timing_rows = [
            json.loads(line)
            for line in (path / "report.timings.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if len(timing_rows) != frame_count or not all(_finite(row) for row in timing_rows):
            return False
        for line in (path / "report.candidates.jsonl").read_text(encoding="utf-8").splitlines():
            if line.strip() and not _finite(json.loads(line)):
                return False
        return True
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
        return False


def _task_dir(root: Path, variant: str, sequence: str) -> Path:
    if variant == "shared_control_production":
        return root / "shared_control" / "production" / _safe(sequence)
    family = {
        "erp_crop_2x_strict": "test01_erp",
        "erp_crop_2x_relaxed": "test01_erp",
        "erp_crop_4x_relaxed": "test01_erp",
        "erp_crop_2x_3x_best": "test01_erp",
        "template_strict": "test02_template",
        "template_relaxed": "test02_template",
        "fusor_weighted_box": "test03_fusor",
        "fusor_robust_spherical_consensus": "test03_fusor",
        "fov_adaptive_both_rounds": "test04_fov",
        "fov_adaptive_round1_only": "test04_fov",
        "iou_refine_head": "test05_iou_refine",
        "distractor_identity_verifier": "test06_distractor_verifier",
        "local_global_recovery_verifier": "test07_local_global_recovery",
        "pipeline_only": "test08_pipeline_geometry",
        "gpu_geometry_only": "test08_pipeline_geometry",
        "pipeline_gpu_geometry": "test08_pipeline_geometry",
    }[variant]
    return root / family / variant / _safe(sequence)


def _baseline_dir(args: argparse.Namespace, sequence: str) -> Path:
    root = args.reuse_baseline_root or args.output_root
    return _task_dir(root, "shared_control_production", sequence)


def _success(path: Path, expectedHash: str) -> bool:
    marker = path / "_SUCCESS.json"
    report = path / "report.json"
    if not marker.is_file() or not report.is_file() or not _artifact_complete(path):
        return False
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
        return (
            value.get("taskHash") == expectedHash
            and value.get("artifactHash") == _artifact_hash(path)
            and json.loads(report.read_text(encoding="utf-8")).get("summary") is not None
        )
    except (OSError, ValueError, json.JSONDecodeError):
        return False


def _artifact_hash(path: Path) -> str:
    digest = hashlib.sha256()
    for name in ("report.json", "report.candidates.jsonl", "report.timings.jsonl"):
        filePath = path / name
        if not filePath.is_file():
            return ""
        digest.update(name.encode("utf-8"))
        digest.update(filePath.read_bytes())
    return digest.hexdigest()


def _write_progress(root: Path, *, total: int, completed: int, failed: int, current: str | None, started: str) -> None:
    _atomic_json(root / "progress.json", {
        "totalTasks": total,
        "completedTasks": completed,
        "failedTasks": failed,
        "currentTask": current,
        "startedAt": started,
        "lastUpdateAt": _now(),
        "remainingTasks": max(0, total - completed),
    })


def _git_commit() -> str | None:
    try:
        import subprocess

        return subprocess.run(
            ["git", "-C", str(REPOSITORY_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _ensure_plan(root: Path, args: argparse.Namespace, sequences: tuple[str, ...], requested: tuple[str, ...]) -> None:
    baselineExecutions = 0 if args.reuse_baseline_root is not None else len(sequences)
    sequencePlan = {
        "seed": "20260821-trying-plan-v1",
        "split": "validation",
        "sequences": list(sequences),
        "manifest": str(args.manifest),
        "manifestSha256": _sha256(args.manifest),
    }
    matrixVariants = ["shared_control_production", *VARIANTS]
    matrix = {
        "baseline": "shared_control_production",
        "allVariants": matrixVariants,
        "requestedVariants": list(requested),
        "sequenceCount": len(sequences),
        "plannedExecutions": baselineExecutions + len(sequences) * len(requested),
        "baselineExecutions": baselineExecutions,
        "experimentExecutions": len(sequences) * len(requested),
        "reusedBaselineRoot": (
            str(args.reuse_baseline_root) if args.reuse_baseline_root is not None else None
        ),
    }
    for name, value in (("sequence_plan.json", sequencePlan), ("experiment_matrix.json", matrix)):
        target = root / name
        if args.resume and target.is_file():
            existing = json.loads(target.read_text(encoding="utf-8"))
            if existing != value:
                raise RuntimeError(f"existing {name} does not match this run; use a new output root")
        else:
            _atomic_json(target, value)


def _preflight(args: argparse.Namespace, sequences: tuple[str, ...], requested: tuple[str, ...]) -> None:
    if args.min_free_gb < 0:
        raise ValueError("--min-free-gb must be non-negative")
    for path, label in ((args.manifest, "manifest"), (args.config, "config"), (args.weights, "weights")):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    if not args.dataset_root.is_dir():
        raise FileNotFoundError(f"dataset root does not exist: {args.dataset_root}")
    config = loadConfig(args.config)
    calibration = config.scoring.calibrationArtifact
    if calibration is None or not calibration.is_file():
        raise FileNotFoundError(f"calibration artifact does not exist: {calibration}")
    if args.output_root.exists() and not args.output_root.is_dir():
        raise RuntimeError(f"output root is not a directory: {args.output_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)
    freeBytes = shutil.disk_usage(args.output_root).free
    if freeBytes < args.min_free_gb * 1024**3:
        raise RuntimeError(f"output volume has {freeBytes / 1024**3:.2f} GiB free, below --min-free-gb={args.min_free_gb}")
    if not sequences:
        raise RuntimeError("at least one validation sequence is required")
    _ensure_plan(args.output_root, args, sequences, requested)
    if args.reuse_baseline_root is not None:
        for sequence in sequences:
            directory = _baseline_dir(args, sequence)
            expectedHash = _task_hash(args, "shared_control_production", sequence)
            if not _success(directory, expectedHash):
                raise RuntimeError(
                    f"reusable shared baseline is missing or stale for {sequence}: {directory}"
                )
    runManifest = {
        "createdAt": _now(),
        "repositoryRoot": str(REPOSITORY_ROOT),
        "gitCommit": _git_commit(),
        "gitDirty": bool(os.system(f'git -C "{REPOSITORY_ROOT}" diff --quiet') != 0),
        "python": sys.version,
        "platform": platform.platform(),
        "manifest": str(args.manifest),
        "manifestSha256": _sha256(args.manifest),
        "config": str(args.config),
        "configSha256": _sha256(args.config),
        "weights": str(args.weights),
        "weightsSha256": _sha256(args.weights),
        "split": "validation",
        "sequences": list(sequences),
        "variants": ["shared_control_production", *requested],
        "plannedExecutions": (
            (0 if args.reuse_baseline_root is not None else len(sequences))
            + len(sequences) * len(requested)
        ),
        "reusedBaselineRoot": (
            str(args.reuse_baseline_root) if args.reuse_baseline_root is not None else None
        ),
    }
    runPath = args.output_root / "run_manifest.json"
    if args.resume and runPath.is_file():
        previous = json.loads(runPath.read_text(encoding="utf-8"))
        for key in ("manifestSha256", "configSha256", "weightsSha256", "sequences", "variants"):
            if previous.get(key) != runManifest.get(key):
                raise RuntimeError(f"resume fingerprint mismatch in run_manifest.json: {key}")
    else:
        _atomic_json(runPath, runManifest)


def _run_one(args: argparse.Namespace, variant: str, sequence: str, taskHash: str) -> tuple[bool, str]:
    directory = _task_dir(args.output_root.expanduser().resolve(), variant, sequence)
    directory.mkdir(parents=True, exist_ok=True)
    reportPath = directory / "report.json"
    command = [
        "--manifest", str(args.manifest.expanduser().resolve()),
        "--dataset-root", str(args.dataset_root.expanduser().resolve()),
        "--config", str(args.config.expanduser().resolve()),
        "--weights", str(args.weights.expanduser().resolve()),
        "--split", "validation", "--sequence", sequence, "--output", str(reportPath),
        "--variant", variant, "--spherical-samples-yaw", "128", "--spherical-samples-pitch", "64", "--quiet",
    ]
    if args.max_frames is not None:
        command.extend(("--max-frames", str(args.max_frames)))
    try:
        code = evaluate(command)
        if code != 0:
            raise RuntimeError(f"evaluator returned {code}")
        if not _artifact_complete(directory):
            raise RuntimeError("evaluator produced incomplete or non-finite artifacts")
        artifactHash = _artifact_hash(directory)
        if not artifactHash:
            raise RuntimeError("evaluator did not produce all atomic artifacts")
        _atomic_json(directory / "_SUCCESS.json", {
            "taskHash": taskHash,
            "artifactHash": artifactHash,
            "variant": variant,
            "sequence": sequence,
            "frameCount": json.loads(reportPath.read_text(encoding="utf-8"))["summary"]["frameCount"],
        })
        if variant != "shared_control_production":
            baselineDirectory = _baseline_dir(args, sequence)
            baselineMarker = baselineDirectory / "_SUCCESS.json"
            if baselineMarker.is_file():
                baseline = json.loads(baselineMarker.read_text(encoding="utf-8"))
                candidateSummary = json.loads(reportPath.read_text(encoding="utf-8"))["summary"]
                baselineSummary = _loadSummary(baselineDirectory) or {}
                timingRatio = {}
                for key in ("latencyP50Ms", "latencyP95Ms", "latencyP99Ms"):
                    baseValue = float(baselineSummary.get(key, 0.0) or 0.0)
                    candValue = float(candidateSummary.get(key, 0.0) or 0.0)
                    timingRatio[key] = (
                        (baseValue - candValue) / baseValue if baseValue > 1e-12 else None
                    )
                _atomic_json(directory / "comparison.json", {
                    "variant": variant,
                    "sequence": sequence,
                    "baselinePath": str(baselineDirectory),
                    "baselineArtifactHash": baseline.get("artifactHash"),
                    "baselineTaskHash": baseline.get("taskHash"),
                    "candidateArtifactHash": artifactHash,
                    "timingImprovementRatio": timingRatio,
                })
        return True, artifactHash
    except Exception as error:
        _atomic_json(directory / "_FAILED.json", {
            "taskHash": taskHash,
            "variant": variant,
            "sequence": sequence,
            "error": repr(error),
            "traceback": traceback.format_exc(),
        })
        return False, ""


def _loadSummary(path: Path) -> dict[str, Any] | None:
    try:
        if not (path / "_SUCCESS.json").is_file() or not _artifact_complete(path):
            return None
        return json.loads((path / "report.json").read_text(encoding="utf-8"))["summary"]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _writeAggregates(
    root: Path,
    variants: tuple[str, ...],
    sequences: tuple[str, ...],
    baseline: dict[str, dict[str, Any]],
    baselineRoot: Path | None = None,
) -> None:
    rows: list[dict[str, Any]] = []
    for variant in ("shared_control_production", *variants):
        summaries = [
            _loadSummary(
                _task_dir(baselineRoot or root, variant, sequence)
                if variant == "shared_control_production"
                else _task_dir(root, variant, sequence)
            )
            for sequence in sequences
        ]
        complete = all(item is not None for item in summaries)
        reports = []
        for sequence in sequences:
            try:
                reportDirectory = (
                    _task_dir(baselineRoot or root, variant, sequence)
                    if variant == "shared_control_production"
                    else _task_dir(root, variant, sequence)
                )
                reports.append(json.loads((reportDirectory / "report.json").read_text(encoding="utf-8")))
            except (OSError, ValueError, json.JSONDecodeError):
                pass
        implemented = variant == "shared_control_production" or (
            bool(reports)
            and all(report.get("experiment", {}).get("variantImplemented") is True for report in reports)
        )
        row: dict[str, Any] = {
            "variant": variant,
            "complete": complete,
            "implemented": implemented,
            "eligible": bool(complete and implemented),
            "sequenceCount": sum(item is not None for item in summaries),
        }
        for field in METRIC_FIELDS:
            values = [float(item[field]) for item in summaries if item is not None and field in item]
            row[field] = sum(values) / len(values) if values else None
        if variant != "shared_control_production":
            comparisons = []
            for sequence, summary in zip(sequences, summaries, strict=True):
                control = baseline.get(sequence)
                if summary is None or control is None:
                    continue
                comparisons.append({
                    "sequence": sequence,
                    **{field: float(summary[field]) - float(control[field]) for field in METRIC_FIELDS if field in summary and field in control},
                })
            row["perSequenceDelta"] = comparisons
        rows.append(row)
        family = "shared_control" if variant == "shared_control_production" else {
            "erp_crop_2x_strict": "test01_erp",
            "erp_crop_2x_relaxed": "test01_erp",
            "erp_crop_4x_relaxed": "test01_erp",
            "erp_crop_2x_3x_best": "test01_erp",
            "template_strict": "test02_template",
            "template_relaxed": "test02_template",
            "fusor_weighted_box": "test03_fusor",
            "fusor_robust_spherical_consensus": "test03_fusor",
            "fov_adaptive_both_rounds": "test04_fov",
            "fov_adaptive_round1_only": "test04_fov",
            "iou_refine_head": "test05_iou_refine",
            "distractor_identity_verifier": "test06_distractor_verifier",
            "local_global_recovery_verifier": "test07_local_global_recovery",
            "pipeline_only": "test08_pipeline_geometry",
            "gpu_geometry_only": "test08_pipeline_geometry",
            "pipeline_gpu_geometry": "test08_pipeline_geometry",
        }[variant]
        _atomic_json(root / family / "aggregate.json", row)
    final = root / "final"
    _atomic_json(final / "overall_matrix.json", rows)
    lines = ["variant,complete,implemented,eligible,sequenceCount," + ",".join(METRIC_FIELDS)]
    for row in rows:
        lines.append(",".join(str(row.get(key, "")) for key in ("variant", "complete", "implemented", "eligible", "sequenceCount", *METRIC_FIELDS)))
    (final / "overall_comparison.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    recommendation = "# TryingPlan recommendation\n\n"
    recommendation += "Only complete and implemented variants are eligible. Runtime and memory are diagnostic only.\n\n"
    recommendation += "The production baseline is shared across every variant; no variant reruns it.\n"
    recommendation += "\nA variant is complete only when all selected sequences have finite report, candidate, timing, and SUCCESS artifacts. Missing baselines or failed retries keep the comparison incomplete.\n"
    recommendation += "\nVariant implementation status is recorded in each evaluator report; fallback variants must not be promoted without a follow-up implementation review.\n"
    (final / "final_recommendation.md").write_text(recommendation, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    args.manifest = args.manifest.expanduser().resolve()
    args.dataset_root = args.dataset_root.expanduser().resolve()
    args.config = args.config.expanduser().resolve()
    args.weights = args.weights.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    if args.reuse_baseline_root is not None:
        args.reuse_baseline_root = args.reuse_baseline_root.expanduser().resolve()
    if not args.manifest.is_relative_to(args.dataset_root):
        raise RuntimeError("manifest must be inside dataset-root")
    if args.retry_count < 1:
        raise ValueError("retry-count must be positive")
    requested = tuple(item for item in (args.variant or VARIANTS) if item != "shared_control_production")
    if args.only:
        requested = tuple(item for item in requested if item in set(args.only))
    records = tuple(loadManifest(args.manifest))
    available = {record.sequenceId for record in records if record.split == "validation"}
    missing = [sequence for sequence in args.sequence if sequence not in available]
    if missing:
        raise RuntimeError(f"validation sequences missing from manifest: {missing}")
    sequences = tuple(args.sequence)
    if sequences != SEQUENCES and not args.allow_custom_sequences:
        raise RuntimeError("TryingPlan requires the fixed 15 validation sequences; use --allow-custom-sequences only for a short test")
    root = args.output_root
    root.mkdir(parents=True, exist_ok=True)
    _preflight(args, sequences, requested)
    baselineTaskCount = 0 if args.reuse_baseline_root is not None else len(sequences)
    total = baselineTaskCount + len(sequences) * len(requested)
    if args.dry_run:
        _atomic_json(root / "progress.json", {
            "totalTasks": total,
            "completedTasks": 0,
            "failedTasks": 0,
            "currentTask": None,
            "startedAt": _now(),
            "lastUpdateAt": _now(),
            "remainingTasks": total,
        })
        print(f"plan valid: {total} tasks; no inference started", flush=True)
        return 0
    baselineHashes: dict[str, str] = {}
    baselineSummaries: dict[str, dict[str, Any]] = {}
    attempts = 0
    completed = 0
    failed = 0
    started = _now()
    _write_progress(root, total=total, completed=0, failed=0, current=None, started=started)
    _append_event(root, {"event": "run_started", "totalTasks": total, "variants": ["shared_control_production", *requested]})
    if args.reuse_baseline_root is not None:
        for sequence in sequences:
            directory = _baseline_dir(args, sequence)
            marker = json.loads((directory / "_SUCCESS.json").read_text(encoding="utf-8"))
            baselineHashes[sequence] = marker.get("artifactHash", "")
            summary = _loadSummary(directory)
            if summary is not None:
                baselineSummaries[sequence] = summary
    scheduled = [
        *([] if args.reuse_baseline_root is not None else [("shared_control_production", sequences)]),
        *[(variant, sequences) for variant in requested],
    ]
    for variant, taskSequences in scheduled:
        for sequence in taskSequences:
            attempts += 1
            taskHash = _task_hash(args, variant, sequence)
            directory = _task_dir(root, variant, sequence)
            _write_progress(root, total=total, completed=completed, failed=failed, current=f"{variant}:{sequence}", started=started)
            if args.resume and _success(directory, taskHash):
                artifactHash = json.loads((directory / "_SUCCESS.json").read_text(encoding="utf-8")).get("artifactHash", "")
                if variant != "shared_control_production":
                    baselineMarker = _baseline_dir(args, sequence) / "_SUCCESS.json"
                    if not baselineMarker.is_file():
                        print(f"[warning] {variant} {sequence}: shared baseline is missing; comparison will be incomplete", flush=True)
                        artifactHash = ""
                    else:
                        baseline = json.loads(baselineMarker.read_text(encoding="utf-8"))
                        comparison = directory / "comparison.json"
                        if not comparison.is_file() or json.loads(comparison.read_text(encoding="utf-8")).get("baselineArtifactHash") != baseline.get("artifactHash"):
                            print(f"[stale] {variant} {sequence}; rerunning because baseline hash changed", flush=True)
                            artifactHash = ""
                        else:
                            print(f"[resume {completed + 1}/{total}] {variant} {sequence}", flush=True)
                else:
                    print(f"[resume {completed + 1}/{total}] {variant} {sequence}", flush=True)
            else:
                artifactHash = ""
            if not artifactHash:
                for retry in range(1, args.retry_count + 1):
                    print(f"[run {completed + 1}/{total}] {variant} {sequence} attempt={retry}", flush=True)
                    _append_event(root, {"event": "task_started", "variant": variant, "sequence": sequence, "attempt": retry})
                    ok, artifactHash = _run_one(args, variant, sequence, taskHash)
                    if ok:
                        _append_event(root, {"event": "task_succeeded", "variant": variant, "sequence": sequence, "attempt": retry, "artifactHash": artifactHash})
                        break
                    failed += 1
                    _append_event(root, {"event": "task_failed", "variant": variant, "sequence": sequence, "attempt": retry})
                    print(f"[failed] {variant} {sequence}; continuing", flush=True)
            if variant == "shared_control_production":
                baselineHashes[sequence] = artifactHash
                summary = _loadSummary(directory)
                if summary is not None:
                    baselineSummaries[sequence] = summary
            completed += 1
            _write_progress(root, total=total, completed=completed, failed=failed, current=None, started=started)
            if args.stop_after is not None and attempts >= args.stop_after:
                _writeAggregates(root, requested, sequences, baselineSummaries, args.reuse_baseline_root)
                _append_event(root, {"event": "run_stopped", "reason": "stop_after", "completedTasks": completed})
                return 0
    for sequence in sequences:
        if not baselineHashes.get(sequence):
            print(f"[warning] missing shared baseline for {sequence}; comparisons remain incomplete", flush=True)
    _writeAggregates(root, requested, sequences, baselineSummaries, args.reuse_baseline_root)
    _write_progress(root, total=total, completed=completed, failed=failed, current=None, started=started)
    _append_event(root, {"event": "run_finished", "completedTasks": completed, "failedAttempts": failed})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
