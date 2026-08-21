# ruff: noqa: E501
"""Compare legacy and current production evaluation reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

SEQUENCES = (
    "train_real/seq_0005",
    "train_real/seq_0018",
    "train_real/seq_0036",
    "train_real/seq_0037",
    "train_sim/seq_0010",
    "train_sim/seq_0017",
    "train_sim/seq_0036",
    "train_sim/seq_0045",
    "train_sim/seq_0048",
    "train_sim/seq_0052",
)

REPORT_FILES = ("report.json", "report.candidates.jsonl", "report.timings.jsonl")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-root", required=True, type=Path)
    parser.add_argument("--new-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--weights", required=True, type=Path)
    return parser


def _directory(root: Path, sequence: str) -> Path:
    return root / sequence.replace("/", "__")


def _load_report(root: Path, sequence: str) -> dict[str, Any]:
    directory = _directory(root, sequence)
    for name in REPORT_FILES:
        path = directory / name
        if not path.is_file() or path.stat().st_size == 0:
            raise RuntimeError(f"missing or empty artifact: {path}")
    report = json.loads((directory / "report.json").read_text(encoding="utf-8"))
    if report.get("format") != "instatarget.manifest-controller-eval.v1":
        raise RuntimeError(f"unexpected report format: {directory}")
    summary = report["summary"]
    if summary["sequence"] != sequence:
        raise RuntimeError(f"sequence mismatch: {directory}")
    for key, value in summary.items():
        if isinstance(value, float) and not math.isfinite(value):
            raise RuntimeError(f"non-finite summary value: {sequence}.{key}")
    return report


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_hash(directory: Path) -> str:
    digest = hashlib.sha256()
    for name in REPORT_FILES:
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        with (directory / name).open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _write_success(
    root: Path,
    sequence: str,
    report: dict[str, Any],
    *,
    config_hash: str,
    weights_hash: str,
) -> None:
    directory = _directory(root, sequence)
    task = json.dumps(
        {
            "variant": "shared_control_production2",
            "sequence": sequence,
            "configSha256": config_hash,
            "weightsSha256": weights_hash,
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    payload = {
        "taskHash": hashlib.sha256(task).hexdigest(),
        "artifactHash": _artifact_hash(directory),
        "variant": "shared_control_production2",
        "sequence": sequence,
        "frameCount": int(report["summary"]["frameCount"]),
    }
    (directory / "_SUCCESS.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def _pct(value: float) -> str:
    return f"{100.0 * value:.2f}%"


def _number(value: float) -> str:
    return f"{value:.4f}"


def _delta(value: float) -> str:
    return f"{value:+.4f}"


def _label(sequence: str) -> str:
    return sequence.replace("train_", "").replace("/seq_", "-")


def _macro(rows: list[dict[str, Any]], side: str, metric: str) -> float:
    return sum(float(row[side][metric]) for row in rows) / len(rows)


def _weighted(rows: list[dict[str, Any]], side: str, metric: str) -> float:
    numerator = sum(
        float(row[side][metric]) * int(row[side]["evaluatedVisibleFrames"])
        for row in rows
    )
    denominator = sum(int(row[side]["evaluatedVisibleFrames"]) for row in rows)
    return numerator / denominator


def main() -> int:
    args = _parser().parse_args()
    config_hash = _sha256(args.config)
    weights_hash = _sha256(args.weights)
    rows: list[dict[str, Any]] = []
    old_weights: set[str] = set()
    new_weights: set[str] = set()
    for sequence in SEQUENCES:
        old_report = _load_report(args.old_root, sequence)
        new_report = _load_report(args.new_root, sequence)
        old = old_report["summary"]
        new = new_report["summary"]
        if int(old["frameCount"]) != int(new["frameCount"]):
            raise RuntimeError(f"frame count mismatch: {sequence}")
        rows.append({"sequence": sequence, "old": old, "new": new})
        old_weights.add(str(old["weights"]))
        new_weights.add(str(new["weights"]))
        _write_success(
            args.new_root,
            sequence,
            new_report,
            config_hash=config_hash,
            weights_hash=weights_hash,
        )

    iou_better = sum(row["new"]["circularErpMeanIoU"] > row["old"]["circularErpMeanIoU"] for row in rows)
    spherical_better = sum(row["new"]["sphericalMeanIoU"] > row["old"]["sphericalMeanIoU"] for row in rows)
    auc_better = sum(row["new"]["successAUC"] > row["old"]["successAUC"] for row in rows)
    loss_better = sum(row["new"]["trackingLossRate"] < row["old"]["trackingLossRate"] for row in rows)
    p50_faster = sum(row["new"]["latencyP50Ms"] < row["old"]["latencyP50Ms"] for row in rows)

    lines = [
        "# Production 与 Production2 结果对比",
        "",
        "## 对比范围",
        "",
        f"- 旧版：`{args.old_root}`",
        f"- 当前版：`{args.new_root}`",
        "- 当前代码基线：`main` 的 PostTrainingV1.4 配置要求，并包含本分支 GPU Geometry/Pipeline 与本次本地稳定性修复。",
        f"- 当前配置 SHA-256：`{config_hash}`",
        f"- 当前权重 SHA-256：`{weights_hash}`",
        f"- 旧报告权重路径：`{'`, `'.join(sorted(old_weights))}`",
        f"- 新报告权重路径：`{'`, `'.join(sorted(new_weights))}`",
        "- 10 个 sequence 的帧数逐项一致；每个新目录均包含 `report.json`、`report.candidates.jsonl`、`report.timings.jsonl` 与 `_SUCCESS.json`。",
        "",
        "> 注意：旧版报告记录的是 `hit_small_stage3.pth`，当前版严格按 main/PostTrainingV1.4 使用 `hit_small_stage3_inference.pth` 与哈希绑定校准文件。因此这里比较的是完整生产版本（代码 + 生产模型产物），不是只替换代码的单变量实验。",
        "",
        "## 总体结论",
        "",
        f"- ERP Mean IoU：{iou_better}/10 个序列提高；宏平均 `{_number(_macro(rows, 'old', 'circularErpMeanIoU'))}` → `{_number(_macro(rows, 'new', 'circularErpMeanIoU'))}`（{_delta(_macro(rows, 'new', 'circularErpMeanIoU') - _macro(rows, 'old', 'circularErpMeanIoU'))}）。",
        f"- Spherical Mean IoU：{spherical_better}/10 个序列提高；宏平均 `{_number(_macro(rows, 'old', 'sphericalMeanIoU'))}` → `{_number(_macro(rows, 'new', 'sphericalMeanIoU'))}`（{_delta(_macro(rows, 'new', 'sphericalMeanIoU') - _macro(rows, 'old', 'sphericalMeanIoU'))}）。",
        f"- Success AUC：{auc_better}/10 个序列提高；宏平均 `{_number(_macro(rows, 'old', 'successAUC'))}` → `{_number(_macro(rows, 'new', 'successAUC'))}`（{_delta(_macro(rows, 'new', 'successAUC') - _macro(rows, 'old', 'successAUC'))}）。",
        f"- Tracking Loss Rate：{loss_better}/10 个序列降低；宏平均 `{_pct(_macro(rows, 'old', 'trackingLossRate'))}` → `{_pct(_macro(rows, 'new', 'trackingLossRate'))}`。",
        f"- P50 延迟：{p50_faster}/10 个序列更快；宏平均 `{_macro(rows, 'old', 'latencyP50Ms'):.1f} ms` → `{_macro(rows, 'new', 'latencyP50Ms'):.1f} ms`，降低 `{100.0 * (1.0 - _macro(rows, 'new', 'latencyP50Ms') / _macro(rows, 'old', 'latencyP50Ms')):.1f}%`。",
        f"- 按可见帧数加权的 ERP Mean IoU：`{_number(_weighted(rows, 'old', 'circularErpMeanIoU'))}` → `{_number(_weighted(rows, 'new', 'circularErpMeanIoU'))}`（{_delta(_weighted(rows, 'new', 'circularErpMeanIoU') - _weighted(rows, 'old', 'circularErpMeanIoU'))}）。",
        "",
        "## 精度与丢失率（新值后的括号为相对旧版的绝对差）",
        "",
        "| Sequence | ERP IoU 旧→新 (Δ) | Spherical IoU 旧→新 (Δ) | Success AUC 旧→新 (Δ) | Success@0.5 旧→新 (Δ) | Loss Rate 旧→新 (Δ) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        old, new = row["old"], row["new"]
        lines.append(
            "| "
            + _label(row["sequence"])
            + " | "
            + f"{_number(old['circularErpMeanIoU'])} → {_number(new['circularErpMeanIoU'])} ({_delta(new['circularErpMeanIoU'] - old['circularErpMeanIoU'])})"
            + " | "
            + f"{_number(old['sphericalMeanIoU'])} → {_number(new['sphericalMeanIoU'])} ({_delta(new['sphericalMeanIoU'] - old['sphericalMeanIoU'])})"
            + " | "
            + f"{_number(old['successAUC'])} → {_number(new['successAUC'])} ({_delta(new['successAUC'] - old['successAUC'])})"
            + " | "
            + f"{_pct(old['successRateAt0.5'])} → {_pct(new['successRateAt0.5'])} ({100.0 * (new['successRateAt0.5'] - old['successRateAt0.5']):+.2f} pp)"
            + " | "
            + f"{_pct(old['trackingLossRate'])} → {_pct(new['trackingLossRate'])} ({100.0 * (new['trackingLossRate'] - old['trackingLossRate']):+.2f} pp)"
            + " |"
        )

    lines.extend(
        [
            "",
            "## 有效率、Absent FPR 与性能",
            "",
            "| Sequence | Valid Rate 旧→新 | Absent FPR 旧→新 | P50 ms 旧→新 | P95 ms 旧→新 | P99 ms 旧→新 |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        old, new = row["old"], row["new"]
        lines.append(
            "| "
            + _label(row["sequence"])
            + " | "
            + f"{_pct(old['validRate'])} → {_pct(new['validRate'])}"
            + " | "
            + f"{_pct(old['absentFalsePositiveRate'])} → {_pct(new['absentFalsePositiveRate'])}"
            + " | "
            + f"{old['latencyP50Ms']:.1f} → {new['latencyP50Ms']:.1f}"
            + " | "
            + f"{old['latencyP95Ms']:.1f} → {new['latencyP95Ms']:.1f}"
            + " | "
            + f"{old['latencyP99Ms']:.1f} → {new['latencyP99Ms']:.1f}"
            + " |"
        )

    biggest_gain = max(rows, key=lambda row: row["new"]["circularErpMeanIoU"] - row["old"]["circularErpMeanIoU"])
    biggest_drop = min(rows, key=lambda row: row["new"]["circularErpMeanIoU"] - row["old"]["circularErpMeanIoU"])
    lines.extend(
        [
            "",
            "## 解释与建议",
            "",
            f"- ERP IoU 最大提升是 `{_label(biggest_gain['sequence'])}`：{_delta(biggest_gain['new']['circularErpMeanIoU'] - biggest_gain['old']['circularErpMeanIoU'])}。",
            f"- ERP IoU 最大下降是 `{_label(biggest_drop['sequence'])}`：{_delta(biggest_drop['new']['circularErpMeanIoU'] - biggest_drop['old']['circularErpMeanIoU'])}。",
            "- 当前版速度提升稳定且显著，但精度收益并不一致；不能只依据总体延迟优势认定所有场景均优于旧版。",
            "- 对精度下降较大的 sequence，建议下一步结合 `frames` 与 `report.candidates.jsonl` 定位首次漂移帧，并分别检查 GPU Geometry 投影、provisional Round-2 预测和无效回退状态。",
            "- 本轮运行中发现并已本地修复两类稳定性问题：CUDA runtime 未完整释放，以及 provisional FOV 越过 PostTrainingV1.4 合法边界。修复未提交远端。",
            "",
        ]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
