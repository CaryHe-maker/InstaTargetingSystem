"""Short user-facing commands for AirSim360 tracking and instance discovery."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from instatarget.app.track_airsim360 import main as trackAirSim360Main
from instatarget.core.errors import InstaTargetError
from instatarget.data.airsim360_source import AirSim360DataSource

REPOSITORY_ROOT = Path(__file__).resolve().parents[3]


def runMain(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="run",
        description="Run one AirSim360 sequence with the RGB-only ARTrackV2 pipeline.",
    )
    parser.add_argument("-RGB_only", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("data")
    parser.add_argument("output")
    parser.add_argument("instance_id", type=int)
    parser.add_argument("--config", default=None)
    parser.add_argument("--sequence", default=None)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--no-mid-visual", action="store_true")
    parser.add_argument("--no-result-visual", action="store_true")
    args = parser.parse_args(argv)

    dataRoot = _resolveUserPath(args.data)
    outputRoot = _resolveUserPath(args.output)
    config = (
        _resolveUserPath(args.config)
        if args.config
        else REPOSITORY_ROOT / "configs" / "RGBonly.yaml"
    )
    resultRoot = outputRoot / "result"
    resultRoot.mkdir(parents=True, exist_ok=True)
    command = [
        "--dataset-root",
        str(dataRoot),
        "--target-instance",
        str(args.instance_id),
        "--output",
        str(resultRoot / "tracking.txt"),
        "--config",
        str(config),
    ]
    if args.sequence:
        command.extend(("--sequence", args.sequence))
    if args.max_frames is not None:
        command.extend(("--max-frames", str(args.max_frames)))
    if not args.no_mid_visual:
        command.extend(("--mid-visual-root", str(outputRoot / "midVisual")))
    if not args.no_result_visual:
        command.extend(("--result-visual-root", str(resultRoot / "visualResult")))
    return trackAirSim360Main(command)


def getInstanceIdMain(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="getInstanceID",
        description="Write the instance IDs visible in the first AirSim360 frame.",
    )
    parser.add_argument("data")
    parser.add_argument("output")
    parser.add_argument("--sequence", default=None)
    args = parser.parse_args(argv)

    source = AirSim360DataSource(maxFrames=1)
    try:
        source.open(str(_resolveUserPath(args.data)), args.sequence)
        frame = source.read()
        if frame is None or frame.segmentation is None or frame.segmentation.instance is None:
            raise ValueError("the first frame has no instance segmentation")
        lines = _formatInstanceIds(
            frame.segmentation.instance,
            frame.segmentation.semantic,
            frame.segmentation.classNames,
        )
        output = _resolveUserPath(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(output)
        return 0
    except (OSError, ValueError, InstaTargetError) as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 1
    finally:
        source.close()


def _formatInstanceIds(
    instance: np.ndarray,
    semantic: np.ndarray | None,
    classNames: dict[int, str],
) -> list[str]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for rawId in np.unique(instance):
        instanceId = int(rawId)
        if instanceId == 0:
            continue
        name = "unknown"
        if semantic is not None:
            mask = instance == instanceId
            semanticIds, counts = np.unique(semantic[mask], return_counts=True)
            if len(semanticIds):
                semanticId = int(semanticIds[int(np.argmax(counts))])
                name = classNames.get(semanticId, f"semantic_{semanticId}")
        grouped[name].append(instanceId)
    lines: list[str] = []
    for name in sorted(grouped):
        for ordinal, instanceId in enumerate(sorted(grouped[name]), start=1):
            lines.append(f"{name} {ordinal} {instanceId}")
        lines.append("")
    while lines and not lines[-1]:
        lines.pop()
    return lines


def _resolveUserPath(value: str) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute() and path.exists():
        return path.resolve()
    # `/data/...` in the short commands means repository-relative on Windows.
    normalized = value.lstrip("/\\")
    candidate = REPOSITORY_ROOT / normalized
    if value.startswith(("/", "\\")) or not path.is_absolute():
        return candidate.resolve()
    return path.resolve()


__all__ = ["getInstanceIdMain", "runMain"]
