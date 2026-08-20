"""Create a lossless inference-only HiT checkpoint for the Docker image."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPOSITORY_ROOT / "src"
HIT_ROOT = SOURCE_ROOT / "instatarget" / "vendor" / "hit"
DEFAULT_SOURCE = REPOSITORY_ROOT / "models" / "hit_small_stage3.pth"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "models" / "hit_small_stage3_inference.pth"
DEFAULT_SOURCE_CALIBRATION = REPOSITORY_ROOT / "models" / "hit_small_stage3.calibration.json"
DEFAULT_OUTPUT_CALIBRATION = (
    REPOSITORY_ROOT / "models" / "hit_small_stage3_inference.calibration.json"
)
MAX_OUTPUT_BYTES = 100 * 1024 * 1024


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Strip optimizer and training metadata from the HiT checkpoint."
    )
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--source-calibration", type=Path, default=DEFAULT_SOURCE_CALIBRATION
    )
    parser.add_argument(
        "--output-calibration", type=Path, default=DEFAULT_OUTPUT_CALIBRATION
    )
    args = parser.parse_args()
    compactCheckpoint(args.source, args.output)
    compactCalibration(
        args.source_calibration,
        args.output_calibration,
        sourceCheckpoint=args.source,
        compactCheckpoint=args.output,
    )
    return 0


def compactCheckpoint(source: Path, output: Path) -> None:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    if source == output:
        raise ValueError("source and output checkpoints must be different files")
    if not source.is_file():
        raise FileNotFoundError(f"source checkpoint does not exist: {source}")

    sys.path[:0] = [str(SOURCE_ROOT), str(HIT_ROOT)]
    import torch

    from instatarget.tracker.pytorch_hit_session import _loadCheckpoint

    checkpoint = _loadCheckpoint(torch, source)
    state = checkpoint.get("model") if isinstance(checkpoint, dict) else None
    if not isinstance(state, dict) or not state:
        raise ValueError("source checkpoint has no non-empty Stage 3 'model' state dictionary")

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        torch.save({"model": state}, temporary)
        compact = torch.load(temporary, map_location="cpu", weights_only=True)
        compactState = compact.get("model") if isinstance(compact, dict) else None
        _verifyState(torch, state, compactState)
        if temporary.stat().st_size >= MAX_OUTPUT_BYTES:
            raise ValueError(
                f"compact checkpoint is still at least 100 MiB: {temporary.stat().st_size} bytes"
            )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    sourceMiB = source.stat().st_size / (1024 * 1024)
    outputMiB = output.stat().st_size / (1024 * 1024)
    print(f"checkpoint: {sourceMiB:.1f} MiB -> {outputMiB:.1f} MiB")


def compactCalibration(
    source: Path,
    output: Path,
    *,
    sourceCheckpoint: Path,
    compactCheckpoint: Path,
) -> None:
    source = source.expanduser().resolve()
    output = output.expanduser().resolve()
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read source calibration: {source}") from error
    if not isinstance(payload, dict):
        raise ValueError("source calibration root must be a mapping")
    sourceHash = _sha256(sourceCheckpoint)
    if payload.get("checkpointSha256") != sourceHash:
        raise ValueError("source calibration is not bound to the source checkpoint")
    payload["checkpointSha256"] = _sha256(compactCheckpoint)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=True, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.expanduser().resolve().open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verifyState(torch: object, source: dict[str, object], compact: object) -> None:
    if not isinstance(compact, dict) or source.keys() != compact.keys():
        raise ValueError("compact checkpoint state keys differ from the source")
    for name, sourceValue in source.items():
        compactValue = compact[name]
        if torch.is_tensor(sourceValue):
            if not torch.is_tensor(compactValue) or not torch.equal(sourceValue, compactValue):
                raise ValueError(f"compact checkpoint tensor differs: {name}")
        elif sourceValue != compactValue:
            raise ValueError(f"compact checkpoint value differs: {name}")


if __name__ == "__main__":
    raise SystemExit(main())
