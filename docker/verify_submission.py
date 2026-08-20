"""Verify that a GitHub checkout can build a seven-layer submission image."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = REPOSITORY_ROOT / "models" / "hit_small_stage3_inference.pth"
CALIBRATION = (
    REPOSITORY_ROOT / "models" / "hit_small_stage3_inference.calibration.json"
)
DOCKERFILE = REPOSITORY_ROOT / "Dockerfile"
EXPECTED_LAYER_COUNT = 7
GITHUB_FILE_LIMIT_BYTES = 100_000_000


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image",
        help="Also inspect a built Docker image and require exactly seven RootFS layers.",
    )
    args = parser.parse_args()
    verifySourceCheckout()
    if args.image:
        verifyBuiltImage(args.image)
    return 0


def verifySourceCheckout() -> None:
    for path in (CHECKPOINT, CALIBRATION, DOCKERFILE):
        if not path.is_file():
            raise RuntimeError(f"required Docker build input is missing: {path}")
    if CHECKPOINT.stat().st_size >= GITHUB_FILE_LIMIT_BYTES:
        raise RuntimeError(
            f"compact checkpoint exceeds GitHub's 100 MB limit: {CHECKPOINT.stat().st_size}"
        )
    try:
        payload = json.loads(CALIBRATION.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot read compact calibration: {CALIBRATION}") from error
    expectedHash = payload.get("checkpointSha256") if isinstance(payload, dict) else None
    actualHash = _sha256(CHECKPOINT)
    if expectedHash != actualHash:
        raise RuntimeError(
            "compact checkpoint/calibration hash mismatch: "
            f"expected={expectedHash}, actual={actualHash}"
        )

    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    finalStage = re.split(r"(?im)^FROM\s+scratch\s*$", dockerfile)
    if len(finalStage) != 2:
        raise RuntimeError("Dockerfile must contain exactly one final 'FROM scratch' stage")
    copies = re.findall(
        r"(?im)^COPY\s+--from=runtime\s+/layer-parts/(\d{2})/\s+/\s*$",
        finalStage[1],
    )
    expected = [f"{index:02d}" for index in range(EXPECTED_LAYER_COUNT)]
    if copies != expected:
        raise RuntimeError(f"final image layer buckets must be {expected}, actual={copies}")
    filesystemInstructions = re.findall(r"(?im)^(?:ADD|RUN|COPY)\b.*$", finalStage[1])
    if len(filesystemInstructions) != EXPECTED_LAYER_COUNT:
        raise RuntimeError(
            "final stage must have exactly seven filesystem instructions: "
            f"actual={filesystemInstructions}"
        )
    print(
        "source checkout verified: compact model hash matches and Dockerfile has "
        "exactly seven final filesystem layers"
    )


def verifyBuiltImage(image: str) -> None:
    try:
        completed = subprocess.run(
            ["docker", "image", "inspect", image, "--format", "{{json .RootFS.Layers}}"],
            check=True,
            capture_output=True,
            text=True,
        )
        layers = json.loads(completed.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as error:
        raise RuntimeError(f"cannot inspect Docker image {image!r}") from error
    if not isinstance(layers, list) or len(layers) != EXPECTED_LAYER_COUNT:
        actual = len(layers) if isinstance(layers, list) else layers
        raise RuntimeError(
            f"submission image must have exactly seven RootFS layers, actual={actual}"
        )
    print(f"image verified: {image} has exactly seven RootFS layers")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
