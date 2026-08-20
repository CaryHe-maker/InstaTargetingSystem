"""Verify that a GitHub checkout can build a seven-layer submission image."""

from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import subprocess
from pathlib import Path, PurePosixPath

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CHECKPOINT = REPOSITORY_ROOT / "models" / "hit_small_stage3_inference.pth"
CALIBRATION = (
    REPOSITORY_ROOT / "models" / "hit_small_stage3_inference.calibration.json"
)
DOCKERFILE = REPOSITORY_ROOT / "Dockerfile"
DOCKERIGNORE = REPOSITORY_ROOT / ".dockerignore"
EXPECTED_LAYER_COUNT = 7
GITHUB_FILE_LIMIT_BYTES = 100_000_000
DOCKER_CONTEXT_REQUIRED = (
    "track.py",
    "configs/RGBonly.yaml",
    "models/hit_small_stage3_inference.pth",
    "models/hit_small_stage3_inference.calibration.json",
    "src/instatarget/app/competition.py",
    "src/instatarget/tracker/pytorch_hit_session.py",
    "src/instatarget/training/__init__.py",
    "src/instatarget/training/model.py",
    "src/instatarget/vendor/hit/configs/HiT_Small.yaml",
)
RUNTIME_IMPORT_CODE = (
    "import sys; "
    "sys.path.insert(0, '/app/src'); "
    "from instatarget.app.competition import runCompetition; "
    "from instatarget.tracker.pytorch_hit_session import validateHiTCheckpoint; "
    "parameter_count = validateHiTCheckpoint("
    "'/app/models/hit_small_stage3_inference.pth'); "
    "assert 'instatarget.data' not in sys.modules; "
    "print(f'runtime imports and {parameter_count} checkpoint parameters verified')"
)
GIT_REQUIRED = (
    ".dockerignore",
    "Dockerfile",
    "docker/partition_image.py",
    "docker/requirements.txt",
    *DOCKER_CONTEXT_REQUIRED,
)


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
    requiredPaths = tuple(REPOSITORY_ROOT / path for path in DOCKER_CONTEXT_REQUIRED)
    for path in (*requiredPaths, DOCKERFILE, DOCKERIGNORE):
        if not path.is_file():
            raise RuntimeError(f"required Docker build input is missing: {path}")
    dockerignore = DOCKERIGNORE.read_text(encoding="utf-8")
    excluded = [
        path
        for path in DOCKER_CONTEXT_REQUIRED
        if _isDockerIgnored(path, dockerignore.splitlines())
    ]
    if excluded:
        raise RuntimeError(f"required runtime files are excluded by .dockerignore: {excluded}")
    _verifyGitTracked(GIT_REQUIRED)
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
    requiredInstructions = (
        "COPY src ./src",
        "COPY configs/RGBonly.yaml ./configs/RGBonly.yaml",
        "COPY models/hit_small_stage3_inference.pth ./models/hit_small_stage3_inference.pth",
        "COPY models/hit_small_stage3_inference.calibration.json "
        "./models/hit_small_stage3_inference.calibration.json",
        "COPY track.py ./track.py",
        "validateHiTCheckpoint('/app/models/hit_small_stage3_inference.pth')",
        'ENTRYPOINT ["python", "/app/track.py"]',
    )
    missingInstructions = [
        instruction for instruction in requiredInstructions if instruction not in dockerfile
    ]
    if missingInstructions:
        raise RuntimeError(
            f"Dockerfile is missing required runtime wiring: {missingInstructions}"
        )
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
        "source checkout verified: required inputs are Git-tracked and present in the "
        "Docker context, the compact model hash matches, and the Dockerfile has exactly "
        "seven final layers"
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
    try:
        subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                "--entrypoint",
                "python",
                image,
                "-c",
                RUNTIME_IMPORT_CODE,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        details = (
            error.stderr.strip()
            if isinstance(error, subprocess.CalledProcessError)
            else error
        )
        raise RuntimeError(
            f"runtime import smoke test failed for Docker image {image!r}: {details}"
        ) from error
    print(
        f"image verified: {image} has exactly seven RootFS layers and strictly loads "
        "the committed checkpoint without network access"
    )


def _verifyGitTracked(paths: tuple[str, ...]) -> None:
    try:
        subprocess.run(
            [
                "git",
                "-C",
                str(REPOSITORY_ROOT),
                "ls-files",
                "--error-unmatch",
                "--",
                *paths,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        details = (
            error.stderr.strip()
            if isinstance(error, subprocess.CalledProcessError)
            else str(error)
        )
        raise RuntimeError(
            "required Docker inputs must be committed before a GitHub checkout can "
            f"reproduce the build: {details}"
        ) from error


def _isDockerIgnored(path: str, rules: list[str]) -> bool:
    ignored = False
    for rawRule in rules:
        rule = rawRule.strip()
        if not rule or rule.startswith("#"):
            continue
        negated = rule.startswith("!")
        if negated:
            rule = rule[1:]
        if _dockerRuleMatches(path, rule):
            ignored = not negated
    return ignored


def _dockerRuleMatches(path: str, rule: str) -> bool:
    normalized = rule.replace("\\", "/").lstrip("/").rstrip("/")
    if not normalized:
        return False
    purePath = PurePosixPath(path)
    candidates = [purePath.as_posix(), *(parent.as_posix() for parent in purePath.parents)]
    if "/" not in normalized:
        return any(fnmatch.fnmatchcase(part, normalized) for part in purePath.parts)
    return any(fnmatch.fnmatchcase(candidate, normalized) for candidate in candidates)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
