"""Verify that a GitHub checkout can build the CUDA 12.8 submission image."""

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
BASE_IMAGE = "pytorch/pytorch:2.11.0-cuda12.8-cudnn9-devel"
PARTITION_LAYER_COUNT = 7
MAX_LAYER_COUNT = 10
MAX_IMAGE_SIZE_BYTES = 5_000_000_000
GITHUB_FILE_LIMIT_BYTES = 100_000_000
DOCKER_CONTEXT_REQUIRED = (
    "track.py",
    "configs/RGBonly.yaml",
    "models/hit_small_stage3_inference.pth",
    "models/hit_small_stage3_inference.calibration.json",
    "src/instatarget/app/competition.py",
    "src/instatarget/geometry/spherical_geometry.py",
    "src/instatarget/tracker/pytorch_hit_session.py",
    "src/instatarget/training/__init__.py",
    "src/instatarget/training/model.py",
    "src/instatarget/vendor/hit/configs/HiT_Small.yaml",
)
RUNTIME_IMPORT_CODE = (
    "import sys, torch, torchvision; "
    "assert sys.version_info[:2] == (3, 12), sys.version; "
    "assert torch.__version__.startswith('2.11.0'), torch.__version__; "
    "assert torchvision.__version__.startswith('0.26.0'), torchvision.__version__; "
    "assert torch.version.cuda == '12.8', torch.version.cuda; "
    "arch_flags = torch._C._cuda_getArchFlags().split(); "
    "assert 'sm_120' in arch_flags, arch_flags; "
    "sys.path.insert(0, '/app/src'); "
    "from instatarget.app.competition import runCompetition; "
    "from instatarget.core.types import BBoxXYWH; "
    "from instatarget.geometry import SphericalGeometryImpl; "
    "from instatarget.tracker.pytorch_hit_session import validateHiTCheckpoint; "
    "geometry = SphericalGeometryImpl(boundarySamplesPerEdge=33); "
    "wrapped_bfov = geometry.bboxToBfov("
    "BBoxXYWH(xPx=0.0, yPx=6.0, widthPx=174.0, heightPx=126.0), 360, 180); "
    "assert 0.0 < wrapped_bfov.horizontalFovRad < 3.141592653589793; "
    "assert 0.0 < wrapped_bfov.verticalFovRad < 3.141592653589793; "
    "parameter_count = validateHiTCheckpoint("
    "'/app/models/hit_small_stage3_inference.pth'); "
    "assert 'instatarget.data' not in sys.modules; "
    "print(f'CUDA 12.8 sm_120 runtime, wrapped BFoV Geometry, and "
    "{parameter_count} checkpoint parameters verified')"
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
        help="Also inspect a built image and enforce the layer and size limits.",
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
        f"FROM {BASE_IMAGE} AS runtime",
        "assert sys.version_info[:2] == (3, 12)",
        "assert torch.__version__.startswith('2.11.0')",
        "assert torchvision.__version__.startswith('0.26.0')",
        "assert torch.version.cuda == '12.8'",
        "assert 'sm_120' in torch._C._cuda_getArchFlags().split()",
        "COPY src ./src",
        "wrapped_bfov = geometry.bboxToBfov",
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
    expected = [f"{index:02d}" for index in range(PARTITION_LAYER_COUNT)]
    if copies != expected:
        raise RuntimeError(f"final image layer buckets must be {expected}, actual={copies}")
    filesystemInstructions = re.findall(r"(?im)^(?:ADD|RUN|COPY)\b.*$", finalStage[1])
    if len(filesystemInstructions) != PARTITION_LAYER_COUNT:
        raise RuntimeError(
            f"final stage must have exactly {PARTITION_LAYER_COUNT} filesystem instructions: "
            f"actual={filesystemInstructions}"
        )
    print(
        "source checkout verified: required inputs are Git-tracked and present in the "
        "Docker context, the compact model hash matches, and the Dockerfile has the "
        f"required CUDA 12.8 base image and {PARTITION_LAYER_COUNT} final layers"
    )


def verifyBuiltImage(image: str) -> None:
    try:
        completed = subprocess.run(
            [
                "docker",
                "image",
                "inspect",
                image,
                "--format",
                "{{json .RootFS.Layers}}|{{.Size}}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        layersJson, sizeText = completed.stdout.strip().rsplit("|", 1)
        layers = json.loads(layersJson)
        imageSize = int(sizeText)
    except (
        OSError,
        ValueError,
        subprocess.CalledProcessError,
        json.JSONDecodeError,
    ) as error:
        raise RuntimeError(f"cannot inspect Docker image {image!r}") from error
    if not isinstance(layers, list) or not 1 <= len(layers) <= MAX_LAYER_COUNT:
        actual = len(layers) if isinstance(layers, list) else layers
        raise RuntimeError(
            f"submission image must have 1-{MAX_LAYER_COUNT} RootFS layers, actual={actual}"
        )
    if imageSize > MAX_IMAGE_SIZE_BYTES:
        raise RuntimeError(
            "submission image exceeds the project size guard: "
            f"actual={imageSize}, maximum={MAX_IMAGE_SIZE_BYTES} bytes"
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
        f"image verified: {image} has {len(layers)} RootFS layers (limit {MAX_LAYER_COUNT}) "
        f"and is {imageSize} bytes (limit {MAX_IMAGE_SIZE_BYTES}), "
        "uses CUDA 12.8 with sm_120 support, and strictly loads the committed checkpoint "
        "without network access"
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
