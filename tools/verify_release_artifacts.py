"""Verify checkpoint/config/Docker release contracts before submission."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import tarfile
from pathlib import Path

import yaml


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--image-tar", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    root = args.root.resolve()
    compactCheckpoint = root / "models" / "hit_small_stage3_inference.pth"
    compactCalibration = root / "models" / "hit_small_stage3_inference.calibration.json"
    _verifyCalibrationPair(compactCheckpoint, compactCalibration, required=True)

    sourceCheckpoint = root / "models" / "hit_small_stage3.pth"
    sourceCalibration = root / "models" / "hit_small_stage3.calibration.json"
    sourcePresent = sourceCheckpoint.is_file() or sourceCalibration.is_file()
    if sourcePresent:
        _verifyCalibrationPair(sourceCheckpoint, sourceCalibration, required=True)
        _verifyCalibrationPayloads(sourceCalibration, compactCalibration)

    config = yaml.safe_load((root / "configs" / "RGBonly.yaml").read_text(encoding="utf-8"))
    if Path(config["model"]["weights"]).name != "hit_small_stage3.pth":
        raise RuntimeError("RGBonly.yaml must select hit_small_stage3.pth")
    if Path(config["scoring"]["calibrationArtifact"]).name != "hit_small_stage3.calibration.json":
        raise RuntimeError("RGBonly.yaml must select the paired Stage 3 calibration")

    tracked = set(
        subprocess.run(
            ["git", "-C", str(root), "ls-files"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    )
    requiredContext = {
        "Dockerfile",
        ".dockerignore",
        "configs/RGBonly.yaml",
        "models/hit_small_stage3_inference.pth",
        "models/hit_small_stage3_inference.calibration.json",
    }
    missing = sorted(requiredContext - tracked)
    if missing:
        raise RuntimeError(f"Git clone would miss Docker build inputs: {missing}")

    dockerfile = (root / "Dockerfile").read_text(encoding="utf-8")
    copies = re.findall(r"^COPY --from=runtime /layer-parts/\d{2}/ /$", dockerfile, re.MULTILINE)
    if len(copies) != 7:
        raise RuntimeError(
            f"Dockerfile must assemble exactly 7 filesystem layers, found {len(copies)}"
        )
    for expected in (
        "models/hit_small_stage3_inference.pth",
        "models/hit_small_stage3_inference.calibration.json",
    ):
        if expected not in dockerfile:
            raise RuntimeError(f"Dockerfile does not copy {expected}")

    dockerignore = (root / ".dockerignore").read_text(encoding="utf-8")
    for expected in (
        "!models/hit_small_stage3_inference.pth",
        "!models/hit_small_stage3_inference.calibration.json",
    ):
        if expected not in dockerignore:
            raise RuntimeError(f".dockerignore does not include {expected}")

    imageTar = args.image_tar.resolve() if args.image_tar else None
    if imageTar is not None:
        _verifyImageLayers(imageTar)
    print(
        json.dumps(
            {
                "status": "ok",
                "filesystemLayers": 7,
                "root": str(root),
                "sourceArtifactsVerified": sourcePresent,
                "hashes": {
                    "compactCheckpoint": _sha256(compactCheckpoint),
                    "compactCalibration": _sha256(compactCalibration),
                    "config": _sha256(root / "configs" / "RGBonly.yaml"),
                    "dockerfile": _sha256(root / "Dockerfile"),
                    "dockerignore": _sha256(root / ".dockerignore"),
                },
            }
        )
    )
    return 0


def _verifyCalibrationPair(checkpoint: Path, calibration: Path, *, required: bool) -> None:
    if not checkpoint.is_file() or not calibration.is_file():
        if required:
            missing = checkpoint if not checkpoint.is_file() else calibration
            raise RuntimeError(f"required release artifact is missing: {missing}")
        return
    payload = json.loads(calibration.read_text(encoding="utf-8"))
    actual = _sha256(checkpoint)
    if payload.get("checkpointSha256") != actual:
        raise RuntimeError(f"calibration hash mismatch: {calibration}")


def _verifyCalibrationPayloads(source: Path, compact: Path) -> None:
    sourcePayload = json.loads(source.read_text(encoding="utf-8"))
    compactPayload = json.loads(compact.read_text(encoding="utf-8"))
    sourcePayload.pop("checkpointSha256", None)
    compactPayload.pop("checkpointSha256", None)
    if sourcePayload != compactPayload:
        raise RuntimeError("compact calibration parameters differ from source calibration")


def _verifyImageLayers(path: Path) -> None:
    _requireFile(path)
    with tarfile.open(path) as archive:
        manifestMember = archive.extractfile("manifest.json")
        if manifestMember is None:
            raise RuntimeError("Docker image archive has no manifest.json")
        manifest = json.load(manifestMember)
    if len(manifest) != 1 or len(manifest[0].get("Layers", [])) != 7:
        raise RuntimeError("Docker image archive must contain exactly 7 filesystem layers")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _requireFile(path: Path) -> None:
    if not path.is_file():
        raise RuntimeError(f"required release artifact is missing: {path}")


if __name__ == "__main__":
    raise SystemExit(main())
