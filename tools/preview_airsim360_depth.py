"""Preview one AirSim360 HDF5 depth map as a colorized PNG."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from instatarget.io.h5_depth_reader import readAirSim360DepthH5
from instatarget.tracker.depth_preprocessor import DepthPreprocessor
from instatarget.visualization.png import writeRgbPng


def buildParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preview one AirSim360 depth HDF5 file.")
    parser.add_argument("input_h5", help="Path to one .h5 depth file.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("artifacts") / "depth_preview" / "depth_color.png",
        help="Where to write the colorized preview PNG.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = buildParser()
    args = parser.parse_args(argv)
    depthPlane = readAirSim360DepthH5(args.input_h5)
    outputPath = args.output
    contrastPath = outputPath.with_name(f"{outputPath.stem}_contrast{outputPath.suffix}")
    edgePath = outputPath.with_name(f"{outputPath.stem}_edges{outputPath.suffix}")
    outputPath.parent.mkdir(parents=True, exist_ok=True)
    result = DepthPreprocessor().preprocess(depthPlane)
    writeRgbPng(outputPath, result.edgeRgb)
    writeRgbPng(contrastPath, _contrastPreview(depthPlane.values))
    writeRgbPng(edgePath, _edgePreview(result.edge, result.validMask))
    print(
        f"shape={depthPlane.values.shape} min={float(depthPlane.values.min()):.3f} "
        f"max={float(depthPlane.values.max()):.3f} "
        f"output={outputPath.resolve()} contrast={contrastPath.resolve()} "
        f"edges={edgePath.resolve()}"
    )
    return 0


def _contrastPreview(values: np.ndarray) -> np.ndarray:
    valid = np.isfinite(values) & (values >= 0.0)
    stretched = np.zeros_like(values, dtype=np.float32)
    if valid.any():
        logValues = np.log1p(np.clip(values, 0.0, None))
        low, high = np.percentile(logValues[valid], (2.0, 98.0))
        scale = max(float(high - low), 1e-6)
        stretched[valid] = np.clip((logValues[valid] - low) / scale, 0.0, 1.0)
    gray = np.rint((1.0 - stretched) * 255.0).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def _edgePreview(edge: np.ndarray, validMask: np.ndarray) -> np.ndarray:
    intensity = np.clip(edge, 0.0, 1.0)
    preview = np.stack(
        (intensity, intensity * 0.15, np.minimum(1.0, intensity * 1.25)),
        axis=-1,
    )
    preview[~validMask] = 0.0
    return np.rint(preview * 255.0).astype(np.uint8)


if __name__ == "__main__":
    raise SystemExit(main())
