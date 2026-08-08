"""CLI entry point for the smoke-test geometry -> tracker pipeline."""

from __future__ import annotations

import argparse
from pathlib import Path

from instatarget.app.driver import runSmokePipeline
from instatarget.core.errors import InstaTargetError
from instatarget.core.types import BBoxXYWH


def buildParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a simple geometry + tracker smoke test.")
    parser.add_argument("input_path", help="Directory or single image to read.")
    parser.add_argument(
        "--initial-box",
        nargs=4,
        type=float,
        metavar=("X", "Y", "W", "H"),
        required=True,
        help="Initial ERP bbox in pixels.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("artifacts") / "smoke-track",
        help="Where visualization artifacts will be written.",
    )
    parser.add_argument("--sequence-id", type=str, default=None, help="Optional sequence label.")
    parser.add_argument("--view-width", type=int, default=256)
    parser.add_argument("--view-height", type=int, default=256)
    parser.add_argument("--recursive", action="store_true", help="Read images recursively.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = buildParser()
    args = parser.parse_args(argv)
    initialBox = BBoxXYWH(*args.initial_box)
    try:
        results = runSmokePipeline(
            args.input_path,
            args.output_root,
            initialBox,
            sequenceId=args.sequence_id,
            viewWidthPx=args.view_width,
            viewHeightPx=args.view_height,
            recursive=args.recursive,
        )
    except InstaTargetError as error:
        parser.error(str(error))
        return 2

    print(f"frames={len(results)} output={Path(args.output_root).resolve()}")
    for result in results:
        print(
            f"{int(result.frameIndex):06d} "
            f"bbox=({result.bbox.xPx:.1f},{result.bbox.yPx:.1f},"
            f"{result.bbox.widthPx:.1f},{result.bbox.heightPx:.1f}) "
            f"score={result.confidence:.3f} status={result.status.name}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
