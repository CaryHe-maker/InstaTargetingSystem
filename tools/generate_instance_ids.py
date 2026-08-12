"""Generate a semantic-grouped InstanceID.txt for one AirSim360 sequence."""

from __future__ import annotations

import argparse
from pathlib import Path

from instatarget.core.errors import DecodeError
from instatarget.data.airsim360_source import AirSim360DataSource
from instatarget.visualization import collectInstanceIdGroups, writeInstanceIdDocument


def buildParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        required=True,
        help="Sequence or dataset parent directory.",
    )
    parser.add_argument(
        "--sequence",
        default=None,
        help="Sequence subdirectory when --dataset-root contains multiple sequences.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file; defaults to artifacts/<data-relative>/InstanceID.txt.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = buildParser().parse_args(argv)
    source = AirSim360DataSource()
    try:
        source.open(args.dataset_root, args.sequence)
        firstFrame = source.read()
        if firstFrame is None:
            raise DecodeError("AirSim360 sequence is empty")
        groups = collectInstanceIdGroups(firstFrame)
    finally:
        source.close()

    outputPath = writeInstanceIdDocument(_resolveOutputPath(args), groups)
    print(
        f"output={outputPath} classes={len(groups)} "
        f"instances={sum(len(group.instanceIds) for group in groups)} frameIndex=0"
    )
    return 0

def _resolveOutputPath(args: argparse.Namespace) -> Path:
    if args.output is not None:
        return args.output
    repositoryRoot = Path(__file__).resolve().parents[1]
    dataRoot = (repositoryRoot / "data").resolve()
    datasetPath = Path(args.dataset_root).expanduser().resolve()
    sequencePath = datasetPath / args.sequence if args.sequence else datasetPath
    try:
        relativeSequence = sequencePath.relative_to(dataRoot)
    except ValueError:
        relativeSequence = Path(sequencePath.name)
    return repositoryRoot / "artifacts" / relativeSequence / "InstanceID.txt"


if __name__ == "__main__":
    raise SystemExit(main())
