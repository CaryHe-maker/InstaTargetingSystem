import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from instatarget.training.manifest_builder import buildManifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Build the indexed 360 training manifest")
    parser.add_argument("--data-root", default=r"E:\NewDownload\train")
    parser.add_argument("--output", default=r"E:\NewDownload\train\manifest.jsonl")
    parser.add_argument("--seed", type=int, default=20260819)
    parser.add_argument(
        "--exclude-file",
        type=Path,
        help="Optional UTF-8 file containing one group/sequence ID per line",
    )
    arguments = parser.parse_args(argv)
    excluded: tuple[str, ...] = ()
    if arguments.exclude_file is not None:
        excluded = tuple(
            line.strip()
            for line in arguments.exclude_file.read_text(encoding="utf-8-sig").splitlines()
            if line.strip()
        )
    counts = buildManifest(
        arguments.data_root,
        arguments.output,
        seed=arguments.seed,
        excludedSequenceIds=excluded,
    )
    print(json.dumps({"output": arguments.output, "frameCounts": counts}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
