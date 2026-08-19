"""Partition a Linux root filesystem into balanced Docker layer buckets."""

from __future__ import annotations

import argparse
import os
import shutil
import stat
from dataclasses import dataclass
from pathlib import Path

MIB = 1024 * 1024
LARGE_LIBRARY_BYTES = 64 * MIB
ROOT_ENTRIES = (
    "app",
    "bin",
    "boot",
    "etc",
    "home",
    "lib",
    "lib32",
    "lib64",
    "libx32",
    "media",
    "mnt",
    "opt",
    "root",
    "run",
    "sbin",
    "srv",
    "tmp",
    "usr",
    "var",
    "workspace",
)
LARGE_LIBRARY_MARKERS = (
    "/site-packages/torch/lib/",
    "/site-packages/nvidia/",
    "/site-packages/cusparselt/",
    "/site-packages/triton/",
)


@dataclass(frozen=True, slots=True)
class Entry:
    source: Path
    relative: Path
    size: int
    isSymlink: bool = False


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layers", type=int, default=50)
    args = parser.parse_args()
    partitionRoot(args.root, args.output, args.layers)
    return 0


def partitionRoot(root: Path, output: Path, layerCount: int) -> None:
    root = root.resolve()
    output = output.resolve()
    if layerCount < 2:
        raise ValueError("layer count must be at least two")
    if output.exists():
        shutil.rmtree(output)
    buckets = [output / f"{index:02d}" for index in range(layerCount)]
    for bucket in buckets:
        bucket.mkdir(parents=True)

    entries, directories = _collectEntries(root)
    large = sorted(
        (entry for entry in entries if _isLargeLibrary(entry)),
        key=lambda entry: entry.size,
        reverse=True,
    )
    if len(large) >= layerCount:
        raise ValueError(
            f"{len(large)} isolated libraries leave no buckets for remaining files"
        )

    assignments: list[list[Entry]] = [[] for _ in buckets]
    sizes = [0] * layerCount
    assigned: set[Path] = set()
    for index, entry in enumerate(large):
        assignments[index].append(entry)
        sizes[index] += entry.size
        assigned.add(entry.relative)

    regularBucketIndexes = tuple(range(len(large), layerCount))
    remaining = sorted(
        (entry for entry in entries if entry.relative not in assigned),
        key=lambda entry: entry.size,
        reverse=True,
    )
    for entry in remaining:
        index = min(regularBucketIndexes, key=sizes.__getitem__)
        assignments[index].append(entry)
        sizes[index] += entry.size

    empty = [index for index, items in enumerate(assignments) if not items]
    if empty:
        raise ValueError(f"partition generated empty layer buckets: {empty}")

    for index, items in enumerate(assignments):
        for entry in items:
            _copyEntry(entry, buckets[index] / entry.relative)
    _copyEmptyDirectories(root, directories, buckets[0])
    _restoreDirectoryMetadata(root, buckets)

    for index, items in enumerate(assignments):
        isolated = " isolated" if index < len(large) else ""
        print(
            f"layer {index:02d}: {sizes[index] / MIB:8.1f} MiB, "
            f"{len(items):6d} entries{isolated}"
        )
    print(f"partitioned {len(entries)} entries into {layerCount} non-empty layers")


def _collectEntries(root: Path) -> tuple[list[Entry], list[Path]]:
    entries: list[Entry] = []
    directories: list[Path] = []
    for name in ROOT_ENTRIES:
        source = root / name
        if not source.exists() and not source.is_symlink():
            continue
        if source.is_symlink():
            entries.append(Entry(source, Path(name), 0, isSymlink=True))
            continue
        if source.is_file():
            entries.append(Entry(source, Path(name), source.stat().st_size))
            continue
        for directory, childNames, fileNames in os.walk(source, followlinks=False):
            directoryPath = Path(directory)
            directories.append(directoryPath.relative_to(root))
            symlinkDirectories = []
            for childName in childNames:
                child = directoryPath / childName
                if child.is_symlink():
                    entries.append(
                        Entry(child, child.relative_to(root), 0, isSymlink=True)
                    )
                    symlinkDirectories.append(childName)
            for childName in symlinkDirectories:
                childNames.remove(childName)
            for fileName in fileNames:
                child = directoryPath / fileName
                metadata = child.lstat()
                if stat.S_ISLNK(metadata.st_mode):
                    entries.append(
                        Entry(child, child.relative_to(root), 0, isSymlink=True)
                    )
                elif stat.S_ISREG(metadata.st_mode):
                    entries.append(Entry(child, child.relative_to(root), metadata.st_size))
    if not entries:
        raise ValueError(f"root filesystem has no partitionable entries: {root}")
    return entries, directories


def _isLargeLibrary(entry: Entry) -> bool:
    if entry.isSymlink or entry.size < LARGE_LIBRARY_BYTES:
        return False
    normalized = "/" + entry.relative.as_posix()
    return any(marker in normalized for marker in LARGE_LIBRARY_MARKERS)


def _copyEntry(entry: Entry, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    metadata = entry.source.lstat()
    if entry.isSymlink:
        target.symlink_to(os.readlink(entry.source))
        os.chown(target, metadata.st_uid, metadata.st_gid, follow_symlinks=False)
        return
    try:
        os.link(entry.source, target)
    except OSError:
        shutil.copy2(entry.source, target, follow_symlinks=False)
    os.chown(target, metadata.st_uid, metadata.st_gid, follow_symlinks=False)


def _copyEmptyDirectories(root: Path, directories: list[Path], bucket: Path) -> None:
    for relative in directories:
        source = root / relative
        try:
            next(source.iterdir())
        except StopIteration:
            (bucket / relative).mkdir(parents=True, exist_ok=True)


def _restoreDirectoryMetadata(root: Path, buckets: list[Path]) -> None:
    for bucket in buckets:
        directories = sorted(
            (
                path
                for path in bucket.rglob("*")
                if path.is_dir() and not path.is_symlink()
            ),
            key=lambda path: len(path.parts),
            reverse=True,
        )
        for target in directories:
            source = root / target.relative_to(bucket)
            if not source.is_dir():
                continue
            metadata = source.stat()
            shutil.copystat(source, target, follow_symlinks=False)
            os.chown(target, metadata.st_uid, metadata.st_gid)


if __name__ == "__main__":
    raise SystemExit(main())
