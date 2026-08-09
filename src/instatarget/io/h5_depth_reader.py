"""Temporary AirSim360 HDF5 depth reader.

This reader only targets the small depth files used in the local smoke tests.
It understands the old-style HDF5 layout used by the AirSim360 sample depth
panoramas: one chunked float32 dataset compressed with the built-in LZF filter.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from instatarget.core.errors import DecodeError
from instatarget.core.types import DepthPlane

HDF5_SIGNATURE = b"\x89HDF\r\n\x1a\n"
TREE_SIGNATURE = b"TREE"


@dataclass(frozen=True, slots=True)
class _ChunkRecord:
    size: int
    yPx: int
    xPx: int
    filterMask: int
    address: int


def readAirSim360DepthH5(path: str | Path) -> DepthPlane:
    """Read one AirSim360 depth panorama stored in an HDF5 ``.h5`` file."""
    depthPath = Path(path)
    try:
        payload = depthPath.read_bytes()
    except OSError as error:
        raise DecodeError(f"cannot read depth file {depthPath}: {error}") from error

    if not payload.startswith(HDF5_SIGNATURE):
        raise DecodeError(f"unsupported HDF5 signature: {depthPath}")

    chunkRecords = _readChunkRecords(payload)
    if not chunkRecords:
        raise DecodeError(f"no chunk records found in depth file: {depthPath}")

    chunkHeightPx, chunkWidthPx = _inferChunkShape(chunkRecords)
    heightPx, widthPx = _inferPlaneShape(chunkRecords, chunkHeightPx, chunkWidthPx)

    values = np.zeros((heightPx, widthPx), dtype=np.float32)
    expectedChunkBytes = chunkHeightPx * chunkWidthPx * 4
    for record in chunkRecords:
        if record.size <= 0 or record.address <= 0:
            continue
        chunkBytes = payload[record.address : record.address + record.size]
        rawBytes = _lzfDecompress(chunkBytes, expectedChunkBytes)
        chunkValues = np.frombuffer(rawBytes, dtype="<f4").reshape(chunkHeightPx, chunkWidthPx)
        values[record.yPx : record.yPx + chunkHeightPx, record.xPx : record.xPx + chunkWidthPx] = (
            chunkValues
        )

    validMask = np.isfinite(values) & (values >= 0.0)
    return DepthPlane(values=values, validMask=validMask, unit="m")


def readAirSim360DepthArray(path: str | Path) -> NDArray[np.float32]:
    """Read one AirSim360 depth panorama as a float32 NumPy array."""
    return readAirSim360DepthH5(path).values


def _readChunkRecords(payload: bytes) -> list[_ChunkRecord]:
    records: list[_ChunkRecord] = []
    seen: set[tuple[int, int, int, int, int]] = set()
    offset = 0
    while True:
        treeOffset = payload.find(TREE_SIGNATURE, offset)
        if treeOffset < 0:
            break
        offset = treeOffset + 1
        if treeOffset + 8 > len(payload):
            continue
        nodeType = payload[treeOffset + 4]
        nodeLevel = payload[treeOffset + 5]
        entries = int.from_bytes(payload[treeOffset + 6 : treeOffset + 8], "little")
        if nodeType != 1 or nodeLevel != 0:
            continue
        headerSize = 24
        recordSize = 40
        for index in range(entries):
            start = treeOffset + headerSize + index * recordSize
            end = start + recordSize
            if end > len(payload):
                break
            size, yPx, xPx, filterMask, address = struct.unpack_from("<QQQQQ", payload, start)
            key = (int(size), int(yPx), int(xPx), int(filterMask), int(address))
            if key in seen:
                continue
            seen.add(key)
            if size == 0 and address == 0:
                continue
            records.append(
                _ChunkRecord(
                    size=int(size),
                    yPx=int(yPx),
                    xPx=int(xPx),
                    filterMask=int(filterMask),
                    address=int(address),
                )
            )
    return records


def _inferChunkShape(records: list[_ChunkRecord]) -> tuple[int, int]:
    yValues = sorted({record.yPx for record in records})
    xValues = sorted({record.xPx for record in records})
    if len(yValues) < 2 or len(xValues) < 2:
        raise DecodeError("cannot infer chunk shape from depth records")
    chunkHeightPx = _smallestPositiveDelta(yValues)
    chunkWidthPx = _smallestPositiveDelta(xValues)
    if chunkHeightPx <= 0 or chunkWidthPx <= 0:
        raise DecodeError("invalid chunk spacing in depth records")
    return chunkHeightPx, chunkWidthPx


def _inferPlaneShape(
    records: list[_ChunkRecord],
    chunkHeightPx: int,
    chunkWidthPx: int,
) -> tuple[int, int]:
    heightPx = max(record.yPx for record in records) + chunkHeightPx
    widthPx = max(record.xPx for record in records) + chunkWidthPx
    return heightPx, widthPx


def _smallestPositiveDelta(values: list[int]) -> int:
    deltas = [b - a for a, b in zip(values, values[1:]) if b > a]
    if not deltas:
        raise DecodeError("cannot infer a positive chunk delta")
    return min(deltas)


def _lzfDecompress(data: bytes, expectedLength: int) -> bytes:
    out = bytearray()
    ip = 0
    inputLength = len(data)
    while ip < inputLength:
        control = data[ip]
        ip += 1
        if control < 32:
            literalLength = control + 1
            if ip + literalLength > inputLength:
                raise DecodeError("truncated LZF literal run")
            out.extend(data[ip : ip + literalLength])
            ip += literalLength
            continue

        matchLength = control >> 5
        refOffset = len(out) - ((control & 0x1F) << 8) - 1
        if matchLength == 7:
            if ip >= inputLength:
                raise DecodeError("truncated LZF match length")
            matchLength += data[ip]
            ip += 1
        if ip >= inputLength:
            raise DecodeError("truncated LZF match offset")
        refOffset -= data[ip]
        ip += 1
        matchLength += 2
        if refOffset < 0:
            raise DecodeError("invalid LZF back-reference")
        for _ in range(matchLength):
            out.append(out[refOffset])
            refOffset += 1

    if len(out) != expectedLength:
        raise DecodeError(
            f"unexpected LZF output length: expected={expectedLength}, actual={len(out)}"
        )
    return bytes(out)


__all__ = ["readAirSim360DepthArray", "readAirSim360DepthH5"]
