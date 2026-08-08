"""Lightweight RGB image decoding helpers for local test data."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from instatarget.core.errors import DecodeError

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def readRgbImage(path: str | Path) -> NDArray[np.uint8]:
    """Decode one image into a uint8 RGB array."""
    imagePath = Path(path)
    try:
        payload = imagePath.read_bytes()
    except OSError as error:
        raise DecodeError(f"cannot read image {imagePath}: {error}") from error

    if payload.startswith(PNG_SIGNATURE):
        try:
            return _readPng(payload, imagePath)
        except DecodeError:
            pass

    return _readWithPillow(imagePath)


def _readPng(payload: bytes, path: Path) -> NDArray[np.uint8]:
    widthPx = heightPx = bitDepth = colorType = None
    idatChunks: list[bytes] = []
    offset = len(PNG_SIGNATURE)
    while offset < len(payload):
        if offset + 8 > len(payload):
            raise DecodeError(f"truncated PNG chunk header: {path}")
        length = struct.unpack(">I", payload[offset : offset + 4])[0]
        chunkType = payload[offset + 4 : offset + 8]
        dataStart = offset + 8
        dataEnd = dataStart + length
        if dataEnd + 4 > len(payload):
            raise DecodeError(f"truncated PNG chunk data: {path}")
        data = payload[dataStart:dataEnd]
        offset = dataEnd + 4
        if chunkType == b"IHDR":
            if length != 13:
                raise DecodeError(f"invalid PNG IHDR chunk: {path}")
            (
                widthPx,
                heightPx,
                bitDepth,
                colorType,
                compressionMethod,
                filterMethod,
                interlaceMethod,
            ) = struct.unpack(">IIBBBBB", data)
            if compressionMethod != 0 or filterMethod != 0 or interlaceMethod != 0:
                raise DecodeError(f"unsupported PNG encoding features: {path}")
        elif chunkType == b"IDAT":
            idatChunks.append(data)
        elif chunkType == b"IEND":
            break

    if None in {widthPx, heightPx, bitDepth, colorType}:
        raise DecodeError(f"missing PNG header fields: {path}")
    if bitDepth != 8:
        raise DecodeError(f"unsupported PNG bit depth {bitDepth}: {path}")

    channels = _pngChannels(int(colorType), path)
    stride = int(widthPx) * channels
    try:
        rawRows = zlib.decompress(b"".join(idatChunks))
    except zlib.error as error:
        raise DecodeError(f"cannot decompress PNG data {path}: {error}") from error

    expectedLength = int(heightPx) * (1 + stride)
    if len(rawRows) != expectedLength:
        raise DecodeError(
            "PNG scanline payload mismatch for "
            f"{path}: expected={expectedLength}, actual={len(rawRows)}"
        )

    rows = np.empty((int(heightPx), stride), dtype=np.uint8)
    previousRow = bytes(stride)
    offset = 0
    for rowIndex in range(int(heightPx)):
        filterType = rawRows[offset]
        scanline = rawRows[offset + 1 : offset + 1 + stride]
        rows[rowIndex] = _unfilterScanline(filterType, scanline, previousRow, channels)
        previousRow = bytes(rows[rowIndex])
        offset += 1 + stride

    return _pngRowsToRgb(rows, int(widthPx), int(heightPx), int(colorType))


def _pngRowsToRgb(
    rows: NDArray[np.uint8],
    widthPx: int,
    heightPx: int,
    colorType: int,
) -> NDArray[np.uint8]:
    if colorType == 2:
        return rows.reshape(heightPx, widthPx, 3).copy()
    if colorType == 6:
        rgba = rows.reshape(heightPx, widthPx, 4).astype(np.uint16, copy=False)
        alpha = rgba[..., 3:4]
        rgb = (rgba[..., :3] * alpha + 127) // 255
        return rgb.astype(np.uint8, copy=False)
    if colorType == 0:
        gray = rows.reshape(heightPx, widthPx, 1)
        return np.repeat(gray, 3, axis=2)
    if colorType == 4:
        grayAlpha = rows.reshape(heightPx, widthPx, 2).astype(np.uint16, copy=False)
        gray = (grayAlpha[..., :1] * grayAlpha[..., 1:2] + 127) // 255
        return np.repeat(gray.astype(np.uint8, copy=False), 3, axis=2)
    raise DecodeError(f"unsupported PNG color type {colorType}")


def _unfilterScanline(
    filterType: int,
    scanline: bytes,
    previousRow: bytes,
    bytesPerPixel: int,
) -> NDArray[np.uint8]:
    recon = bytearray(len(scanline))
    if filterType == 0:
        recon[:] = scanline
    elif filterType == 1:
        for index, value in enumerate(scanline):
            left = recon[index - bytesPerPixel] if index >= bytesPerPixel else 0
            recon[index] = (value + left) & 0xFF
    elif filterType == 2:
        for index, value in enumerate(scanline):
            up = previousRow[index]
            recon[index] = (value + up) & 0xFF
    elif filterType == 3:
        for index, value in enumerate(scanline):
            left = recon[index - bytesPerPixel] if index >= bytesPerPixel else 0
            up = previousRow[index]
            recon[index] = (value + ((left + up) >> 1)) & 0xFF
    elif filterType == 4:
        for index, value in enumerate(scanline):
            left = recon[index - bytesPerPixel] if index >= bytesPerPixel else 0
            up = previousRow[index]
            upLeft = previousRow[index - bytesPerPixel] if index >= bytesPerPixel else 0
            recon[index] = (value + _paeth(left, up, upLeft)) & 0xFF
    else:
        raise DecodeError(f"unsupported PNG filter type {filterType}")
    return np.frombuffer(bytes(recon), dtype=np.uint8)


def _paeth(left: int, up: int, upLeft: int) -> int:
    prediction = left + up - upLeft
    leftDelta = abs(prediction - left)
    upDelta = abs(prediction - up)
    upLeftDelta = abs(prediction - upLeft)
    if leftDelta <= upDelta and leftDelta <= upLeftDelta:
        return left
    if upDelta <= upLeftDelta:
        return up
    return upLeft


def _pngChannels(colorType: int, path: Path) -> int:
    if colorType == 0:
        return 1
    if colorType == 2:
        return 3
    if colorType == 4:
        return 2
    if colorType == 6:
        return 4
    raise DecodeError(f"unsupported PNG color type {colorType}: {path}")


def _readWithPillow(path: Path) -> NDArray[np.uint8]:
    try:
        from PIL import Image
    except ModuleNotFoundError as error:
        raise DecodeError(
            f"cannot decode image {path}: install Pillow for non-PNG formats"
        ) from error

    try:
        with Image.open(path) as image:
            return np.asarray(image.convert("RGB"), dtype=np.uint8)
    except OSError as error:
        raise DecodeError(f"cannot decode image {path}: {error}") from error


__all__ = ["readRgbImage"]
