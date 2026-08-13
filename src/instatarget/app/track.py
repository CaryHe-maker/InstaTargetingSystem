"""Standard tracking CLI entry."""

from __future__ import annotations

import argparse
import sys

from instatarget.app.driver import buildRuntime, finalizeSink, openSink, runTracking
from instatarget.core.config import loadConfig
from instatarget.core.errors import ConfigError, DecodeError, DepthError, GeometryError, InstaTargetError, ModelError, OutputError, ProtocolError
from instatarget.core.types import BBoxXYWH
from instatarget.data.frame_source import FrameSource

EXIT_CONFIG = 2
EXIT_DECODE = 3
EXIT_MODEL = 4
EXIT_OUTPUT = 5
EXIT_INVARIANT = 10


def buildParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m instatarget.track")
    parser.add_argument("--input", required=True)
    parser.add_argument("--init-box", required=True, help="x,y,width,height")
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--sequence-id", default=None)
    parser.add_argument("--recursive", action="store_true")
    parser.add_argument(
        "--hit-full-frame-experiment",
        action="store_true",
        help="initialize HiT from the geometry crop, then infer each later frame from one raw ERP image",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = buildParser().parse_args(argv)
    runtime = None
    source = None
    try:
        config = loadConfig(args.config)
        runtime = buildRuntime(config)
        source = FrameSource(recursive=args.recursive, sequenceId=args.sequence_id)
        source.open(args.input)
        openSink(runtime.sink, args.output)
        initialBox = _parseBox(args.init_box)
        resultCount = runTracking(
            source=source,
            initialBox=initialBox,
            geometry=runtime.geometry,
            controller=runtime.controller,
            backend=runtime.backend,
            sink=runtime.sink,
            depthProcessor=runtime.depthProcessor,
            recorder=runtime.recorder,
            hitFullFrameExperiment=args.hit_full_frame_experiment,
        )
        finalizeSink(runtime.sink, resultCount if getattr(source, "frameCount", 0) <= 0 else source.frameCount)
        return 0
    except ConfigError as error:
        _report(error)
        return EXIT_CONFIG
    except DecodeError as error:
        _report(error)
        return EXIT_DECODE
    except (ModelError, DepthError) as error:
        _report(error)
        return EXIT_MODEL
    except OutputError as error:
        _report(error)
        return EXIT_OUTPUT
    except (GeometryError, ProtocolError, InstaTargetError) as error:
        _report(error)
        return EXIT_INVARIANT
    except Exception as error:
        _report(error)
        return EXIT_INVARIANT
    finally:
        if runtime is not None:
            try:
                runtime.backend.close()
            except Exception:
                pass
        if source is not None:
            try:
                source.close()
            except Exception:
                pass


def _parseBox(text: str) -> BBoxXYWH:
    try:
        xPx, yPx, widthPx, heightPx = (float(part.strip()) for part in text.split(","))
    except ValueError as error:
        raise ConfigError("--init-box must contain four comma-separated numbers") from error
    return BBoxXYWH(xPx=xPx, yPx=yPx, widthPx=widthPx, heightPx=heightPx)


def _report(error: Exception) -> None:
    print(f"{type(error).__name__}: {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
