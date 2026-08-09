"""AirSim360 tracking CLI entry."""

from __future__ import annotations

import argparse
import sys

from instatarget.app.driver import buildRuntime, finalizeSink, openSink, runTracking
from instatarget.core.config import loadConfig
from instatarget.core.errors import ConfigError, DecodeError, DepthError, GeometryError, InstaTargetError, ModelError, OutputError, ProtocolError
from instatarget.data.airsim360_source import AirSim360DataSource
from instatarget.data.pseudo_track_builder import PseudoTrackBuilder

EXIT_CONFIG = 2
EXIT_DECODE = 3
EXIT_MODEL = 4
EXIT_OUTPUT = 5
EXIT_INVARIANT = 10


def buildParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m instatarget.track_airsim360")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--sequence", required=True)
    parser.add_argument("--target-instance", required=True, type=int)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = buildParser().parse_args(argv)
    source = AirSim360DataSource()
    runtime = None
    try:
        config = loadConfig(args.config)
        runtime = buildRuntime(config)
        source.open(args.dataset_root, args.sequence)
        initialFrame = source.read()
        if initialFrame is None:
            raise DecodeError("AirSim360 sequence is empty")
        initialBox = PseudoTrackBuilder().buildInitialBox(initialFrame, args.target_instance)
        source.close()
        source.open(args.dataset_root, args.sequence)
        openSink(runtime.sink, args.output)
        resultCount = runTracking(
            source=source,
            initialBox=initialBox,
            geometry=runtime.geometry,
            controller=runtime.controller,
            backend=runtime.backend,
            sink=runtime.sink,
            depthProcessor=runtime.depthProcessor,
            recorder=runtime.recorder,
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
        try:
            source.close()
        except Exception:
            pass


def _report(error: Exception) -> None:
    print(f"{type(error).__name__}: {error}", file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
