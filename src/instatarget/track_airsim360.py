"""Module entry point for the AirSim360 tracking CLI."""

from instatarget.app.track_airsim360 import main

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
