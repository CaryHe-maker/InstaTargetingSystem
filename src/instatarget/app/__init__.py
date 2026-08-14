"""Application entry package."""

from instatarget.app.driver import RuntimeBundle, buildRuntime, runTracking


def __getattr__(name: str):
    """Load non-competition entrypoints only when explicitly requested."""
    if name == "trackMain":
        from instatarget.app.track import main

        return main
    if name == "trackAirSim360Main":
        from instatarget.app.track_airsim360 import main

        return main
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = ["RuntimeBundle", "buildRuntime", "runTracking", "trackAirSim360Main", "trackMain"]
