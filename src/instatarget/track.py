"""Module entry point for the standard tracking CLI."""

from instatarget.app.track import main

__all__ = ["main"]


if __name__ == "__main__":
    raise SystemExit(main())
