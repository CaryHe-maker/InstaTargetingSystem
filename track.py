#!/usr/bin/env python3
"""No-argument entry point for the official competition container."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

runCompetition = importlib.import_module("instatarget.app.competition").runCompetition


if __name__ == "__main__":
    try:
        raise SystemExit(runCompetition())
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        raise SystemExit(1) from error
