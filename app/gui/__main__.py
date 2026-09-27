"""``python -m app.gui`` - the event review workstation."""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional

# Qt must be imported after argv is known, and the platform plugin can be forced
# for headless verification before QApplication is constructed.
from .app import launch, main

__all__ = ["launch", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
