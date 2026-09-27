"""Where separation outputs live, and the rule that they are never overwritten.

docs/DATA_AND_STORAGE.md and docs/SOURCE_SEPARATION.md section 12: one event
may be separated repeatedly, and each attempt is independent.  A reviewer
comparing "a whisper" against "a person speaking" against "background
mechanical noise" needs all three to still exist.

So the layout is::

    events/<date>/event_000042/
      original.wav
      metadata.json
      separation_001/
        isolated.wav        <- raw model output, authoritative
        enhanced.wav        <- optional, never replaces the above
        metadata.json
      separation_002/
        ...

Allocation scans the existing directories, so an attempt number is never
reused even after the directory listing changes.  Writes are atomic, as
everywhere else in the project: a partially written separation must not look
complete.
"""

import json
import os
import re
from typing import Any, Optional

ATTEMPT_PATTERN = re.compile(r"^separation_(\d+)$")


class SeparationStore:
    """Allocates attempt directories and writes their contents."""

    def __init__(self, root: str = "events", width: int = 3) -> None:
        self.root = os.path.abspath(root)
        self.width = width

    # ------------------------------------------------------------------
    def event_directory(self, day: str, event_id: str) -> str:
        return os.path.join(self.root, day, event_id)

    def allocate_attempt(self, event_directory: str) -> str:
        """Return a fresh ``separation_NNN`` path, never an existing one.

        The scan is the source of truth: if ``separation_001`` is already
        there, the next attempt is ``separation_002`` even if the directory
        was created by a previous run of the application or by hand.
        """
        highest = 0
        if os.path.isdir(event_directory):
            for name in os.listdir(event_directory):
                match = ATTEMPT_PATTERN.match(name)
                if match:
                    highest = max(highest, int(match.group(1)))
        index = highest + 1
        path = os.path.join(
            event_directory, f"separation_{index:0{self.width}d}"
        )
        # Belt and braces: a directory created between the scan and here
        # must not be reused either.
        while os.path.exists(path):
            index += 1
            path = os.path.join(
                event_directory, f"separation_{index:0{self.width}d}"
            )
        os.makedirs(path, exist_ok=False)
        return path

    def list_attempts(self, event_directory: str) -> list[str]:
        """Existing attempt directory names, in numeric order."""
        if not os.path.isdir(event_directory):
            return []
        found = []
        for name in os.listdir(event_directory):
            match = ATTEMPT_PATTERN.match(name)
            if match and os.path.isdir(os.path.join(event_directory, name)):
                found.append((int(match.group(1)), name))
        return [name for _, name in sorted(found)]

    def count_attempts(self, event_directory: str) -> int:
        return len(self.list_attempts(event_directory))

    # ------------------------------------------------------------------
    def write_metadata_as(
        self, path: str, metadata: dict[str, Any]
    ) -> str:
        """Write metadata to an exact path, atomically."""
        directory = os.path.dirname(os.path.abspath(path))
        if directory:
            os.makedirs(directory, exist_ok=True)
        temporary = path + ".part"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(metadata, handle, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
        return path

    def write_metadata(
        self, attempt_directory: str, metadata: dict[str, Any]
    ) -> str:
        """Write ``metadata.json`` atomically."""
        return self.write_metadata_as(
            os.path.join(attempt_directory, "metadata.json"), metadata
        )

    def read_metadata(self, attempt_directory: str) -> Optional[dict]:
        path = os.path.join(attempt_directory, "metadata.json")
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as handle:
                return json.load(handle)
        except (OSError, json.JSONDecodeError):
            return None

    def find_day(self, event_id: str) -> Optional[str]:
        """Locate the day directory for an event id, if the tree has it.

        The GUI knows an event id and a timestamp but not necessarily the
        directory, and separation must write beside the original rather than
        guess a fresh tree.
        """
        if not os.path.isdir(self.root):
            return None
        for day in sorted(os.listdir(self.root)):
            candidate = os.path.join(self.root, day, event_id)
            if os.path.isdir(candidate):
                return day
        return None
