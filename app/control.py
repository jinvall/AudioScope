"""A small control channel from the review window to the capture process.

The two are separate processes that talk through files, and until now that
channel ran one way: capture published live status, the window read it.  There
was no way to *ask* capture to change anything while it ran, so settings that
only capture can apply - the continuous-recording chunk length being the one
that matters - were fixed at launch and then unreachable.

A second file, written by the window and read by capture, completes the loop.
Deliberately not a socket: a listener would make the window a participant in
capture, and a window that has to be open for capture to work is a window that
can break capture.  A file can be absent, stale or unreadable, and the failure
mode of each is "the setting does not change", which is safe.

Two rules, both load-bearing:

* **The window never deletes.**  It only ever writes a request.  Capture
  decides whether to honour it, and a setting that is refused has to be
  refused *visibly* - see the echo in the published status.
* **Capture records what it is actually doing**, not what was requested.  The
  window displays the value in force, so a request that did not take effect
  cannot be mistaken for one that did.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Optional

#: Filename inside the recordings directory.
CONTROL_FILENAME = "control.json"

#: Shortest chunk worth recording.  Below this a chunk holds too little audio
#: to be worth a file, and the directory fills with near-empty chunks.
MIN_CHUNK_SECONDS = 10.0

#: Longest chunk.  Beyond this a single interrupted write risks a very large
#: partial file, and a crash loses more work than any realistic session needs.
MAX_CHUNK_SECONDS = 24 * 3600.0


def control_path(recordings_dir: str) -> str:
    return os.path.join(os.path.abspath(recordings_dir), CONTROL_FILENAME)


def write_control(recordings_dir: str, **settings: Any) -> Optional[str]:
    """Ask capture to change something.  Returns the path, or None.

    A request the reader cannot parse is left in place rather than removed, so
    the failure is visible in the file rather than silent.  Callers report the
    outcome in their own status line.
    """
    if not settings:
        return None
    path = control_path(recordings_dir)
    document = dict(settings)
    document["requested_at"] = time.time()
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        temporary = path + ".part"
        with open(temporary, "w", encoding="utf-8") as handle:
            json.dump(document, handle)
        os.replace(temporary, path)
        return path
    except OSError:
        return None


def read_control(recordings_dir: str) -> Optional[dict]:
    """The current request, or None.  Any failure reads as no request."""
    try:
        with open(control_path(recordings_dir), "r", encoding="utf-8") as h:
            document = json.load(h)
    except (OSError, ValueError):
        return None
    return document if isinstance(document, dict) else None


def apply_chunk_seconds(document: Optional[dict], current: float) -> Optional[float]:
    """The chunk length a control document asks for, if it is a usable one.

    Returns None when there is nothing to change, so the caller can tell "no
    request" from "a request I am honouring" from "a request I am refusing",
    which is what makes a refusal reportable.
    """
    if not document or "chunk_seconds" not in document:
        return None
    try:
        seconds = float(document["chunk_seconds"])
    except (TypeError, ValueError):
        return None
    if not (MIN_CHUNK_SECONDS <= seconds <= MAX_CHUNK_SECONDS):
        return None
    if abs(seconds - float(current)) < 1e-6:
        return None
    return seconds


class ControlWatcher:
    """Capture-side: apply requests as they appear.

    Remembers the last ``requested_at`` it acted on, so the same file being
    read five times a second does not mean re-applying it five times a second,
    and so a request that asks for a value already in force is a no-op rather
    than a spurious change.
    """

    def __init__(self, recordings_dir: str) -> None:
        self.directory = os.path.abspath(recordings_dir)
        self._lock = threading.Lock()
        self._last_seen: Optional[float] = None
        self.last_error: Optional[str] = None

    def poll(self, current_chunk_seconds: float) -> Optional[float]:
        """Return a new chunk length if one was requested and is usable."""
        document = read_control(self.directory)
        if not document:
            return None
        requested_at = document.get("requested_at")
        with self._lock:
            if requested_at is not None and requested_at == self._last_seen:
                return None
            if requested_at is not None:
                self._last_seen = requested_at
        seconds = apply_chunk_seconds(document, current_chunk_seconds)
        if seconds is None and "chunk_seconds" in document:
            self.last_error = (
                f"chunk length {document.get('chunk_seconds')!r} is outside "
                f"{MIN_CHUNK_SECONDS:g}s..{MAX_CHUNK_SECONDS:g}s and was refused"
            )
        elif seconds is not None:
            self.last_error = None
        return seconds

    def seen(self) -> Optional[float]:
        with self._lock:
            return self._last_seen
