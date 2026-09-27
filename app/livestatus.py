"""Live capture status, published for the review window to read.

The capture process and the review window are separate processes: the launcher
starts capture as a child, and the window reads the database.  There is
therefore no in-process way for the window to ask "is the phone actually
sending audio right now?", and that question is the one an operator asks
first when the app looks inert.

The answer is a small JSON file, written atomically and polled.  Deliberately a
file rather than a socket or a pipe:

* the processes are started independently and may outlive each other;
* a reader that is not running must not affect the writer;
* it survives a window restart, so the window can report "capture is
  connected" immediately on open instead of waiting for the next publish.

What is published, and why each field earns its place:

* ``state`` - the operator's question, answered directly: waiting, connected,
  streaming, or stalled.  ``stalled`` is the one that matters: a socket can be
  open while the sender has stopped sending, and a monitor that only showed
  "connected" would say everything is fine while nothing is recorded.
* ``since_bytes_changed`` - the evidence behind ``stalled``, measured rather
  than inferred.
* ``wire_rate_hz`` and ``wire_rate_origin`` - the rate the stream is being read
  at, and whether the sender *declared* it or it is an assumption.  This is
  limitation 3 in STATE.md: the wire is raw PCM with no header, so a wrong
  assumption is silent everywhere else.
* ``level_dbfs`` / ``peak_dbfs`` and ``envelope`` - the live audio, taken from
  the same analysis frames the detector consumes, so the meter cannot disagree
  with what was measured.
* ``seconds_received`` and ``bytes_received`` - progress, from the source's own
  counters rather than counted again here.

The publisher is deliberately cheap.  It is called on the analysis worker for
the level trace, and polled at a few hertz for everything else; the file write
is throttled and atomic so a reader can never observe half a document.
"""

from __future__ import annotations

import json
import os
import threading
import time
from typing import Any, Optional

#: Filename inside the recordings directory.
LIVE_STATUS_FILENAME = "live_status.json"

#: Bumped when the document shape changes, so a reader can tell.
LIVE_STATUS_VERSION = 1

#: How often the file is rewritten.  Fast enough to feel live, slow enough
#: that the write is not a measurable cost on the capture process.
PUBLISH_INTERVAL_SECONDS = 0.2

#: A sender that has not advanced its byte counter for this long is treated as
#: stalled.  Generous, because a phone under load can pause for a second; a
#: tighter bound would flap the indicator and train the operator to ignore it.
STALE_SECONDS = 2.0

#: Envelope columns published per strip.  At 10 ms hops this is 2.4 s of
#: history at full rate, which is a useful glance without being a recording.
ENVELOPE_COLUMNS = 240

STATE_WAITING = "waiting_for_client"
STATE_CONNECTED = "connected"
STATE_STREAMING = "streaming"
STATE_STALLED = "stalled"
STATE_STOPPED = "stopped"


class LiveStatusPublisher:
    """Collects live capture state and publishes it as JSON.

    Thread-safe: the level trace arrives on the analysis worker thread while
    the connection facts are polled from the capture loop.
    """

    def __init__(
        self,
        directory: str,
        sample_rate: int = 48_000,
        source: Any = None,
        envelope_columns: int = ENVELOPE_COLUMNS,
        publish_interval: float = PUBLISH_INTERVAL_SECONDS,
    ) -> None:
        self.directory = os.path.abspath(directory)
        self.path = os.path.join(self.directory, LIVE_STATUS_FILENAME)
        self.sample_rate = int(sample_rate)
        self.envelope_columns = int(envelope_columns)
        self.publish_interval = float(publish_interval)

        self._lock = threading.Lock()
        self._source = source
        self._levels: list[float] = []      # dBFS per frame, oldest first
        self._level_dbfs: Optional[float] = None
        self._peak_dbfs: Optional[float] = None
        self._last_bytes: Optional[int] = None
        self._last_advance = time.time()
        self._last_publish = 0.0
        self._connected_once = False
        self._stopped = False
        self.published = 0
        self.publish_errors = 0

    # ------------------------------------------------------------------
    def attach_source(self, source: Any) -> None:
        """Bind the audio source, for the connection facts."""
        with self._lock:
            self._source = source

    def on_frame(self, result: Any) -> None:
        """Record one analysis frame's level.

        Wired to ``AnalysisWorker``'s ``on_result``, so the trace is the same
        measurement the detector makes - not a second, cheaper estimate that
        could disagree with it.
        """
        try:
            rms_db = float(result.rms_db)
            peak_db = float(result.features.peak_db)
        except Exception:
            # A malformed result must never take the analysis worker down.
            return
        if rms_db != rms_db or peak_db != peak_db:   # NaN
            return
        with self._lock:
            self._levels.append(rms_db)
            if len(self._levels) > self.envelope_columns:
                del self._levels[: len(self._levels) - self.envelope_columns]
            self._level_dbfs = rms_db
            self._peak_dbfs = peak_db

    def mark_stopped(self) -> None:
        with self._lock:
            self._stopped = True

    # ------------------------------------------------------------------
    def snapshot(self) -> dict[str, Any]:
        """The document to publish.  Never guesses a value it does not have."""
        with self._lock:
            source = self._source
            stopped = self._stopped
            levels = list(self._levels)
            level = self._level_dbfs
            peak = self._peak_dbfs
            last_bytes = self._last_bytes
            last_advance = self._last_advance
            connected_once = self._connected_once
            # The byte counter is re-read inside the lock, but from the
            # source, so the snapshot is consistent with itself.
            clients = _client_facts(source)
            total_bytes = sum(c["bytes_received"] for c in clients)
            if last_bytes is None or total_bytes != last_bytes:
                last_bytes = total_bytes
                last_advance = time.time()
                self._last_bytes = total_bytes
                self._last_advance = last_advance
            if clients:
                self._connected_once = True

            now = time.time()
            silence = now - last_advance
            if stopped:
                state = STATE_STOPPED
            elif not clients:
                state = STATE_WAITING
            elif total_bytes == 0:
                # Connected but nothing has arrived yet.  This is a normal,
                # brief state at the start of every session, and calling it
                # "stalled" would raise a false alarm the moment the phone
                # connects.
                state = (
                    STATE_CONNECTED if silence < STALE_SECONDS
                    else STATE_STALLED
                )
            elif silence > STALE_SECONDS:
                state = STATE_STALLED
            else:
                state = STATE_STREAMING

            document: dict[str, Any] = {
                "version": LIVE_STATUS_VERSION,
                "state": state,
                "published_at": now,
                "silence_seconds": round(silence, 3),
                "connected_once": connected_once,
                "bytes_received": total_bytes,
                "internal_sample_rate": self.sample_rate,
                "level_dbfs": round(level, 2) if level is not None else None,
                "peak_dbfs": round(peak, 2) if peak is not None else None,
                "envelope": [round(value, 2) for value in levels],
                "envelope_hop_ms": None,
                "clients": clients,
            }
            document.update(_source_facts(source, clients))
        return document

    # ------------------------------------------------------------------
    def publish(self, force: bool = False) -> bool:
        """Write the document if the interval has elapsed.  Returns True if written.

        The write is atomic: a temporary file in the same directory followed by
        a rename, so a reader either sees the previous complete document or the
        new one, never a truncated file.  The window polls this from a timer,
        and a half-written file would look like capture failing.
        """
        now = time.time()
        if not force and now - self._last_publish < self.publish_interval:
            return False
        document = self.snapshot()
        try:
            os.makedirs(self.directory, exist_ok=True)
            temporary = self.path + ".part"
            with open(temporary, "w", encoding="utf-8") as handle:
                json.dump(document, handle)
            os.replace(temporary, self.path)
            self._last_publish = now
            self.published += 1
            return True
        except OSError:
            # Publication is a convenience for the UI.  Capture must not care.
            self.publish_errors += 1
            return False

    def clear(self) -> None:
        """Remove the document, so a stale file cannot outlive the process.

        Without this, a closed capture process would leave a file claiming to
        be live, and the window would report a connection that ended hours ago.
        """
        self.mark_stopped()
        try:
            os.remove(self.path)
        except OSError:
            pass


# ----------------------------------------------------------------------
def read_live_status(path: str) -> Optional[dict[str, Any]]:
    """Read a published document, or ``None`` if there is nothing to read.

    Any failure - missing, half-written, truncated by a kill, wrong JSON - reads
    as absent rather than raising.  The window's job is to show the truth about
    capture, and "I cannot tell" is a legitimate answer that it must be able to
    display.
    """
    try:
        with open(path, "r", encoding="utf-8") as handle:
            document = json.load(handle)
    except (OSError, ValueError):
        return None
    return document if isinstance(document, dict) else None


# ----------------------------------------------------------------------
def _client_facts(source: Any) -> list[dict[str, Any]]:
    """Per-client counters, read defensively.

    The source is the authority for these numbers, so they are taken from it
    rather than counted again; anything unreadable is reported as unknown
    rather than defaulted to zero, because a zero would read as "connected and
    silent" when it may mean "not measurable".
    """
    if source is None:
        return []
    clients = getattr(source, "clients", None) or []
    facts = []
    for client in clients:
        try:
            facts.append({
                "address": str(client.address),
                "bytes_received": int(client.bytes_received),
                "seconds_received": round(float(client.seconds_received), 2),
                "dropped_bytes": int(getattr(client, "dropped_bytes", 0)),
                "config": dict(getattr(client, "config", {}) or {}),
                "amplification": getattr(client, "amplification", None),
            })
        except Exception:
            continue
    return facts


def _source_facts(source: Any, clients: list) -> dict[str, Any]:
    """Stream-level facts: the port, the wire rate, and its provenance.

    ``wire_rate_origin`` is the field that matters.  ``declared`` means the
    sender said what rate it was using; ``assumed`` means we are taking its
    word for 44 100 Hz, and everything recorded since is derived from that
    assumption.  The window shows it rather than burying it, because this is the
    one place in the capture chain where being wrong is silent.
    """
    facts: dict[str, Any] = {
        "source": type(source).__name__ if source is not None else None,
        "port": None,
        "wire_rate_hz": None,
        "wire_rate_origin": None,
        "wire_format": None,
        "converting": None,
    }
    if source is None:
        return facts
    try:
        status = source.status()
    except Exception:
        return facts
    if not isinstance(status, dict):
        return facts
    for key in ("port", "wire_rate_hz", "wire_rate_origin", "wire_format",
                "converting"):
        if key in status:
            facts[key] = status[key]
    facts["reserved_ports"] = status.get("reserved_ports")
    facts["reserved_failed"] = status.get("reserved_failed")
    return facts
