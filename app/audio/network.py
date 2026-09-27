"""Network audio source: raw PCM stream from an Android device.

Wire protocol
-------------
The primary audio source is a live TCP stream on port **8090**.  The protocol
is the one the existing AMP receiver expects
(``/home/jason/amp/server/audio_receiver.py``), so the same phone can talk to
either application:

1. The client connects to port 8190.
2. The client *optionally* sends one newline-terminated JSON configuration
   object, then
3. the client streams **raw, unframed** signed 16-bit little-endian PCM at
   **44 100 Hz, mono**, with no length prefix and no end marker.

The server sends nothing back on this port.  A client that simply disconnects
has ended its stream.

The optional config line is detected, not assumed: if the connection starts
with ``{`` and a newline arrives within the first 4 KiB, the bytes up to that
newline are parsed as JSON and everything after it is audio.  Otherwise the
connection is treated as raw PCM from the first byte.  Binary PCM is not
reliably distinguishable from text, so this mirrors the existing receiver's
heuristic rather than pretending to a certainty it does not have.

Conversion
----------
The wire format is 16-bit PCM at 44 100 Hz.  The internal format is float32 at
48 000 Hz (``app/config.py``).  Both conversions happen once, here at the input
boundary, and nothing downstream deals with 16-bit or 44.1 kHz:

    s16le bytes -> float32 (/ 32768) -> mono -> 48 kHz

Int16 is divided by 32768 rather than 32767, so the *scaling* step maps the
wire range into (-1.0, +1.0) and never produces exactly full scale.

Note that this does not bound the final samples: the anti-alias filter has
finite stopband rejection, so a discontinuous waveform (a square wave, a
clipped syllable) produces Gibbs overshoot above 1.0 after resampling.  That is
correct filter behaviour, not a bug.  It is harmless here because evidence is
stored as float32 WAV, which holds values beyond +/-1.0 without clipping, so an
overshoot is recorded faithfully instead of being silently flattened.

Amplification
-------------
A config line may carry ``amplification``.  That value is **recorded and
reported, never applied to the stored audio** (AGENTS.md section 2.1): the
recording is evidence, and a gain applied at capture time would be baked in
permanently.  Apply it at playback instead.

Port reservation
----------------
While streaming, ports **806[0-4]-806[0-4]** are reserved by binding and holding them,
so "those ports belong to this session" is true at the OS level rather than
merely documented.  ``SO_REUSEADDR`` is deliberately not set, so a port already
owned elsewhere is a hard, immediate error instead of a surprise later.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

import numpy as np

from ..config import AppConfig
from .capture import AudioSource

# The primary audio stream port.
#
# Deliberately NOT 8090-8099: the AMP receiver (/home/jason/amp/server/
# audio_receiver.py) binds PCM, control and visualization listeners inside
# 8090..8099, and the 808x range is congested on this host.  Two applications
# cannot hold the same port, so this project takes a disjoint range and keeps
# the same shape AMP uses (one stream port, five reserved 30 below it).
STREAM_PORT = 8190

# Reserved for this session while the stream is active.
RESERVED_PORTS = (8060, 8061, 8062, 8063, 8064)

# Wire format, matching the sender.
#
# The sample rate is an ASSUMPTION, not a negotiated value.  The sender
# (/home/jason/amp/android-app/.../AudioStreamerService.kt) captures with
# AudioRecord at 44100 for microphone-class sources but falls back to 16000 and
# then 8000 for Bluetooth and wired-headset sources, and its config line does
# not carry the rate.  So the rate is *learned* if the sender ever declares it
# (a "sample_rate" or "rate" key) and otherwise assumed, and every event records
# which of the two it was - because getting this wrong is silent and would
# corrupt every duration, spectrum and fingerprint downstream.
WIRE_SAMPLE_RATE = 44_100
RATE_CONFIG_KEYS = ("sample_rate", "rate", "samplerate")
WIRE_CHANNELS = 1
WIRE_SAMPLE_WIDTH = 2  # bytes
WIRE_DTYPE = "<i2"

BYTES_PER_SECOND = WIRE_SAMPLE_RATE * WIRE_CHANNELS * WIRE_SAMPLE_WIDTH

# An initial read with no newline in this many bytes means the client is
# streaming raw PCM with no config line.
CONFIG_PROBE_BYTES = 4096
# How long to keep looking for the optional config line once a client has
# connected.  This is a one-off cost at connection setup, not a per-frame one,
# so it can afford to be generous: at 2 s a client that was merely slow to
# start sending - which happens under load - lost its config line
# permanently, because there is no second chance once the probe gives up.
CONFIG_PROBE_TIMEOUT = 5.0

# 250 ms of audio.  Large enough to absorb scheduling jitter, small enough
# that a stalled client does not add noticeable latency.
RECV_SIZE = 64 * 1024

# Bounded buffer.  A client faster than the pipeline must not grow memory
# without limit (docs/PERFORMANCE.md section 7).  4 MiB of float32 at 48 kHz
# is a little over 20 s of audio.
MAX_QUEUED_FRAMES = 4 * 1024 * 1024 // 4


class NetworkError(RuntimeError):
    """Raised for unusable port configuration or unrecoverable setup errors."""


@dataclass
class ClientState:
    """What we know about one connected sender.

    Each client owns its own resampler.  A single shared one would interleave
    two senders into a single filter stream, and would make a disconnect
    ambiguous: flushing the shared filter for a sender that just left would
    corrupt the audio of a sender that is still connected.
    """

    address: str
    config: dict[str, Any] = field(default_factory=dict)
    resampler: Any = None
    bytes_received: int = 0
    frames_delivered: int = 0
    connected_at: float = field(default_factory=time.time)
    dropped_bytes: int = 0
    tail_samples_dropped: int = 0

    @property
    def seconds_received(self) -> float:
        return self.bytes_received / BYTES_PER_SECOND

    @property
    def amplification(self) -> Optional[float]:
        value = self.config.get("amplification")
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    def describe(self) -> str:
        amp = self.amplification
        amp_text = f", amplification={amp:g}" if amp is not None else ""
        return (
            f"{self.address}: {self.seconds_received:.1f}s received, "
            f"{len(self.config)} config key(s){amp_text}"
        )


class PortReservation:
    """Holds a set of TCP ports for the lifetime of this object.

    A port that cannot be bound is recorded rather than skipped: silently not
    reserving a port would defeat the purpose of reserving it.
    """

    def __init__(self, ports=RESERVED_PORTS) -> None:
        self.ports = tuple(ports)
        self._sockets: list[socket.socket] = []
        self._held: list[int] = []
        self.errors: dict[int, str] = {}

    @property
    def held(self) -> tuple[int, ...]:
        """Ports currently bound and held by this reservation."""
        return tuple(self._held)

    @property
    def failed(self) -> dict[int, str]:
        return dict(self.errors)

    def acquire(self) -> tuple[int, ...]:
        """Bind every port.  Raises if none could be bound.

        Partial success is allowed and reported: the return value says what is
        actually held.
        """
        for port in self.ports:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # Deliberately not SO_REUSEADDR: a hard conflict is the point.
            try:
                sock.bind(("0.0.0.0", port))
                sock.listen(1)
            except OSError as exc:
                self.errors[port] = str(exc)
                sock.close()
                continue
            self._sockets.append(sock)
            self._held.append(port)
        if not self._sockets:
            raise NetworkError(
                f"could not reserve any of ports {list(self.ports)}: {self.errors}"
            )
        return self.held

    def release(self) -> None:
        while self._sockets:
            sock = self._sockets.pop()
            if self._held:
                self._held.pop()
            try:
                sock.close()
            except OSError:
                pass

    def __enter__(self) -> "PortReservation":
        self.acquire()
        return self

    def __exit__(self, *exc) -> None:
        self.release()


class NetworkSource(AudioSource):
    """Accepts streaming clients on port 8090 and emits internal-format audio.

    One client streams at a time.  A second concurrent connection is refused
    with a clear message rather than mixed in.

    Summing two senders sounds obvious and is wrong here: the chunks arrive on
    independent TCP connections with independent jitter, so pairing "one chunk
    from each per read" adds samples captured at different wall-clock instants.
    That produces a signal that is neither device's audio and quietly corrupts
    the evidence.  Refusing is honest; a plausible-looking bad mix is not.
    Multi-device capture needs proper time alignment, which is a separate piece
    of work.
    """

    def __init__(
        self,
        config: AppConfig,
        port: int = STREAM_PORT,
        host: str = "0.0.0.0",
        reserve_ports: bool = True,
        reserved: tuple[int, ...] = RESERVED_PORTS,
        on_status: Optional[Callable[[str], None]] = None,
    ) -> None:
        super().__init__(config)
        self.port = int(port)
        self.host = host
        self.on_status = on_status or (lambda message: None)
        self.reservation: Optional[PortReservation] = (
            PortReservation(reserved) if reserve_ports else None
        )

        self._listener: Optional[socket.socket] = None
        self._accept_thread: Optional[threading.Thread] = None
        self._clients: dict[int, ClientState] = {}
        self._client_socks: dict[int, socket.socket] = {}
        self._client_threads: list[threading.Thread] = []
        self._lock = threading.Lock()

        # Decoded + resampled audio from the active client.
        self._out: list[np.ndarray] = []
        self._out_cond = threading.Condition()
        self._active_key: int | None = None

        # The most recent client seen, kept after it disconnects.  An event can
        # be *decided* just after the sender has gone - the release period has
        # to elapse first - and at that point the live client list is empty, so
        # without this the event's provenance would say "awaiting-client" and
        # the config that actually applied to that audio would be lost.
        self._last_client_address: Optional[str] = None
        self._last_client_configs: list = []
        self._declared_rate: Optional[int] = None
        # How the current client's rate was established, for event provenance.
        self._rate_origin = "assumed"

        self._resampler = self._make_resampler(WIRE_SAMPLE_RATE)
        self._client_rate = WIRE_SAMPLE_RATE
        self._bytes_since_second = 0
        self._running = False

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------
    @property
    def wire_rate_declared(self) -> bool:
        """True when the sender told us its rate rather than us assuming it."""
        with self._lock:
            return bool(self._declared_rate)

    @property
    def declared_rate(self) -> Optional[int]:
        with self._lock:
            return self._declared_rate

    @property
    def name(self) -> str:
        count = len(self._clients)
        if count == 0:
            # Fall back to the last client that was actually seen, so an event
            # decided shortly after a disconnect still describes its sender.
            if self._last_client_address is not None:
                return f"network:{self._last_client_address}"
            return "network:awaiting-client"
        if count == 1:
            return f"network:{next(iter(self._clients.values())).address}"
        return f"network:{count}-clients"

    @property
    def native_sample_rate(self) -> int:
        return WIRE_SAMPLE_RATE

    @property
    def is_converting(self) -> bool:
        return self._resampler is not None

    @property
    def client_count(self) -> int:
        with self._lock:
            return len(self._clients)

    @property
    def clients(self) -> list[ClientState]:
        with self._lock:
            return list(self._clients.values())

    @property
    def client_configs(self) -> list[dict[str, Any]]:
        with self._lock:
            live = [dict(c.config) for c in self._clients.values() if c.config]
        if live:
            return live
        # No client connected at this instant; report the configuration that
        # applied to the audio most recently received.
        with self._lock:
            return [dict(c) for c in self._last_client_configs]

    @property
    def connected(self) -> bool:
        return self.client_count > 0

    @property
    def reserved_ports(self) -> tuple[int, ...]:
        return self.reservation.held if self.reservation else ()

    @property
    def output_device(self) -> str:
        if not self.connected:
            return f"network stream on port {self.port} (no client)"
        parts = [c.address for c in self.clients]
        joined = ", ".join(parts)
        return (
            f"network stream {joined} @ {WIRE_SAMPLE_RATE} Hz s16le "
            f"(converted to {self.sample_rate} Hz float32)"
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def start(self) -> None:
        """Bind the reserved ports and begin listening.

        The reserved ports are taken *before* the stream port is offered, so
        that by the time a client can connect, 806[0-4]-806[0-4] already belong to this
        session.

        Idempotent: the CLI starts the source so it can accept a client before
        the pipeline exists, and the pipeline calls ``start()`` again.  Without
        this guard the second call would re-bind ports this object already
        holds and fail against itself.
        """
        if self._listener is not None:
            return
        self._stop.clear()
        self._running = True

        if self.reservation is not None:
            held = self.reservation.acquire()
            if self.reservation.failed:
                self.on_status(
                    "port reservation incomplete: "
                    + ", ".join(
                        f"{p}: {e}" for p, e in self.reservation.failed.items()
                    )
                )
            self.on_status(f"reserved ports {list(held)}")

        listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            listener.bind((self.host, self.port))
            # A backlog of 1 gives one pending connection; a retrying client
            # would then be refused even though the server is healthy.
            listener.listen(8)
            listener.settimeout(0.5)
        except OSError as exc:
            listener.close()
            if self.reservation:
                self.reservation.release()
            raise NetworkError(f"cannot listen on port {self.port}: {exc}") from exc
        self._listener = listener
        self.on_status(
            f"listening on port {self.port} "
            f"({WIRE_SAMPLE_RATE} Hz s16le mono, converted to "
            f"{self.sample_rate} Hz)"
        )

        self._accept_thread = threading.Thread(
            target=self._accept_loop, name="network-accept", daemon=True
        )
        self._accept_thread.start()

    def _accept_loop(self) -> None:
        while self._running and not self._stop.is_set():
            listener = self._listener
            if listener is None:
                break
            try:
                conn, addr = listener.accept()
            except socket.timeout:
                continue
            except OSError:
                break

            # One malformed or racing client must never be able to kill the
            # accept loop; the whole source would go deaf silently otherwise.
            try:
                self._handle_new_client(conn, addr)
            except Exception as exc:  # pragma: no cover - defensive
                self.on_status(
                    f"accept loop: failed to handle a connection from "
                    f"{addr[0]}:{addr[1]}: {type(exc).__name__}: {exc}"
                )
                try:
                    conn.close()
                except OSError:
                    pass

    def _handle_new_client(self, conn: socket.socket, addr) -> None:
        """Register a new connection, or refuse it if one is already active."""
        state = ClientState(
            address=f"{addr[0]}:{addr[1]}",
            resampler=self._make_resampler(WIRE_SAMPLE_RATE),
        )
        with self._lock:
            active_state = (
                self._clients.get(self._active_key)
                if self._active_key is not None
                else None
            )
            if active_state is not None:
                # Refuse rather than mix.  See the class docstring.
                try:
                    conn.sendall(b"ERR server is already streaming\n")
                except OSError:
                    pass
                try:
                    conn.close()
                except OSError:
                    pass
                self.on_status(
                    f"refused {state.address}: a client is already streaming "
                    f"from {active_state.address}; two senders are not mixed "
                    "(they cannot be aligned)"
                )
                return
            key = conn.fileno()
            self._active_key = key
            self._clients[key] = state
            self._client_socks[key] = conn
        with self._lock:
            if self._last_client_address is None:
                self._last_client_address = state.address
        self.on_status(f"client connected: {state.address}")
        thread = threading.Thread(
            target=self._client_reader,
            args=(key, conn, state),
            name=f"network-client-{key}",
            daemon=True,
        )
        thread.start()
        self._client_threads.append(thread)

    # ------------------------------------------------------------------
    def _client_reader(self, key: int, conn: socket.socket, state: ClientState) -> None:
        """Read one client: optional JSON config, then raw PCM forever."""
        try:
            leftover = self._read_optional_config(conn, state)
            if leftover:
                self._ingest(key, state, leftover)
            while self._running and not self._stop.is_set():
                try:
                    chunk = conn.recv(RECV_SIZE)
                except socket.timeout:
                    continue
                except OSError as exc:
                    self.on_status(f"client {state.address} recv error: {exc}")
                    break
                if not chunk:
                    break
                self._ingest(key, state, chunk)
        finally:
            self._flush_client(key, state)
            with self._lock:
                self._clients.pop(key, None)
                self._client_socks.pop(key, None)
                if self._active_key == key:
                    self._active_key = None
            try:
                conn.close()
            except OSError:
                pass
            self.on_status(
                f"client disconnected: {state.address} "
                f"after {state.seconds_received:.1f}s"
            )

    def _flush_client(self, key: int, state: ClientState) -> None:
        """Emit whatever the departing client's resampler still owes.

        The streaming resampler holds back up to ``down - 1`` input samples for
        rate alignment, so up to ~3 ms of real audio would otherwise be lost
        when a sender disconnects.  Flushing is safe here precisely because
        each client owns its resampler: doing this to a shared filter would
        corrupt any other sender still connected.
        """
        resampler = state.resampler
        if resampler is None:
            return
        try:
            tail = resampler.flush_input()
        except Exception as exc:  # pragma: no cover - defensive
            self.on_status(f"resampler flush failed for {state.address}: {exc}")
            return
        if tail.size:
            state.frames_delivered += int(tail.size)
            with self._out_cond:
                self._out.append(np.ascontiguousarray(tail, dtype=np.float32))
                self._out_cond.notify_all()
        remaining = int(getattr(resampler, "pending_input_samples", 0))
        if remaining:
            state.tail_samples_dropped = remaining
            self.on_status(
                f"client {state.address}: dropped {remaining} tail sample(s) "
                f"(under 3 ms) held for rate alignment"
            )

    def _read_optional_config(
        self, conn: socket.socket, state: ClientState
    ) -> bytes:
        """Consume an optional JSON config line; return any PCM that followed.

        Returns b'' when the client is sending raw PCM.  Mirrors the AMP
        receiver: a leading ``{`` plus a newline inside the probe window means
        JSON; anything else is audio from the first byte.
        """
        buffer = bytearray()
        deadline = time.monotonic() + CONFIG_PROBE_TIMEOUT
        conn.settimeout(0.5)
        while time.monotonic() < deadline and len(buffer) < CONFIG_PROBE_BYTES:
            try:
                chunk = conn.recv(1024)
            except socket.timeout:
                # A quiet moment is not an answer.  Returning here would
                # permanently give up on the config line and treat a late but
                # perfectly good client as raw-PCM-only.  Keep waiting until
                # the overall probe deadline expires.
                continue
            except OSError:
                return bytes(buffer)
            if not chunk:
                return bytes(buffer)
            buffer.extend(chunk)

            index = buffer.find(b"\n")
            if index >= 0:
                head = bytes(buffer[:index]).decode("utf-8", "replace").strip()
                leftover = bytes(buffer[index + 1:])
                if head.startswith("{"):
                    try:
                        state.config = json.loads(head)
                        declared = _rate_from_config(state.config)
                        with self._lock:
                            self._last_client_configs = [dict(state.config)]
                            self._last_client_address = state.address
                            if declared:
                                # A declared rate overrides the assumption and
                                # re-targets the resampler, so a sender that
                                # fell back to a lower rate is not misread.
                                if declared != self._client_rate:
                                    state.resampler = self._make_resampler(
                                        declared
                                    )
                                self._client_rate = declared
                                self._declared_rate = declared
                                self._rate_origin = "declared"
                            else:
                                self._client_rate = WIRE_SAMPLE_RATE
                                self._declared_rate = None
                                self._rate_origin = "assumed"
                    except (ValueError, TypeError) as exc:
                        self.on_status(
                            f"client {state.address} sent invalid JSON config "
                            f"({exc}); ignoring it and treating the stream as PCM"
                        )
                    else:
                        self.on_status(
                            f"client {state.address} config: "
                            f"{json.dumps(state.config, sort_keys=True)}"
                        )
                    return leftover
                # Not a config line.  The newline was a byte of the audio
                # stream, so *everything* read so far is audio - including the
                # bytes before the newline.  Returning only `leftover` would
                # silently discard the head of the recording.
                self.on_status(
                    f"client {state.address} sent no config line; "
                    "treating the stream as raw PCM"
                )
                return bytes(buffer)
            if not bytes(buffer[:1]).startswith(b"{"):
                # Definitely not a JSON object; it is audio.
                return bytes(buffer)

        # No newline within the probe window.
        return bytes(buffer)

    # ------------------------------------------------------------------
    def _ingest(self, key: int, state: ClientState, chunk: bytes) -> None:
        """Decode, downmix, resample, and queue one wire chunk."""
        usable = len(chunk) - (len(chunk) % WIRE_SAMPLE_WIDTH)
        if usable == 0:
            return
        data = np.frombuffer(chunk[:usable], dtype=WIRE_DTYPE)
        # int16 -> float32 in [-1, 1), divided by 32768 so the result can never
        # reach +/-1.0 and clip on conversion.
        mono = (data.astype(np.float32) / np.float32(32768.0)).copy()

        if state.resampler is not None:
            mono = state.resampler.process(mono)

        state.bytes_received += usable
        state.frames_delivered += int(mono.size)

        with self._out_cond:
            queue = self._out
            queue.append(mono)
            # Bound each client's queue.  A client faster than the pipeline
            # must not grow memory without limit (docs/PERFORMANCE.md 7).
            # Dropping the *oldest* audio suits a live monitor, and the drop is
            # counted so the loss is visible rather than silent.
            limit = MAX_QUEUED_FRAMES
            queued = 0
            while len(queue) > 1 and (queued := sum(b.size for b in queue)) > limit:
                dropped = queue.pop(0)
                state.dropped_bytes += int(dropped.size) * WIRE_SAMPLE_WIDTH
            self._out_cond.notify_all()

    # ------------------------------------------------------------------
    def read(
        self, frames: Optional[int] = None, timeout: Optional[float] = None
    ) -> Optional[np.ndarray]:
        """Return the next block of internal-format audio.

        With no ``timeout`` this blocks until audio arrives or the source is
        stopped, which is what the capture pipeline wants: a live source has
        no end of stream, and returning early would tear the pipeline down.

        With a ``timeout``, a quiet period returns None instead of blocking,
        so a caller can poll.  In that mode None means "nothing right now",
        not "finished" - check :attr:`is_running` to tell them apart.

        Successive blocks from one client are consecutive in time and are never
        added to each other.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        with self._out_cond:
            while not self._out:
                if self._stop.is_set() or not self._running:
                    return None
                remaining = 0.5 if deadline is None else deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._out_cond.wait(timeout=remaining)

            block = self._out.pop(0)
        return np.ascontiguousarray(block, dtype=np.float32)

    def drain(self) -> list[np.ndarray]:
        """Take everything currently queued without waiting."""
        with self._out_cond:
            blocks = self._out
            self._out = []
            return blocks

    # ------------------------------------------------------------------
    def wait_for_client(self, timeout: Optional[float] = None) -> bool:
        """Wait until at least one client is streaming.

        The audio port is passive, so this only observes; the accept loop runs
        in the background.
        """
        deadline = None if timeout is None else time.monotonic() + timeout
        while not self._stop.is_set():
            if self.connected:
                return True
            if deadline is not None and time.monotonic() >= deadline:
                return False
            time.sleep(0.05)
        return False

    def stop(self) -> None:
        super().stop()
        self._running = False
        with self._out_cond:
            self._out_cond.notify_all()

    def close(self) -> None:
        self.stop()
        with self._lock:
            socks = list(self._client_socks.values())
            self._client_socks.clear()
        for sock in socks:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                sock.close()
            except OSError:
                pass
        listener, self._listener = self._listener, None
        if listener is not None:
            try:
                listener.close()
            except OSError:
                pass
        thread, self._accept_thread = self._accept_thread, None
        if thread is not None:
            thread.join(timeout=2.0)
        if self.reservation is not None:
            self.reservation.release()

    # ------------------------------------------------------------------
    def status(self) -> dict[str, Any]:
        """Observable state, for the CLI and the UI status panel."""
        return {
            "port": self.port,
            "wire_rate_hz": self._client_rate,
            "wire_rate_origin": self._rate_origin,
            "wire_format": (
                f"s16le {self._client_rate} Hz mono"
                + ("" if self._rate_origin == "declared" else " (rate assumed)")
            ),
            "internal_format": f"float32 {self.sample_rate} Hz mono",
            "converting": self.is_converting,
            "reserved_ports": list(self.reserved_ports),
            "reserved_failed": dict(
                self.reservation.failed if self.reservation else {}
            ),
            "clients": [
                {
                    "address": c.address,
                    "seconds_received": round(c.seconds_received, 2),
                    "config": c.config,
                    "amplification": c.amplification,
                }
                for c in self.clients
            ],
        }


def _rate_from_config(config: Any) -> Optional[int]:
    """Extract a sender-declared sample rate, if it sent one.

    The current sender does not, so this normally returns None and the rate
    stays an assumption.  It exists so that a sender which *does* declare a
    rate is believed rather than overridden, and so a fallback to a lower rate
    cannot silently corrupt every duration, spectrum and fingerprint.
    """
    if not isinstance(config, dict):
        return None
    for key in RATE_CONFIG_KEYS:
        value = config.get(key)
        if value is None:
            continue
        try:
            rate = int(value)
        except (TypeError, ValueError):
            continue
        # Plausible audio rate, so a stray value cannot break the resampler.
        if 4000 <= rate <= 192_000:
            return rate
    return None
