"""Network source: the 8090 PCM stream and the 8060-8064 reservation.

The wire format is the one the AMP receiver uses
(``/home/jason/amp/server/audio_receiver.py``): an optional newline-terminated
JSON config line, then **raw unframed** signed 16-bit little-endian PCM at
44 100 Hz mono.  These tests speak that format over a real socket, so the
protocol is verified rather than assumed.
"""

from __future__ import annotations

import json
import socket
import threading
import time

import numpy as np
import pytest

from app.audio.network import (
    BYTES_PER_SECOND,
    RESERVED_PORTS,
    STREAM_PORT,
    WIRE_SAMPLE_RATE,
    NetworkError,
    NetworkSource,
    PortReservation,
)
from app.config import AppConfig


def _free_port() -> int:
    """A port we can actually bind.

    Asking the OS for port 0 and taking whatever it hands back is not enough:
    a previous test's server may still hold it, and the resulting "address
    already in use" surfaces far from the cause.  Verify the port is bindable
    before returning it.
    """
    for _ in range(20):
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        check = socket.socket()
        try:
            check.bind(("127.0.0.1", port))
        except OSError:
            continue
        finally:
            check.close()
        return port
    raise RuntimeError("could not find a bindable port")


def _s16le(samples: np.ndarray) -> bytes:
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    return np.round(clipped * 32767.0).astype("<i2").tobytes()


class RawClient:
    """Sends the real protocol: optional JSON line, then raw PCM."""

    def __init__(self, port: int) -> None:
        self.sock = socket.create_connection(("127.0.0.1", port), timeout=5)
        self.sock.settimeout(5)

    def send_config(self, config: dict) -> None:
        line = json.dumps(config, separators=(",", ":"))
        self.sock.sendall(line.encode("utf-8") + b"\n")

    def send_pcm(self, samples: np.ndarray) -> None:
        self.sock.sendall(_s16le(samples))

    def close(self) -> None:
        try:
            self.sock.close()
        except OSError:
            pass


class StreamingClient(RawClient):
    """A client that keeps sending, like the real device.

    ``read()`` on a live source blocks until audio arrives, so a test that
    waits for a fixed frame count must keep the sender running.  This mirrors
    how the device behaves rather than sending one burst and going silent.
    """

    def __init__(self, port: int, samples: np.ndarray, rate: int = WIRE_SAMPLE_RATE,
                 config: dict | None = None, autostart: bool = True):
        super().__init__(port)
        # The config line must reach the server before any PCM, exactly as the
        # device sends it.  Doing it after the streaming thread starts is a
        # race: the source would see binary PCM first, decide there is no
        # config line, and treat the config line itself as audio.
        if config is not None:
            self.send_config(config)
        self._samples = samples
        self._rate = rate
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        if autostart:
            self.start_streaming()

    def start_streaming(self) -> None:
        """Begin sending audio.  Separated from construction so a test that
        cares about the config line can let the server observe it first,
        instead of depending on how quickly the reader thread gets scheduled."""
        if not self._thread.is_alive():
            self._thread.start()

    def _run(self) -> None:
        chunk = 4096
        index = 0
        while not self._stop.is_set():
            block = self._samples[index:index + chunk]
            if block.size == 0:
                index = 0
                continue
            try:
                self.send_pcm(block)
            except OSError:
                return
            index += block.size
            time.sleep(block.size / self._rate)

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=2.0)
        self.close()


def _drain(source: NetworkSource, min_frames: int, timeout: float = 5.0):
    """Collect audio until ``min_frames`` samples have arrived.

    Bounded by a deadline that is checked *between* reads.  ``read()`` blocks
    while the stream is idle, so a deadline that is only checked after a read
    returns can never fire; the sender must therefore still be running.
    """
    out = []
    total = 0
    deadline = time.monotonic() + timeout
    while total < min_frames and time.monotonic() < deadline:
        # Poll with a timeout: read() blocks indefinitely by design, so a
        # drain that ran out of audio would otherwise never notice the
        # deadline.
        block = source.read(timeout=0.25)
        if block is not None:
            out.append(block)
            total += block.size
    return np.concatenate(out) if out else np.zeros(0, dtype=np.float32)


@pytest.fixture
def source(config):
    port = _free_port()
    src = NetworkSource(
        config, port=port, host="127.0.0.1", reserve_ports=False
    )
    src.start()
    yield src
    src.close()


# ----------------------------------------------------------------------
# Port reservation
# ----------------------------------------------------------------------
def test_default_ports_are_the_documented_ones():
    assert STREAM_PORT == 8190
    assert RESERVED_PORTS == (8060, 8061, 8062, 8063, 8064)


def test_ports_do_not_collide_with_the_amp_receiver():
    """AMP binds PCM/control/viz inside 8090..8099; we must stay out of it."""
    amp_range = set(range(8090, 8100))
    assert STREAM_PORT not in amp_range
    assert not amp_range.intersection(RESERVED_PORTS)


def test_wire_format_matches_the_amp_receiver():
    assert WIRE_SAMPLE_RATE == 44100
    assert BYTES_PER_SECOND == 44100 * 2


def test_reservation_holds_all_five_ports():
    reservation = PortReservation(RESERVED_PORTS)
    try:
        assert set(reservation.acquire()) == set(RESERVED_PORTS)
        assert not reservation.failed
    finally:
        reservation.release()


def test_reservation_conflicts_are_reported_not_hidden():
    ports = (_free_port(), _free_port())
    first = PortReservation(ports)
    first.acquire()
    try:
        second = PortReservation(ports)
        with pytest.raises(NetworkError) as excinfo:
            second.acquire()
        assert "could not reserve" in str(excinfo.value)
        assert set(second.failed) == set(ports)
        second.release()
    finally:
        first.release()


def test_partial_reservation_reports_only_the_conflict():
    busy, free = _free_port(), _free_port()
    holder = PortReservation((busy,))
    holder.acquire()
    try:
        reservation = PortReservation((busy, free))
        assert reservation.acquire() == (free,)
        assert set(reservation.failed) == {busy}
        reservation.release()
    finally:
        holder.release()


def test_reservation_reports_partial_failure_to_the_caller():
    busy, free = _free_port(), _free_port()
    holder = PortReservation((busy,))
    holder.acquire()
    try:
        messages: list[str] = []
        src = NetworkSource(
            AppConfig(), port=_free_port(), host="127.0.0.1",
            reserve_ports=True, reserved=(busy, free), on_status=messages.append,
        )
        src.start()
        try:
            assert any("reservation incomplete" in m for m in messages)
        finally:
            src.close()
    finally:
        holder.release()


# ----------------------------------------------------------------------
# Lifecycle
# ----------------------------------------------------------------------
def test_start_is_idempotent(config):
    port, reserved = _free_port(), (_free_port(),)
    src = NetworkSource(
        config, port=port, host="127.0.0.1",
        reserve_ports=True, reserved=reserved,
    )
    try:
        src.start()
        assert src.reserved_ports == reserved
        src.start()
        src.start()
        assert src.reserved_ports == reserved
    finally:
        src.close()


def test_close_is_idempotent_and_frees_ports(config):
    port, reserved = _free_port(), (_free_port(), _free_port())
    src = NetworkSource(
        config, port=port, host="127.0.0.1",
        reserve_ports=True, reserved=reserved,
    )
    src.start()
    src.close()
    src.close()
    assert src._listener is None

    again = PortReservation(reserved)
    assert set(again.acquire()) == set(reserved)
    again.release()

    probe = socket.socket()
    try:
        probe.bind(("127.0.0.1", port))
    finally:
        probe.close()


def test_read_blocks_until_stopped_when_no_client(source):
    """With no client, read() waits for audio rather than reporting end.

    A live source has no end of stream until it is stopped, so returning None
    here would make the pipeline tear itself down the moment it started.
    """
    returned = threading.Event()
    result: list = []

    def reader():
        result.append(source.read())
        returned.set()

    thread = threading.Thread(target=reader, daemon=True)
    thread.start()
    # It should still be waiting after a short while.
    assert not returned.wait(timeout=1.0), "read() returned with no client attached"

    source.stop()
    assert returned.wait(timeout=5), "stop() did not release a waiting read()"
    assert result == [None]


# ----------------------------------------------------------------------
# Streaming
# ----------------------------------------------------------------------
def test_raw_pcm_is_received_and_resampled_to_48k(source):
    """The headline path: 44.1 kHz s16le on the wire, 48 kHz float32 inside."""
    assert source.native_sample_rate == 44100
    assert source.is_converting, "44.1 kHz input must be converted to 48 kHz"

    seconds = 1.0
    t = np.arange(int(WIRE_SAMPLE_RATE * seconds)) / WIRE_SAMPLE_RATE
    tone = (0.5 * np.sin(2 * np.pi * 1000 * t)).astype(np.float32)
    client = StreamingClient(source.port, tone)

    try:
        assert source.wait_for_client(timeout=5)
        out = _drain(source, int(0.6 * 48000))
    finally:
        client.stop()

    assert out.size >= int(0.5 * 48000)
    # Amplitude and frequency preserved through s16 quantisation + resampling.
    out_t = np.arange(out.size) / 48000
    amplitude = abs(2 * np.mean(out * np.exp(-2j * np.pi * 1000 * out_t)))
    assert amplitude == pytest.approx(0.5, rel=0.05)


def test_s16_is_scaled_into_range_without_clipping(source):
    """int16 -> float32 scaling must not itself clip.

    Checked with a smooth tone.  A full-scale square wave is deliberately not
    used here: the anti-alias filter overshoots on a discontinuity (Gibbs),
    which is real filter behaviour and is asserted separately below.
    """
    t = np.arange(int(WIRE_SAMPLE_RATE * 1.0)) / WIRE_SAMPLE_RATE
    tone = np.sin(2 * np.pi * 440 * t).astype(np.float32)
    client = StreamingClient(source.port, tone)
    try:
        assert source.wait_for_client(timeout=5)
        out = _drain(source, 20000)
    finally:
        client.stop()

    assert out.size >= 20000
    # +32767/32768 is the largest value the scaling can produce, so a
    # full-scale sine must sit just under 1.0 and must not exceed it.
    assert np.max(np.abs(out)) <= 1.0, "scaling produced out-of-range samples"
    assert np.max(np.abs(out)) > 0.99, "full-scale sine did not reach full scale"


def test_filter_overshoot_on_a_square_wave_is_preserved_not_clipped(source):
    """A resampled square wave can exceed 1.0; the file must keep the peaks.

    Gibbs overshoot near a discontinuity is correct filter behaviour, not a
    bug.  Because evidence is stored as float32 WAV the peaks survive intact
    rather than being silently clipped, which is what makes the recording
    trustworthy.
    """
    values = np.array([32767, -32768, 0, 32767, -32768], dtype=np.int16)
    square = (np.tile(values, 2000).astype(np.float32)) / 32767.0
    client = RawClient(source.port)
    client.send_pcm(square)
    try:
        assert source.wait_for_client(timeout=5)
        out = _drain(source, 100000)
    finally:
        client.close()

    assert out.size > 0
    # Overshoot above the nominal full scale is expected for this waveform.
    assert np.max(np.abs(out)) > 1.0, "expected filter overshoot on a square wave"
    # The recording is float32, so nothing was clipped away.
    assert out.dtype == np.float32


def test_config_line_is_parsed_and_recorded(source):
    """The JSON config line must be consumed, not mistaken for audio."""
    config = {
        "amplification": 2.185,
        "breathing_sensitivity": 48,
        "breathing_cooldown": 5,
        "segment_duration_min": 5,
    }
    t = np.arange(int(WIRE_SAMPLE_RATE * 1.0)) / WIRE_SAMPLE_RATE
    tone = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    # Audio is held back until the server has taken the config line.  Letting a
    # streaming thread run concurrently made this test depend on how quickly
    # the reader thread happened to be scheduled, which is the one thing a unit
    # test should not depend on.
    client = StreamingClient(
        source.port, tone, config=config, autostart=False
    )
    # Captured inside the try, before the client is stopped: stopping closes the
    # socket, the server drops the client, and its config goes with it.
    received: list = []
    addresses: list = []
    live_amplification = None
    server_said: list = []
    source.on_status = server_said.append
    try:
        assert source.wait_for_client(timeout=5)
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline and not source.client_configs:
            time.sleep(0.02)
        received = list(source.client_configs)
        addresses = [c.address for c in source.clients]
        # Read while the client is still attached.  Reading this after the
        # finally block races the server's reader thread noticing the close,
        # and passed or failed depending on which won - that was the flake.
        if source.clients:
            live_amplification = source.clients[0].amplification
        client.start_streaming()
        out = _drain(source, 1000)
    finally:
        client.stop()

    assert received, (
        "the config line was not parsed; "
        f"clients={[(c.address, c.config) for c in source.clients]} "
        f"seen={addresses} server_said={server_said}"
    )
    received = received[0]
    assert received["amplification"] == 2.185
    assert received["breathing_sensitivity"] == 48
    assert received["segment_duration_min"] == 5

    assert live_amplification == 2.185
    # The audio after the config line is still audio, not config text.
    assert out.size > 0


def test_config_and_pcm_coalesced_in_one_packet(source):
    """A config line and the first audio in one TCP segment must both survive."""
    client = RawClient(source.port)
    # Config line and the first audio deliberately coalesced into one write.
    payload = (
        json.dumps({"amplification": 1.5}, separators=(",", ":")).encode()
        + b"\n"
        + _s16le(np.full(44100, 0.2, dtype=np.float32))
    )
    client.sock.sendall(payload)
    # Captured before the socket is closed; see the note in the test above.
    received: list = []
    try:
        assert source.wait_for_client(timeout=5)
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not source.client_configs:
            time.sleep(0.05)
        out = _drain(source, 1000)
        received = list(source.client_configs)
    finally:
        client.close()

    assert received, "the config line was not parsed"
    assert received[0]["amplification"] == 1.5
    # 4410 samples at 44.1 kHz becomes 4800 at 48 kHz.
    assert out.size >= 4000


def test_invalid_json_config_is_reported_and_stream_continues(source):
    t = np.arange(int(WIRE_SAMPLE_RATE * 1.0)) / WIRE_SAMPLE_RATE
    tone = (0.3 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    client = StreamingClient(source.port, tone)
    client.sock.sendall(b"{not valid json\n")
    try:
        assert source.wait_for_client(timeout=5)
        out = _drain(source, 1000)
    finally:
        client.stop()

    assert out.size > 0, "audio after a bad config line was lost"
    # A malformed config line must not be reported as a parsed config.
    assert not source.client_configs


def test_client_disconnect_is_a_clean_end_of_stream(source):
    t = np.arange(int(WIRE_SAMPLE_RATE * 1.0)) / WIRE_SAMPLE_RATE
    tone = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    client = StreamingClient(source.port, tone)
    try:
        assert source.wait_for_client(timeout=5)
        _drain(source, 100)
    finally:
        client.stop()
        client.close()

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and source.client_count:
        time.sleep(0.05)
    assert source.client_count == 0
    # Still serving whatever was buffered; no exception.
    # The rate is an assumption unless the sender declares it, and the status
    # says so rather than presenting a guess as a fact.
    status = source.status()
    assert status["wire_rate_hz"] == WIRE_SAMPLE_RATE
    assert status["wire_rate_origin"] == "assumed"
    assert "assumed" in status["wire_format"]


def test_a_second_client_is_refused_rather_than_misaligned(source):
    """Two senders must not be summed; they cannot be time-aligned.

    Chunks from two TCP connections carry independent jitter, so adding one
    chunk from each per read would combine samples captured at different
    instants.  The server refuses the second connection instead, and says so.
    """
    messages: list[str] = []
    source.on_status = messages.append

    t = np.arange(int(WIRE_SAMPLE_RATE * 1.0)) / WIRE_SAMPLE_RATE
    tone = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    first = StreamingClient(source.port, tone)
    try:
        assert source.wait_for_client(timeout=5)
        assert source.client_count == 1

        # Before probing, make sure the precondition really holds.  If the
        # first sender is not still registered as active there is nothing to
        # refuse, and the failure would otherwise look like a timing problem.
        assert source._active_key is not None, (
            "the first client is no longer active; "
            f"clients={[c.address for c in source.clients]}, messages={messages}"
        )
        assert source._accept_thread is not None and source._accept_thread.is_alive(), (
            f"the accept loop is not running: {messages}"
        )

        # Retry the second connection: the accept loop polls at 0.5 s, and a
        # connection can be queued in the backlog without yet being handled.
        # Assert on the outcome, not on a single attempt's timing.
        reply = b""
        for _ in range(10):
            second = RawClient(source.port)
            second.sock.settimeout(5)
            try:
                reply = second.sock.recv(256)
            except OSError:
                reply = b""
            finally:
                second.close()
            if reply:
                break
            time.sleep(0.2)

        # The server sends ERR just before it logs the refusal, so receiving
        # ERR does not mean the status line has been emitted yet.  Wait for it
        # rather than racing the log call.
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline and not any(
            "refused" in m for m in messages
        ):
            time.sleep(0.02)

        assert any("refused" in m for m in messages), (
            f"messages={messages} client_count={source.client_count} "
            f"active_key={source._active_key} "
            f"accept_alive={source._accept_thread.is_alive()} "
            f"port={source.port} reply={reply!r}"
        )
        assert any("not mixed" in m for m in messages), messages
        assert b"ERR" in reply, (
            f"the refused client was not told why (reply={reply!r})"
        )
        # Still exactly one client, and its audio still flows.
        assert source.client_count == 1
        assert _drain(source, 1000).size >= 1000
    finally:
        first.stop()


def test_a_new_client_is_accepted_after_the_first_disconnects(source):
    """The stream is reusable once the first sender goes away."""
    t = np.arange(int(WIRE_SAMPLE_RATE * 1.0)) / WIRE_SAMPLE_RATE
    tone = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    first = StreamingClient(source.port, tone)
    assert source.wait_for_client(timeout=5)
    first.stop()

    deadline = time.monotonic() + 5
    while time.monotonic() < deadline and source.client_count:
        time.sleep(0.05)
    assert source.client_count == 0

    second = StreamingClient(source.port, tone)
    try:
        assert source.wait_for_client(timeout=5)
        assert source.client_count == 1
        assert _drain(source, 1000).size >= 1000
    finally:
        second.stop()


def test_status_is_serialisable(source):
    import json as _json

    t = np.arange(int(WIRE_SAMPLE_RATE * 1.0)) / WIRE_SAMPLE_RATE
    tone = (0.1 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    client = StreamingClient(source.port, tone, config={"amplification": 1.0})
    try:
        assert source.wait_for_client(timeout=5)
        _drain(source, 100)
        text = _json.dumps(source.status())
        assert "s16le 44100 Hz mono" in text
        assert "float32 48000 Hz mono" in text
    finally:
        client.stop()


def test_a_declared_sample_rate_is_believed_not_overridden():
    """A sender that states its rate must be taken at its word.

    The current sender does not state it and can fall back to a lower rate for
    some sources, so this path exists for a sender that does: reading a 16 kHz
    stream as 44.1 kHz would be silently wrong in pitch, duration and every
    measurement taken from it.
    """
    from app.audio.network import _rate_from_config

    assert _rate_from_config({"sample_rate": 16000}) == 16000
    assert _rate_from_config({"rate": 8000}) == 8000
    assert _rate_from_config({"samplerate": 48000}) == 48000
    # Absent, unusable or implausible values leave the assumption in place.
    assert _rate_from_config({"amplification": 2.1}) is None
    assert _rate_from_config({"rate": "not a number"}) is None
    assert _rate_from_config({"rate": 5}) is None
    assert _rate_from_config(None) is None


def test_wire_rate_origin_defaults_to_assumed(source):
    """With no client yet, the rate is an assumption, and says so."""
    status = source.status()
    assert status["wire_rate_origin"] == "assumed"
    assert status["wire_rate_hz"] == WIRE_SAMPLE_RATE
