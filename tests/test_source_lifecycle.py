"""Source lifecycle and playback.

The double-``start()`` bug found during live testing is pinned here: the CLI
starts a source so it can accept a client before the pipeline exists, and the
pipeline then calls ``start()`` again.  A source that re-binds its ports on the
second call fails against itself.
"""

from __future__ import annotations

import socket

import numpy as np
import pytest

from app.audio.network import NetworkSource, PortReservation
from app.audio.playback import AudioPlayer, PlaybackError, PlaybackState
from app.config import AppConfig


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ----------------------------------------------------------------------
# Idempotent start
# ----------------------------------------------------------------------
def test_network_start_is_idempotent(config):
    """Starting twice must not re-bind ports the object already holds."""
    port = _free_port()
    reserved = (_free_port(),)
    source = NetworkSource(
        config, port=port, host="127.0.0.1", reserve_ports=True, reserved=reserved
    )
    try:
        source.start()
        held_after_first = source.reserved_ports
        assert held_after_first == reserved

        # The pipeline calls start() again.  This used to raise
        # "Address already in use" against the ports held by this very object.
        source.start()
        source.start()

        assert source.reserved_ports == reserved, "reservation was lost"
        assert source._listener is not None
    finally:
        source.close()


def test_network_close_is_idempotent(config):
    port = _free_port()
    source = NetworkSource(
        config, port=port, host="127.0.0.1", reserve_ports=False
    )
    source.start()
    source.close()
    source.close()  # must not raise
    assert source._listener is None


def test_ports_are_released_on_close(config):
    """A finished run must not leave the ports held."""
    port = _free_port()
    reserved = (_free_port(), _free_port())
    source = NetworkSource(
        config, port=port, host="127.0.0.1", reserve_ports=True, reserved=reserved
    )
    source.start()
    source.close()

    # The ports can now be bound by something else.
    again = PortReservation(reserved)
    held = again.acquire()
    assert set(held) == set(reserved)
    again.release()

    # And the stream port is free again.
    probe = socket.socket()
    try:
        probe.bind(("127.0.0.1", port))
    finally:
        probe.close()


def test_device_start_is_idempotent():
    """The device source must also tolerate a second start()."""
    from app.audio.capture import DeviceCapture

    config = AppConfig().validate()
    try:
        source = DeviceCapture(config)
    except Exception as exc:
        pytest.skip(f"no input device available: {exc}")
    try:
        source.start()
        stream = source._stream
        assert stream is not None
        source.start()
        assert source._stream is stream, "the stream was reopened"
    finally:
        source.close()


# ----------------------------------------------------------------------
# Playback
# ----------------------------------------------------------------------
def test_playback_requires_loaded_audio():
    player = AudioPlayer()
    try:
        with pytest.raises(PlaybackError):
            player.play()
    finally:
        player.close()


def test_position_and_duration():
    player = AudioPlayer(sample_rate=48000)
    try:
        player.load(np.zeros(48000, dtype=np.float32))
        assert player.frames_loaded == 48000
        assert player.duration_seconds == pytest.approx(1.0)
        player.seek(0.5)
        assert player.position_seconds == pytest.approx(0.5)
    finally:
        player.close()


def test_seek_is_clamped():
    player = AudioPlayer(sample_rate=48000)
    try:
        player.load(np.zeros(48000, dtype=np.float32))
        player.seek(-5.0)
        assert player.position_seconds == 0.0
        player.seek(999.0)
        assert player.position_seconds == pytest.approx(1.0)
    finally:
        player.close()


def test_seek_relative():
    player = AudioPlayer(sample_rate=48000)
    try:
        player.load(np.zeros(48000, dtype=np.float32))
        player.seek(0.25)
        player.seek_relative(0.25)
        assert player.position_seconds == pytest.approx(0.5)
    finally:
        player.close()


def test_volume_is_validated():
    player = AudioPlayer()
    try:
        player.set_volume(0.5)
        assert player.volume == 0.5
        with pytest.raises(PlaybackError):
            player.set_volume(1.5)
        with pytest.raises(PlaybackError):
            player.set_volume(-0.1)
    finally:
        player.close()


def test_load_rejects_a_mismatched_sample_rate():
    """Playing 44.1 kHz audio at 48 kHz would be wrong, so it is refused."""
    player = AudioPlayer(sample_rate=48000)
    try:
        with pytest.raises(PlaybackError):
            player.load(np.zeros(1000, dtype=np.float32), 44100)
    finally:
        player.close()


def test_load_file_converts_to_the_internal_rate(tmp_path, tone):
    from app.audio.wavio import write_wav_atomic

    path = str(tmp_path / "a.wav")
    write_wav_atomic(path, tone(440.0, 0.5, rate=44100), 44100)

    player = AudioPlayer(sample_rate=48000)
    try:
        player.load_file(path)
        assert player.frames_loaded == pytest.approx(0.5 * 48000, rel=0.01)
    finally:
        player.close()


def test_loop_flag():
    player = AudioPlayer()
    try:
        assert player.loop is False
        player.set_loop(True)
        assert player.loop is True
    finally:
        player.close()


def test_stop_rewinds():
    player = AudioPlayer(sample_rate=48000)
    try:
        player.load(np.zeros(48000, dtype=np.float32))
        player.seek(0.75)
        player.stop()
        assert player.position_seconds == 0.0
        assert player.state is PlaybackState.STOPPED
    finally:
        player.close()


def test_pause_without_playing_is_a_no_op():
    player = AudioPlayer(sample_rate=48000)
    try:
        player.load(np.zeros(48000, dtype=np.float32))
        player.pause()
        assert player.state is PlaybackState.STOPPED
    finally:
        player.close()


def test_unload_clears_audio():
    player = AudioPlayer(sample_rate=48000)
    try:
        player.load(np.zeros(48000, dtype=np.float32))
        player.unload()
        assert player.frames_loaded == 0
        assert player.is_loaded is False
    finally:
        player.close()
