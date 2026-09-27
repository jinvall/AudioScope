"""Tests for the live capture monitor.

Two things are being promised by this feature, and they are tested separately
because they fail differently:

* **Truth.** The panel must say what the stream is doing, and the hardest
  cases are the ones where a plausible-looking lie is available: a document
  nobody is updating any more, a socket that is open while nothing flows, and
  a sample rate that was assumed rather than declared.
* **Not being wrong when it knows nothing.** Every reader path has to survive a
  missing file, a truncated file, a sender that never connected, and a frame
  that never arrived - and must report absence rather than invent a level.
"""

from __future__ import annotations

import json
import os
import time

import pytest

from app.gui.audioview import (
    LIVE_ENVELOPE_FLOOR_DB,
    LIVE_STATUS_STALE_SECONDS,
    level_to_fraction,
    live_envelope,
    summarise_live_status,
)
from app.livestatus import (
    ENVELOPE_COLUMNS,
    LIVE_STATUS_FILENAME,
    STATE_CONNECTED,
    STATE_STALLED,
    STATE_STOPPED,
    STATE_STREAMING,
    STATE_WAITING,
    LiveStatusPublisher,
    read_live_status,
)


# ======================================================================
# A source that behaves, without a network
# ======================================================================
class FakeClient:
    def __init__(self, address="10.0.0.242:42628", bytes_received=0,
                 config=None, amplification=None):
        self.address = address
        self.bytes_received = bytes_received
        self.config = dict(config or {})
        self.dropped_bytes = 0
        self._amplification = amplification

    @property
    def seconds_received(self):
        return self.bytes_received / (44100 * 2)

    @property
    def amplification(self):
        return self._amplification


class FakeSource:
    def __init__(self, clients=None):
        self.clients = list(clients or [])

    def status(self):
        return {
            "port": 8190,
            "wire_rate_hz": 44100,
            "wire_rate_origin": "assumed",
            "wire_format": "s16le 44100 Hz mono (rate assumed)",
            "converting": False,
            "reserved_ports": [8060, 8061],
            "reserved_failed": {},
        }


class FakeResult:
    def __init__(self, rms_db=-24.0, peak_db=-8.0):
        self.rms_db = rms_db
        self.features = type("F", (), {"peak_db": peak_db})()


@pytest.fixture
def publisher(tmp_path):
    return LiveStatusPublisher(
        str(tmp_path), sample_rate=48000, source=FakeSource(),
        publish_interval=0.0,
    )


# ======================================================================
# The publisher
# ======================================================================
class TestPublisher:
    def test_writes_a_readable_document(self, publisher, tmp_path):
        assert publisher.publish(force=True) is True
        document = read_live_status(
            os.path.join(str(tmp_path), LIVE_STATUS_FILENAME))
        assert document is not None
        assert document["version"] >= 1
        assert document["state"] == STATE_WAITING
        assert document["internal_sample_rate"] == 48000

    def test_publish_is_atomic(self, publisher, tmp_path):
        publisher.publish(force=True)
        # No partial file is left behind for a reader to trip over.
        assert not os.path.exists(publisher.path + ".part")

    def test_waiting_with_no_client(self, publisher):
        publisher.publish(force=True)
        document = read_live_status(publisher.path)
        assert document["state"] == STATE_WAITING
        assert document["clients"] == []
        assert document["level_dbfs"] is None

    def test_streaming_once_bytes_arrive(self, publisher):
        publisher.attach_source(FakeSource([FakeClient(bytes_received=441000)]))
        publisher.on_frame(FakeResult())
        publisher.publish(force=True)
        document = read_live_status(publisher.path)
        assert document["state"] == STATE_STREAMING
        assert document["bytes_received"] == 441000
        assert document["level_dbfs"] == -24.0
        assert document["peak_dbfs"] == -8.0

    def test_connected_but_silent_is_its_own_state(self, publisher):
        publisher.attach_source(FakeSource([FakeClient(bytes_received=0)]))
        publisher.publish(force=True)
        assert read_live_status(publisher.path)["state"] == STATE_CONNECTED

    def test_a_sender_that_stops_is_stalled_not_connected(self, publisher):
        client = FakeClient(bytes_received=441000)
        publisher.attach_source(FakeSource([client]))
        publisher.on_frame(FakeResult())
        publisher.publish(force=True)
        assert read_live_status(publisher.path)["state"] == STATE_STREAMING

        # Bytes stop advancing while the socket stays open: the stall case,
        # which is the one worth being able to see.
        with publisher._lock:
            publisher._last_advance = time.time() - 10.0
        publisher.publish(force=True)
        document = read_live_status(publisher.path)
        assert document["state"] == STATE_STALLED
        # The client is still listed: the connection did not fail, the audio
        # did.
        assert len(document["clients"]) == 1

    def test_stopping_is_published(self, publisher):
        publisher.attach_source(FakeSource([FakeClient(bytes_received=441000)]))
        publisher.mark_stopped()
        publisher.publish(force=True)
        assert read_live_status(publisher.path)["state"] == STATE_STOPPED

    def test_envelope_is_bounded(self, publisher):
        for _ in range(ENVELOPE_COLUMNS * 3):
            publisher.on_frame(FakeResult())
        publisher.publish(force=True)
        document = read_live_status(publisher.path)
        assert len(document["envelope"]) == ENVELOPE_COLUMNS

    def test_envelope_keeps_the_newest_values(self, publisher):
        for index in range(ENVELOPE_COLUMNS + 10):
            publisher.on_frame(FakeResult(rms_db=float(-index % 50)))
        publisher.publish(force=True)
        envelope = read_live_status(publisher.path)["envelope"]
        # The oldest kept value is the 11th appended, not the first.
        assert envelope[0] == (-10) % 50
        assert envelope[-1] == (-(ENVELOPE_COLUMNS + 9)) % 50

    def test_a_malformed_frame_is_ignored(self, publisher):
        class Broken:
            rms_db = "not a number"
            features = FakeResult().features

        publisher.on_frame(Broken())
        publisher.publish(force=True)
        assert read_live_status(publisher.path)["envelope"] == []

    def test_a_nan_level_is_ignored(self, publisher):
        publisher.on_frame(FakeResult(rms_db=float("nan")))
        publisher.publish(force=True)
        assert read_live_status(publisher.path)["level_dbfs"] is None

    def test_publish_errors_do_not_raise(self, tmp_path):
        # A plain file where the directory should be: publishing is a
        # convenience for the UI and must never take capture down.
        blocker = tmp_path / "blocker"
        blocker.write_text("not a directory")
        publisher = LiveStatusPublisher(
            os.path.join(str(blocker), "recordings"), source=FakeSource())
        assert publisher.publish(force=True) is False
        assert publisher.publish_errors >= 1
        # And the snapshot itself still works, for a caller that wants it.
        assert publisher.snapshot()["state"] == STATE_WAITING

    def test_clear_removes_the_document(self, publisher):
        publisher.publish(force=True)
        assert os.path.exists(publisher.path)
        publisher.clear()
        assert not os.path.exists(publisher.path)

    def test_unreadable_source_does_not_raise(self, tmp_path):
        class Hostile:
            clients = None

            def status(self):
                raise RuntimeError("no")

            def __getattr__(self, name):
                raise RuntimeError("no")

        publisher = LiveStatusPublisher(str(tmp_path), source=Hostile())
        publisher.publish(force=True)
        assert read_live_status(publisher.path) is not None


class TestReader:
    def test_missing_file_reads_as_absent(self, tmp_path):
        assert read_live_status(str(tmp_path / "nothing.json")) is None

    def test_truncated_file_reads_as_absent(self, tmp_path):
        # A reader that raised here would take the window's timer down.
        path = tmp_path / LIVE_STATUS_FILENAME
        path.write_text('{"state": "stream')
        assert read_live_status(str(path)) is None

    def test_a_json_list_reads_as_absent(self, tmp_path):
        path = tmp_path / LIVE_STATUS_FILENAME
        path.write_text("[1, 2, 3]")
        assert read_live_status(str(path)) is None

    def test_an_empty_file_reads_as_absent(self, tmp_path):
        path = tmp_path / LIVE_STATUS_FILENAME
        path.write_text("")
        assert read_live_status(str(path)) is None


# ======================================================================
# Reading it into something displayable
# ======================================================================
class TestSummary:
    def test_absent_reports_that_capture_is_not_running(self):
        summary = summarise_live_status(None)
        assert summary["available"] is False
        assert summary["state"] == "unavailable"
        assert "Is capture running" in summary["detail"]
        # No invented measurements.
        assert summary["level_dbfs"] is None
        assert summary["envelope"] == []

    def test_every_key_is_present_whatever_the_document(self):
        # The panel has to be able to draw "nothing" as well as "something".
        for document in (None, {}, {"state": "streaming"},
                         {"clients": "not a list"}, {"envelope": "nonsense"}):
            summary = summarise_live_status(document)
            for key in ("available", "state", "state_text", "detail",
                        "address", "level_dbfs", "peak_dbfs", "envelope",
                        "seconds_received", "bytes_received", "wire_rate_hz",
                        "wire_rate_origin", "port", "age_seconds", "stale",
                        "config_keys", "amplification", "dropped_bytes"):
                assert key in summary, f"{key} missing for {document}"

    def test_waiting_names_the_dependency(self):
        summary = summarise_live_status(
            {"state": "waiting_for_client", "port": 8190})
        assert "Waiting" in summary["state_text"]
        # The phone only streams to a receiver that is already running, and
        # the operator has to be told that or they will blame the phone.
        assert "receiver that is already running" in summary["detail"]
        assert "8190" in summary["detail"]

    def test_streaming_with_an_assumed_rate_says_so(self):
        summary = summarise_live_status({
            "state": "streaming", "wire_rate_hz": 44100,
            "wire_rate_origin": "assumed", "port": 8190,
        })
        assert summary["state_text"] == "Receiving audio"
        assert "assumed" in summary["detail"]

    def test_streaming_with_a_declared_rate_does_not_warn(self):
        summary = summarise_live_status({
            "state": "streaming", "wire_rate_hz": 48000,
            "wire_rate_origin": "declared",
        })
        assert "assumed" not in summary["detail"]

    def test_stalled_is_distinguishable(self):
        summary = summarise_live_status({
            "state": "stalled", "clients": [{"address": "10.0.0.242:1"}],
        })
        assert summary["state_text"] == "Stalled"
        # The socket is still there, which is exactly why it is worth showing
        # as its own state rather than as "connected".
        assert summary["address"] == "10.0.0.242:1"

    def test_a_document_nobody_is_updating_is_not_trusted(self):
        summary = summarise_live_status({
            "state": "streaming",
            "published_at": time.time() - (LIVE_STATUS_STALE_SECONDS * 4),
        })
        assert summary["stale"] is True
        assert "No update" in summary["state_text"]
        # A stale document must not claim audio is arriving.
        assert summary["state"] == "streaming"   # the state is preserved...
        assert summary["stale"] is True          # ... but flagged as untrusted

    def test_a_fresh_document_is_trusted(self):
        summary = summarise_live_status(
            {"state": "streaming", "published_at": time.time()})
        assert summary["stale"] is False

    def test_counters_are_summed_across_clients(self):
        summary = summarise_live_status({
            "state": "streaming",
            "bytes_received": 300,
            "clients": [
                {"address": "a", "seconds_received": 1.5,
                 "dropped_bytes": 2, "config": {}},
                {"address": "b", "seconds_received": 2.5,
                 "dropped_bytes": 3, "config": {}},
            ],
        })
        assert summary["seconds_received"] == 4.0
        assert summary["dropped_bytes"] == 5

    def test_the_senders_config_is_reported_not_applied(self):
        summary = summarise_live_status({
            "state": "streaming",
            "clients": [{
                "address": "a", "config": {"amplification": 2.0},
                "amplification": 2.0,
            }],
        })
        assert summary["config_keys"] == ["amplification"]
        assert summary["amplification"] == 2.0

    def test_envelope_is_passed_through_cleaned(self):
        summary = summarise_live_status({
            "state": "streaming", "envelope": [-20.0, "x", None, -30.0],
        })
        assert summary["envelope"] == [-20.0, -30.0]


class TestLevelMapping:
    def test_silence_is_zero(self):
        assert level_to_fraction(None) == 0.0
        assert level_to_fraction(-200.0) == 0.0
        assert level_to_fraction(LIVE_ENVELOPE_FLOOR_DB) == 0.0

    def test_full_scale_is_one(self):
        assert level_to_fraction(0.0) == 1.0

    def test_clamped_above_zero(self):
        assert level_to_fraction(6.0) == 1.0

    def test_monotonic_and_in_range(self):
        previous = -1.0
        for value in (-90, -70, -50, -30, -10, 0):
            fraction = level_to_fraction(value)
            assert 0.0 <= fraction <= 1.0
            assert fraction >= previous
            previous = fraction

    def test_quiet_audio_is_visible(self):
        # The point of a dB mapping rather than a linear one: -50 dBFS is
        # tiny in amplitude but must not be invisible.
        assert level_to_fraction(-50.0) > 0.4

    def test_nan_is_zero(self):
        assert level_to_fraction(float("nan")) == 0.0


class TestEnvelopeExtraction:
    def test_none_and_garbage(self):
        assert live_envelope(None) == []
        assert live_envelope({}) == []
        assert live_envelope({"envelope": "nope"}) == []

    def test_nan_and_infinity_are_dropped(self):
        values = live_envelope(
            {"envelope": [-20.0, float("nan"), float("inf"), -30.0]})
        assert values == [-20.0, -30.0]

    def test_numeric_strings_are_coerced(self):
        assert live_envelope({"envelope": ["-20.0", -30.0]}) == [-20.0, -30.0]


# ======================================================================
# Round trip
# ======================================================================
def test_publish_then_summarise(tmp_path):
    """The whole path: frames and a client in, a displayable summary out."""
    source = FakeSource([FakeClient(bytes_received=882000)])
    publisher = LiveStatusPublisher(str(tmp_path), source=source,
                                   publish_interval=0.0)
    for _ in range(20):
        publisher.on_frame(FakeResult(rms_db=-30.0, peak_db=-12.0))
    publisher.publish(force=True)

    summary = summarise_live_status(read_live_status(publisher.path))
    assert summary["available"] is True
    assert summary["state"] == STATE_STREAMING
    assert summary["state_text"] == "Receiving audio"
    assert summary["address"] == "10.0.0.242:42628"
    assert summary["port"] == 8190
    assert summary["wire_rate_origin"] == "assumed"
    assert summary["level_dbfs"] == -30.0
    assert len(summary["envelope"]) == 20
    assert level_to_fraction(summary["level_dbfs"]) > 0.0


# ======================================================================
# The widget
# ======================================================================
@pytest.fixture(scope="module")
def qapp():
    """One QApplication for the module; Qt allows only one."""
    from PyQt5.QtWidgets import QApplication

    existing = QApplication.instance()
    return existing or QApplication([])


@pytest.fixture(scope="module")
def theme():
    from app.gui.theme import load_theme

    return load_theme("dark")


def test_the_widget_paints_every_state(qapp, theme):
    """A paint crash would blank the tab, so every state is painted."""
    from app.gui.widgets import LiveMonitorWidget

    widget = LiveMonitorWidget(theme)
    for document in (
        None,
        {"state": "waiting_for_client", "port": 8190},
        {"state": "connected", "clients": [{"address": "a"}]},
        {"state": "streaming", "level_dbfs": -20.0, "peak_dbfs": -5.0,
         "envelope": [-20.0] * 40, "wire_rate_origin": "assumed",
         "wire_rate_hz": 44100},
        {"state": "stalled", "clients": [{"address": "a"}]},
        {"state": "stopped"},
    ):
        widget.set_summary(summarise_live_status(document))
        widget.set_footer("status file: /tmp/x")
        assert not widget.grab().isNull()


def test_the_widget_survives_an_empty_summary(qapp, theme):
    from app.gui.widgets import LiveMonitorWidget

    widget = LiveMonitorWidget(theme)
    widget.set_summary({})
    widget.set_footer("")
    assert not widget.grab().isNull()
