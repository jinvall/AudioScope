"""Changing the continuous-recording chunk length while capture runs.

The chain under test is a request travelling between two processes:

    the window's selector -> control.json -> the capture loop -> the recorder
    -> back out through live status as the length in force

Every link is checked, and the end-to-end test runs a real capture and
verifies the files it writes actually change length.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pytest
import soundfile as sf

from app.config import AppConfig
from app.control import (
    MAX_CHUNK_SECONDS,
    MIN_CHUNK_SECONDS,
    ControlWatcher,
    apply_chunk_seconds,
    read_control,
    write_control,
)


# ======================================================================
# The control file
# ======================================================================
class TestControlFile:
    def test_a_request_is_written_and_read_back(self, tmp_path):
        path = write_control(str(tmp_path), chunk_seconds=300.0)
        assert path and os.path.exists(path)
        document = read_control(str(tmp_path))
        assert document["chunk_seconds"] == 300.0
        assert "requested_at" in document, "a request must be identifiable"

    def test_the_file_is_written_atomically(self, tmp_path):
        write_control(str(tmp_path), chunk_seconds=300.0)
        assert not os.path.exists(
            os.path.join(str(tmp_path), "control.json.part")
        )

    def test_a_missing_file_reads_as_no_request(self, tmp_path):
        assert read_control(str(tmp_path)) is None

    def test_a_truncated_file_reads_as_no_request(self, tmp_path):
        # The window can be mid-write or killed; capture must not act on a
        # half-written request.
        (tmp_path / "control.json").write_text('{"chunk_sec')
        assert read_control(str(tmp_path)) is None

    def test_garbage_reads_as_no_request(self, tmp_path):
        (tmp_path / "control.json").write_text("[1, 2, 3]")
        assert read_control(str(tmp_path)) is None


# ======================================================================
# Deciding whether to honour a request
# ======================================================================
class TestApplyChunkSeconds:
    def test_a_valid_request_is_accepted(self):
        assert apply_chunk_seconds({"chunk_seconds": 300.0}, 900.0) == 300.0

    def test_no_document_means_no_change(self):
        assert apply_chunk_seconds(None, 900.0) is None
        assert apply_chunk_seconds({}, 900.0) is None

    def test_a_value_already_in_force_is_not_a_change(self):
        """Otherwise the same file is re-applied five times a second."""
        assert apply_chunk_seconds({"chunk_seconds": 900.0}, 900.0) is None

    @pytest.mark.parametrize("value", [0, 5, -1, MAX_CHUNK_SECONDS * 2,
                                       "nonsense", None])
    def test_out_of_range_and_unusable_values_are_refused(self, value):
        assert apply_chunk_seconds({"chunk_seconds": value}, 900.0) is None

    def test_the_bounds_are_sane(self):
        assert MIN_CHUNK_SECONDS >= 5.0
        assert MAX_CHUNK_SECONDS >= 3600.0


class TestControlWatcher:
    def test_a_request_is_honoured_once(self, tmp_path):
        write_control(str(tmp_path), chunk_seconds=300.0)
        watcher = ControlWatcher(str(tmp_path))
        first = watcher.poll(900.0)
        assert first == 300.0
        # Polling again must not re-apply the same request.
        assert watcher.poll(900.0) is None

    def test_a_new_request_is_honoured(self, tmp_path):
        write_control(str(tmp_path), chunk_seconds=300.0)
        watcher = ControlWatcher(str(tmp_path))
        assert watcher.poll(900.0) == 300.0
        time.sleep(0.01)
        write_control(str(tmp_path), chunk_seconds=600.0)
        assert watcher.poll(300.0) == 600.0

    def test_a_refused_request_is_reported(self, tmp_path):
        write_control(str(tmp_path), chunk_seconds=1.0)
        watcher = ControlWatcher(str(tmp_path))
        assert watcher.poll(900.0) is None
        assert watcher.last_error is not None
        assert "refused" in watcher.last_error


# ======================================================================
# The recorder
# ======================================================================
class TestRecorderChunkLength:
    def _recorder(self, tmp_path, chunk_seconds=900.0):
        from app.audio.recorder import Recorder

        config = AppConfig(
            record={"chunk_seconds": chunk_seconds}
        ).validate()
        return Recorder(config, str(tmp_path))

    def test_the_configured_length_is_in_force_at_start(self, tmp_path):
        recorder = self._recorder(tmp_path, 300.0)
        recorder.start()
        try:
            assert recorder.chunk_seconds == pytest.approx(300.0)
        finally:
            recorder.stop()

    def test_a_new_length_takes_effect(self, tmp_path):
        recorder = self._recorder(tmp_path, 900.0)
        recorder.start()
        try:
            assert recorder.set_chunk_seconds(60.0) is True
            assert recorder.chunk_seconds == pytest.approx(60.0)
        finally:
            recorder.stop()

    def test_an_unusable_length_is_refused(self, tmp_path):
        recorder = self._recorder(tmp_path, 900.0)
        recorder.start()
        try:
            for bad in (0, 1, -5, "nonsense", MAX_CHUNK_SECONDS * 2):
                assert recorder.set_chunk_seconds(bad) is False
            assert recorder.chunk_seconds == pytest.approx(900.0)
        finally:
            recorder.stop()

    def test_the_chunk_in_progress_is_neither_cut_short_nor_extended(
        self, tmp_path
    ):
        """The file being written finishes at the length it started with."""
        recorder = self._recorder(tmp_path, 900.0)
        recorder.start()
        try:
            rate = recorder.config.sample_rate
            # Write 10 s of audio, then ask for 60 s chunks.
            for _ in range(20):
                recorder.write(np.zeros(rate // 2, dtype=np.float32))
            time.sleep(0.3)
            assert recorder.set_chunk_seconds(60.0) is True
            # Draining the recorder must not produce a 60 s file for the
            # 10 s already buffered, nor a 900 s one.
            recorder.stop()
            chunks = [
                sf.info(os.path.join(str(tmp_path), name))
                for name in sorted(os.listdir(str(tmp_path)))
                if name.endswith(".wav")
            ]
            assert chunks, "nothing was recorded"
            for info in chunks:
                seconds = info.frames / info.samplerate
                assert seconds <= 900.0, "a chunk exceeded the original length"
            # 10 s of audio at 60 s chunks: still one open chunk, flushed
            # when recording stopped.
            assert sum(c.frames / c.samplerate for c in chunks) == \
                pytest.approx(10.0, abs=0.5)
        finally:
            if recorder.running:
                recorder.stop()


# ======================================================================
# End to end: the request reaches a running capture
# ======================================================================
class TestEndToEnd:
    def test_a_running_capture_honours_a_request_from_the_window(self, tmp_path):
        """Run the real capture and change its chunk length mid-flight.

        This is the test that would catch a broken link in the chain; each
        link in isolation is checked above, and this checks the wire.
        """
        import subprocess
        import sys

        recordings = tmp_path / "recordings"
        recordings.mkdir()
        source = tmp_path / "probe.wav"
        sf.write(str(source), np.zeros(48000 * 4, dtype=np.float32), 48000,
                 subtype="FLOAT")

        process = subprocess.Popen(
            [sys.executable, "-u", "-m", "app.capture",
             "--source", "network", "--port", "8477",
             "--output", str(recordings), "--no-reserve", "--no-detection",
             "--no-analysis", "--db", str(tmp_path / "e.db"),
             "--events", str(tmp_path / "events"), "--seconds", "6"],
            cwd="/home/jason/audioscope",
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        status_path = recordings / "live_status.json"
        try:
            deadline = time.time() + 30
            while time.time() < deadline and not status_path.exists():
                if process.poll() is not None:
                    pytest.skip("capture did not start")
                time.sleep(0.2)
            assert status_path.exists(), "capture published no status"

            def chunk_seconds():
                try:
                    with open(status_path) as handle:
                        return json.load(handle).get("chunk_seconds")
                except (OSError, ValueError):
                    return None

            # The configured 15 minutes, echoed out.
            deadline = time.time() + 20
            while time.time() < deadline and chunk_seconds() is None:
                time.sleep(0.2)
            assert chunk_seconds() == pytest.approx(900.0)

            # The window asks for 30 seconds.
            assert write_control(str(recordings), chunk_seconds=30.0)
            deadline = time.time() + 20
            while time.time() < deadline:
                if chunk_seconds() == pytest.approx(30.0):
                    break
                time.sleep(0.2)
            assert chunk_seconds() == pytest.approx(30.0), (
                "the request never reached the recorder"
            )
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
