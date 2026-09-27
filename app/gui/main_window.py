"""The review workstation window.

Layout follows the brief's conceptual sketch: event list on the left, inspector
on the right, with waveform, optional spectrogram, information, playback and the
review controls on the right-hand side.

Three things this window is careful about:

* **Playback reuses the backend's :class:`~app.audio.playback.AudioPlayer`.**
  Qt Multimedia is not available in this environment, and a second decoding
  pipeline would be worse anyway.  ``AudioPlayer`` already plays on its own
  callback thread, so pressing play returns immediately.
* **The GUI thread never waits.**  Audio decode, envelope and spectrogram are
  all dispatched to worker threads by the controller; the window renders what is
  ready and refreshes when a result lands.
* **Detection is untouched.**  The window reads the database; it never starts,
  reconfigures or blocks the detector.
"""

from __future__ import annotations

import os
import shutil
from typing import Optional

from PyQt5.QtCore import QObject, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QKeySequence
from PyQt5.QtWidgets import (
    QAction,
    QApplication,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSlider,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..audio.playback import AudioPlayer, PlaybackError, PlaybackState
from ..events.database import Decision
from .controller import ReviewController
from .formatting import build_rows, describe_event
from .theme import Theme, build_stylesheet
from .widgets import (
    LIVE_POLL_MS,
    PLAYHEAD_MS,
    EventListWidget,
    FieldPanel,
    LiveMonitorWidget,
    ReviewBar,
    SeparationPanel,
    SpectrogramWidget,
    WaveformWidget,
)


#: How often the window re-reads the capture process's published status.
LIVE_CAPTURE_POLL_MS = 400


class _SeparationBridge(QObject):
    """Carries separation callbacks from the worker thread to the GUI thread.

    The controller invokes its listeners on the separation worker thread.  Qt
    widgets may only be touched from the GUI thread, so the window cannot
    simply connect a lambda to the controller - it needs a queued signal, which
    is what this object provides.
    """

    changed = pyqtSignal(object)      # SeparationResult, or None for state


class MainWindow(QMainWindow):
    """Event review workstation."""

    def __init__(self, controller: ReviewController, theme: Theme,
                 db_path: str = "events.db"):
        super().__init__()
        self.controller = controller
        self.theme = theme
        self.db_path = db_path

        self._stored = None
        self._duration = 0.0
        self._pending_audio = None
        self._pending_envelope = None
        self._pending_spectrogram = None
        self._pending_extract = None
        self._change_marker = controller.change_marker()
        # Separation state.  ``_event_duration`` is the length of the event's
        # own audio, kept separately from ``_duration`` because the transport
        # can be playing a separated file, and a region dragged on the
        # waveform has to be measured against the original, not against
        # whatever is currently loaded.
        self._pending_separation_audio = None
        self._event_duration = 0.0
        self._region: Optional[tuple] = None
        self._playback_source = "original"
        self._separation_bridge = _SeparationBridge()
        # The controller calls this from its worker thread; the bridge re-emits
        # as a queued signal so the slot runs on the GUI thread.  Registering
        # only appends to a list - it does not create the worker, so the model
        # process is still not started here.
        self._separation_listener = self._separation_bridge.changed.emit
        controller.add_separation_listener(self._separation_listener)
        self._separation_bridge.changed.connect(self._on_separation_event)

        self.player = AudioPlayer(sample_rate=controller.sample_rate)
        self.player.set_loop(False)

        self.setWindowTitle("Audio Microscope")
        self.resize(1280, 820)
        self._build()
        self._install_actions()
        self._install_timers()
        self.refresh()

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------
    def _build(self) -> None:
        central = QWidget()
        outer = QVBoxLayout(central)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)
        outer.addWidget(self._build_header())

        splitter = QSplitter(Qt.Horizontal)
        self.event_list = EventListWidget(self.theme)
        self.event_list.event_selected.connect(self.select_event)
        splitter.addWidget(self.event_list)
        splitter.addWidget(self._build_inspector())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        # The event list needs room for four columns without eliding them.
        splitter.setSizes([430, 850])
        outer.addWidget(splitter, 1)

        self.status_label = QLabel("")
        self.status_label.setObjectName("Subtle")
        outer.addWidget(self.status_label)
        self.setCentralWidget(central)

    def _build_header(self) -> QWidget:
        frame = QWidget()
        frame.setObjectName("Header")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(16, 10, 16, 10)
        title = QLabel("Audio Microscope")
        title.setObjectName("Title")
        layout.addWidget(title)
        layout.addSpacing(18)
        layout.addWidget(QLabel("Review:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(
            ["All", "Not reviewed", "Saved", "Confirmed", "Uncertain",
             "Rejected"]
        )
        self.filter_combo.currentIndexChanged.connect(self.refresh)
        layout.addWidget(self.filter_combo)
        layout.addStretch(1)
        self.header_status = QLabel("")
        self.header_status.setObjectName("Subtle")
        layout.addWidget(self.header_status)
        return frame

    def _build_inspector(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        self.title_label = QLabel("No event selected")
        self.title_label.setObjectName("Title")
        layout.addWidget(self.title_label)

        self.subtitle_label = QLabel("")
        self.subtitle_label.setObjectName("Muted")
        layout.addWidget(self.subtitle_label)

        self.waveform = WaveformWidget(self.theme)
        self.waveform.seek_requested.connect(self._on_seek)
        self.waveform.region_changed.connect(self._on_region_changed)
        layout.addWidget(self.waveform)

        layout.addWidget(self._build_transport())

        # Information tabs: the human view by default, the technical view
        # available but not in the way.
        # Panel title -> attribute on EventDescription holding its fields.
        # Spelled out rather than derived: "Event" maps to `headline`, and a
        # lowercased key would look for a non-existent attribute.
        self._panel_sections = (
            ("Event", "Event", "headline"),
            ("Acoustics", "Acoustic summary", "acoustics"),
            ("Source", "Source", "source"),
            ("Review", "Review", "review"),
            ("Details", "Technical details", "technical"),
        )
        self._panels = {
            title: FieldPanel(heading, self.theme)
            for title, heading, _section in self._panel_sections
        }
        self.tabs = QTabWidget()
        for key in ("Event", "Acoustics", "Source", "Review", "Details"):
            self.tabs.addTab(self._tab_with(self._panels[key]), key)
        self.tabs.addTab(self._build_spectrogram_tab(), "Spectrogram")
        self.tabs.addTab(self._build_similar_tab(), "Similar")
        self.tabs.addTab(self._build_separation_tab(), "Separate")
        self.tabs.addTab(self._build_live_tab(), "Live")
        # The two derived views are built only when opened, so neither the
        # spectrogram nor a similarity scan runs for events nobody inspects.
        self.tabs.currentChanged.connect(self._on_tab_changed)
        layout.addWidget(self.tabs, 1)

        self.review_bar = ReviewBar(self.theme)
        self.review_bar.annotation_saved.connect(self._on_annotation)
        layout.addWidget(self.review_bar)
        return page

    @staticmethod
    def _scroll_holder() -> tuple:
        area = QScrollArea()
        area.setWidgetResizable(True)
        holder = QWidget()
        layout = QVBoxLayout(holder)
        layout.setContentsMargins(8, 8, 8, 8)
        # The holder must be owned by the scroll area.  Without this it has no
        # parent, Python collects it, and every panel inside it - and their
        # layouts - is destroyed while still referenced from Python.
        area.setWidget(holder)
        return area, holder, layout

    def _tab_with(self, widget) -> QScrollArea:
        area, holder, layout = self._scroll_holder()
        layout.addWidget(widget)
        layout.addStretch(1)
        return area

    def _build_transport(self) -> QWidget:
        frame = QWidget()
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_play)
        self.play_button.setEnabled(False)
        layout.addWidget(self.play_button)

        replay = QPushButton("Replay")
        replay.clicked.connect(self.replay)
        replay.setEnabled(False)
        self.replay_button = replay
        layout.addWidget(replay)

        layout.addWidget(QLabel("Position"))
        self.position_label = QLabel("0.0 s")
        self.position_label.setObjectName("Muted")
        self.position_label.setFixedWidth(90)
        layout.addWidget(self.position_label)

        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setRange(0, 1000)
        self.seek_slider.setEnabled(False)
        self.seek_slider.sliderReleased.connect(
            lambda: self._on_seek(self.seek_slider.value() / 1000.0)
        )
        layout.addWidget(self.seek_slider, 1)

        self.duration_label = QLabel("")
        self.duration_label.setObjectName("Muted")
        self.duration_label.setFixedWidth(90)
        layout.addWidget(self.duration_label)

        loop = QPushButton("Loop")
        loop.setCheckable(True)
        loop.clicked.connect(lambda checked: self.player.set_loop(checked))
        layout.addWidget(loop)

        # Selecting a region and keeping it.  One toggle arms selection on
        # both the waveform and the spectrogram, because they show the same
        # event on the same timeline and a region drawn on one should be
        # visible on the other.
        self.select_region_button = QPushButton("Select")
        self.select_region_button.setCheckable(True)
        self.select_region_button.setToolTip(
            "Drag on the waveform or the spectrogram to select a region"
        )
        self.select_region_button.toggled.connect(
            self._on_select_region_toggled
        )
        layout.addWidget(self.select_region_button)

        self.extract_button = QPushButton("Extract selection")
        self.extract_button.setEnabled(False)
        self.extract_button.setToolTip(
            "Save the selected region as an event of its own, with its own "
            "audio, measurements and provenance"
        )
        self.extract_button.clicked.connect(self._on_extract_selection)
        layout.addWidget(self.extract_button)

        self.selection_label = QLabel("")
        self.selection_label.setObjectName("Muted")
        layout.addWidget(self.selection_label)

        layout.addWidget(QLabel("Volume"))
        volume = QSlider(Qt.Horizontal)
        volume.setRange(0, 100)
        volume.setValue(80)
        volume.setFixedWidth(110)
        volume.valueChanged.connect(
            lambda v: self.player.set_volume(v / 100.0)
        )
        layout.addWidget(volume)
        return frame

    def _build_spectrogram_tab(self) -> QWidget:
        area, holder, layout = self._scroll_holder()
        self.spectrogram = SpectrogramWidget(self.theme)
        self.spectrogram.region_changed.connect(self._on_region_changed)
        layout.addWidget(self.spectrogram, 1)
        note = QLabel(
            "Generated on demand for this event only, using the same analysis "
            "as the detector. Not computed for every event in the background."
        )
        note.setObjectName("Subtle")
        note.setWordWrap(True)
        layout.addWidget(note)
        self.spectro_note = note
        return area

    def _build_similar_tab(self) -> QWidget:
        area, holder, layout = self._scroll_holder()
        self.similar_label = QLabel("Not computed")
        self.similar_label.setObjectName("Muted")
        self.similar_label.setWordWrap(True)
        layout.addWidget(self.similar_label)
        layout.addStretch(1)
        return area

    def _build_separation_tab(self) -> QWidget:
        """The separation panel: query, attempts, and A/B comparison.

        Built with the rest of the window but kept inert until it is opened,
        so opening the window never starts the model process (see
        :meth:`_on_tab_changed`).
        """
        area, holder, layout = self._scroll_holder()
        self.separation = SeparationPanel(self.theme)
        self.separation.separate_requested.connect(self._on_separate)
        self.separation.compare_requested.connect(self._on_compare_separation)
        self.separation.save_requested.connect(self._on_save_separation)
        # The panel's own "Select a region" and the transport's "Select" are
        # two controls for one state, so both go through the window.  Wiring
        # them to the views separately left the two disagreeing: using one
        # armed the waveform while the other still thought it was off.
        self.separation.selection_toggled.connect(
            self._on_select_region_toggled
        )
        layout.addWidget(self.separation, 1)

        note = QLabel(
            "A separation costs several seconds of CPU per second of audio, so "
            "it runs in the background while capture and review continue. Each "
            "attempt is kept beside the event; repeating this event with a "
            "different query adds another attempt rather than replacing one."
        )
        note.setObjectName("Subtle")
        note.setWordWrap(True)
        layout.addWidget(note)
        return area

    def _build_live_tab(self) -> QWidget:
        """Connection verification and a live level trace for capture.

        Not wrapped in a scroll area, unlike the information tabs: this is a
        fixed monitor rather than a long document, and a scrollbar here would
        only hide the level trace below the fold.
        """
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        self.live_monitor = LiveMonitorWidget(self.theme)
        layout.addWidget(self.live_monitor, 1)
        return page

    def _refresh_live(self) -> None:
        """Poll the published capture status.  Cheap, and never blocks.

        A short file read per tick: the alternative - a socket between the two
        processes - would make the window a participant in capture, and a
        window that has to be open for capture to work is a window that can
        break capture.
        """
        from .audioview import summarise_live_status

        try:
            document = self.controller.read_live_status()
        except Exception as exc:  # pragma: no cover - defensive
            document = None
            detail = f"Could not read capture status: {exc}"
        else:
            detail = ""
        summary = summarise_live_status(document)
        self.live_monitor.set_summary(summary)

        # The footer, drawn inside the panel: the trace then owns all the
        # space that is left, at any window size.
        bits = [f"status file: {self.controller.capture_dir()}"]
        if document is None:
            bits.append("capture is not publishing status")
        elif summary.get("config_keys"):
            bits.append("sender config keys: "
                        + ", ".join(summary["config_keys"]))
        if summary.get("amplification") is not None:
            bits.append(
                f"sender asked for {summary['amplification']:g}x amplification; "
                f"recorded, not applied - apply it at playback"
            )
        if summary.get("dropped_bytes"):
            bits.append(f"sender dropped {summary['dropped_bytes']} byte(s)")
        if detail:
            bits.insert(0, detail)
        bits.append(
            "read from the capture process's own published status"
        )
        self.live_monitor.set_footer("   ·   ".join(bits))

    # ------------------------------------------------------------------
    # Separation
    # ------------------------------------------------------------------
    def _on_separation_event(self, result=None) -> None:
        """Refresh the panel.  Runs on the GUI thread via the bridge signal."""
        self._refresh_separation_attempts()
        self._refresh_separation_status()
        if result is not None and not result.ok:
            self._status(
                f"Separation failed: {result.error}", error=True
            )

    def _refresh_separation_attempts(self) -> None:
        if self._stored is None:
            self.separation.set_attempts([])
            return
        try:
            attempts = self.controller.separation_attempts(self._stored)
        except Exception as exc:  # pragma: no cover - defensive
            self.separation.set_status(f"Could not read attempts: {exc}")
            return
        self.separation.set_attempts(attempts)

    def _refresh_separation_status(self) -> None:
        try:
            status = self.controller.separation_status()
        except Exception as exc:  # pragma: no cover - defensive
            self.separation.set_status(f"Separation unavailable: {exc}")
            return
        worker = status.get("worker") or {}
        model = status.get("model") or {}
        reason = status.get("unavailable_reason")
        self.separation.set_availability(not reason, reason)
        if reason:
            return
        bits = []
        if worker.get("busy"):
            bits.append(f"running: {worker.get('current_query') or ''}")
        elif worker.get("pending"):
            bits.append(f"{worker['pending']} queued")
        if model.get("loaded"):
            bits.append(f"model loaded, {float(model.get('rss_mb') or 0):.0f} MB")
        ratio = worker.get("last_realtime_ratio")
        if ratio:
            bits.append(f"last run {ratio:.1f}x realtime")
        if worker.get("last_error"):
            bits.append(f"last error: {worker['last_error']}")
        self.separation.set_status("   ".join(bits))

    def _on_separate(self, query: str, use_region: bool) -> None:
        if self._stored is None:
            self._status("Select an event first", error=True)
            return
        start = end = None
        if use_region and self._region is not None:
            start, end = self._region
        accepted, message = self.controller.separate(
            self._stored, query, start, (end - start) if start is not None else None
        )
        self._status(message, error=not accepted)
        self._refresh_separation_status()

    def _on_compare_separation(self, path: str) -> None:
        """A/B: load the original, or a separated file, into the transport."""
        if path:
            self._pending_separation_audio = (
                self.controller.begin_separation_audio(path)
            )
            self._playback_source = "isolated"
        else:
            if self._stored is None:
                return
            self._pending_audio = self.controller.begin_audio(self._stored)
            self._playback_source = "original"
        # _pump_pending re-schedules itself while work is outstanding, so the
        # decode result is collected however long it takes.

    def _on_save_separation(self, path: str) -> None:
        target, _ = QFileDialog.getSaveFileName(
            self, "Save separated audio", os.path.basename(path), "WAV (*.wav)"
        )
        if not target:
            return
        try:
            shutil.copyfile(path, target)
        except OSError as exc:
            self._status(f"Could not save: {exc}", error=True)
            return
        self._status(f"Saved to {target}")

    def _on_region_changed(self, start_fraction: float, end_fraction: float) -> None:
        """Turn a dragged region into seconds on the event's own timeline.

        Driven by either the waveform or the spectrogram, and mirrored onto
        both: they show the same event on the same timeline, so a region drawn
        on one has to appear on the other or the operator cannot tell which
        span is selected.
        """
        collapsed = (end_fraction - start_fraction < 1e-4)
        if collapsed or self._event_duration <= 0:
            self._region = None
        else:
            self._region = (
                start_fraction * self._event_duration,
                end_fraction * self._event_duration,
            )
        self._sync_region_views()

    def _sync_region_views(self) -> None:
        """Push the current region to every view that shows one."""
        region = self._region
        for view in (self.waveform, self.spectrogram):
            if region is None:
                view.clear_region()
            else:
                start = region[0] / self._event_duration \
                    if self._event_duration else 0.0
                end = region[1] / self._event_duration \
                    if self._event_duration else 0.0
                view.set_region(start, end)
        self.separation.set_region(*region) if region else \
            self.separation.set_region(None, None)
        if region is None:
            self.selection_label.setText("")
            self.extract_button.setEnabled(False)
        else:
            self.selection_label.setText(
                f"selected {region[0]:.2f}-{region[1]:.2f}s "
                f"({region[1] - region[0]:.2f}s)"
            )
            self.extract_button.setEnabled(
                self._stored is not None and bool(self._stored.audio_path)
            )

    def _on_select_region_toggled(self, enabled: bool) -> None:
        """Arm or disarm region selection everywhere, from either control."""
        self.waveform.set_selection_enabled(enabled)
        self.spectrogram.set_selection_enabled(enabled)
        # Reflect the state on the other control, without re-entering this
        # handler: setChecked only emits on a change, and the guard makes that
        # impossible regardless.
        if self.separation.region_button.isChecked() != enabled:
            self.separation.region_button.blockSignals(True)
            self.separation.region_button.setChecked(enabled)
            self.separation.region_button.blockSignals(False)
            # The panel keeps its own "use this region" flag, which the
            # blocked signal would otherwise have left stale.
            self.separation._use_region = bool(enabled)
        if not enabled:
            self._clear_region()

    def _clear_region(self) -> None:
        self._region = None
        self._sync_region_views()

    # ------------------------------------------------------------------
    # Extracting a selected region as its own event
    # ------------------------------------------------------------------
    def _on_extract_selection(self) -> None:
        """Save the selected region as a new event.

        The point of the feature: a long event can contain the sound worth
        keeping along with a car going past and somebody honking, and the
        operator should be able to keep just the part they care about without
        losing the original.

        The extraction re-analyses the selection with the real pipeline, so the
        result is a proper event - measured, classified and fingerprinted - and
        not a renamed file.  It runs on a worker thread; the window stays
        usable while it works.
        """
        if self._stored is None or self._region is None:
            self._status("Select a region first", error=True)
            return
        if not self._stored.audio_path:
            self._status(
                "This event has no audio to extract from; its fingerprint and "
                "measurements are still available.",
                error=True,
            )
            return
        start, end = self._region
        self._pending_extract = self.controller.begin_extract_selection(
            self._stored, start, end
        )
        self.extract_button.setEnabled(False)
        self._status(
            f"Extracting {start:.2f}-{end:.2f}s as a new event... "
            f"analysing it takes a moment."
        )
        QTimer.singleShot(200, self._pump_extract)

    def _pump_extract(self) -> None:
        pending = self._pending_extract
        if pending is None:
            return
        if not pending.done:
            QTimer.singleShot(200, self._pump_extract)
            return
        self._pending_extract = None
        if pending.error:
            self._status(f"Extraction failed: {pending.error}", error=True)
            self._sync_region_views()
            return
        result = pending.result
        if result is None or not result.ok:
            self._status("Extraction produced nothing", error=True)
            self._sync_region_views()
            return
        self.refresh()
        stored, description = self.controller.select(
            result.stored_events[0])
        if stored is not None:
            self._stored = stored
            self._description = description
            self._fill_panels(description)
            self.review_bar.load(stored)
            self.waveform.clear()
            self.spectrogram.clear()
            self._event_duration = 0.0
            self._duration = 0.0
            self._playback_source = "original"
            self._clear_region()
            self._set_transport_enabled(False)
            self._pending_audio = self.controller.begin_audio(stored)
            self._pump_pending()
        note = ""
        if result.stored_without_detection:
            note = (
                "  The detector did not fire on the selection, so it is "
                "stored as given: audio, duration and level, with no "
                "classification and no fingerprint."
            )
        self._status(
            f"Saved {result.stored_events[0]} from "
            f"{result.parent_event_id} "
            f"({result.duration_seconds:.2f}s).{note}"
        )

    # ------------------------------------------------------------------
    def _install_actions(self) -> None:
        play = QAction("Play/Pause", self)
        play.setShortcut(QKeySequence(Qt.Key_Space))
        play.triggered.connect(self.toggle_play)
        self.addAction(play)

    def _install_timers(self) -> None:
        # Playhead: only while something is playing.
        self._playhead = QTimer(self)
        self._playhead.setInterval(PLAYHEAD_MS)
        self._playhead.timeout.connect(self._on_playhead)

        # New-event detection.  One cheap aggregate query per tick, and the list
        # is only rebuilt when that aggregate actually changes.
        self._live = QTimer(self)
        self._live.setInterval(LIVE_POLL_MS)
        self._live.timeout.connect(self._on_live_tick)
        self._live.start()

        # Capture status. Faster than the event poll, because the question it
        # answers - is audio arriving now - changes in milliseconds when the
        # answer changes.
        self._live_capture = QTimer(self)
        self._live_capture.setInterval(LIVE_CAPTURE_POLL_MS)
        self._live_capture.timeout.connect(self._refresh_live)
        self._live_capture.start()
        self._refresh_live()

    # ------------------------------------------------------------------
    # Data
    # ------------------------------------------------------------------
    def refresh(self) -> None:
        """Reload the list from the database."""
        choice = self.filter_combo.currentText()
        mapping = {
            "All": None,
            "Not reviewed": Decision.UNREVIEWED.value,
            "Saved": Decision.SAVED.value,
            "Confirmed": Decision.CONFIRMED.value,
            "Uncertain": Decision.UNCERTAIN.value,
            "Rejected": Decision.REJECTED.value,
        }
        wanted = mapping.get(choice)
        events = self.controller.list_events(
            decisions=[wanted] if wanted else None
        )
        self.event_list.set_rows(build_rows(events))
        try:
            stats = self.controller.stats()
            reviewed = sum(
                count for state, count in stats["by_decision"].items()
                if state != Decision.UNREVIEWED.value
            )
            self.event_list.set_summary(
                f"{reviewed} of {stats['events']} reviewed"
            )
        except Exception as exc:  # pragma: no cover - defensive
            self.event_list.set_summary(f"unavailable: {exc}")

    # ------------------------------------------------------------------
    def select_event(self, event_id: str) -> None:
        """Load one event.  Never recomputes analysis or a fingerprint."""
        # Only tear the audio stream down if there is one.  Calling stop()
        # unconditionally closes and reopens a PortAudio stream on every
        # selection, which is tens of milliseconds on the GUI thread.
        if self.player.is_loaded or self.player.state is not PlaybackState.STOPPED:
            self.player.stop()
        self._playhead.stop()
        self.play_button.setText("Play")

        stored, description = self.controller.select(event_id)
        if stored is None:
            self.title_label.setText("Event not found")
            self.subtitle_label.setText("")
            return
        self._stored = stored
        self._description = description

        self.title_label.setText(
            f"{stored.event_id}    {description.headline[2].value}"
        )
        headline = {f.label: f.value for f in description.headline}
        source = {f.label: f.value for f in description.source}
        # The source belongs next to the event identity, not buried in a tab:
        # "which input did this come from" is the first question about a capture.
        self.subtitle_label.setText(
            f"{headline.get('Time', '')}   "
            f"{headline.get('Decision', '')}   "
            f"source: {source.get('Source', '')}   "
            f"{headline.get('Audio completeness', '')}"
        )
        self._fill_panels(description)
        self.review_bar.load(stored)

        self.waveform.clear()
        self.spectrogram.clear()
        self.similar_label.setText("Not computed")
        self._duration = 0.0
        self._event_duration = 0.0
        self._playback_source = "original"
        self._clear_region()
        self._set_transport_enabled(False)

        # Audio, then the envelope derived from it.  Both on worker threads.
        self._pending_audio = self.controller.begin_audio(stored)
        self._pump_pending()

    def _fill_panels(self, description) -> None:
        for title, _heading, section in self._panel_sections:
            self._panels[title].set_fields(getattr(description, section))

    # ------------------------------------------------------------------
    def _pump_pending(self) -> None:
        """Collect finished background work.  Never blocks."""
        audio = self._pending_audio
        if audio is not None and audio.done:
            self._pending_audio = None
            if audio.ok and audio.samples is not None:
                try:
                    self.player.load(audio.samples, self.controller.sample_rate)
                    self._duration = audio.samples.size / (
                        self.controller.sample_rate or 1
                    )
                    # The event's own length, for measuring a dragged region.
                    # Not the transport's length: the transport may be holding
                    # a separated file.
                    if audio.kind == "audio":
                        self._event_duration = self._duration
                    self._set_transport_enabled(True)
                    self.duration_label.setText(
                        f"{self._duration:.1f} s"
                    )
                    self._pending_envelope = (
                        self.controller.begin_envelope(self._stored)
                    )
                except PlaybackError as exc:
                    self.waveform.set_message(f"Cannot load audio: {exc}")
                    self._set_transport_enabled(False)
            else:
                # Missing or corrupt audio must not stop review.
                message = audio.error or "Audio unavailable"
                if audio.error_kind == "corrupt":
                    message = f"Audio unreadable - {message}"
                elif audio.error_kind == "missing":
                    message = "Audio unavailable"
                self.waveform.set_message(message)
                self.spectrogram.clear(message)
                self._set_transport_enabled(False)
                self._status(f"{self._stored.event_id}: {message}")

        separated = self._pending_separation_audio
        if separated is not None and separated.done:
            self._pending_separation_audio = None
            if separated.ok and separated.samples is not None:
                try:
                    self.player.load(
                        separated.samples, self.controller.sample_rate
                    )
                    self._duration = separated.samples.size / (
                        self.controller.sample_rate or 1
                    )
                    self._set_transport_enabled(True)
                    self.duration_label.setText(f"{self._duration:.1f} s")
                    self._status(
                        f"Playing the separated audio ({self._duration:.1f} s). "
                        f"Use Compare: original to switch back."
                    )
                except PlaybackError as exc:
                    self.waveform.set_message(f"Cannot load audio: {exc}")
                    self._set_transport_enabled(False)
            else:
                self._set_transport_enabled(False)
                self._status(
                    f"Separated audio unavailable: {separated.error}",
                    error=True,
                )

        envelope = self._pending_envelope
        if envelope is not None and envelope.done:
            self._pending_envelope = None
            cached = self.controller.cached_envelope(
                self._stored.event_id,
                columns=self.controller.envelope_columns,
            )
            self.waveform.set_envelope(cached)
            if cached is not None:
                self.waveform.set_message("")

        # Keep collecting until nothing is outstanding.  Audio, envelope and
        # separated audio are decoded on worker threads, so the call that
        # started them almost always returns before they are finished; without
        # re-scheduling, the result is never collected and the waveform stays
        # empty.  Same pattern the spectrogram pump already uses.
        if any(
            pending is not None and not pending.done
            for pending in (
                self._pending_audio, self._pending_envelope,
                self._pending_separation_audio,
            )
        ):
            QTimer.singleShot(120, self._pump_pending)

    # ------------------------------------------------------------------
    def _on_tab_changed(self, index: int) -> None:
        """Build a derived view only when it is actually opened."""
        if self.tabs.tabText(index) == "Spectrogram":
            self._build_spectrogram()
        elif self.tabs.tabText(index) == "Similar":
            self._build_similar()
        elif self.tabs.tabText(index) == "Separate":
            # Touching the controller here is what creates the worker, so the
            # model process is not started by opening the window.
            self._refresh_separation_status()
            self._refresh_separation_attempts()
            try:
                self.separation.set_queries(
                    self.controller.recent_queries()
                )
            except Exception:  # pragma: no cover - defensive
                pass

    def _build_similar(self) -> None:
        if self._stored is None:
            self.similar_label.setText("Select an event first")
            return
        try:
            hits = self.controller.similar(self._stored.event_id, limit=5)
        except Exception as exc:
            self.similar_label.setText(f"Similarity unavailable: {exc}")
            return
        if not hits:
            self.similar_label.setText(
                "No comparable events stored, or this event has no "
                "fingerprint."
            )
            return
        lines = ["Closest stored events:", ""]
        for other_id, distance, other in hits:
            label = (other.label or "no label") if other else "unavailable"
            lines.append(f"  {other_id}   distance {distance:.3f}   {label}")
        self.similar_label.setText("\n".join(lines))

    def _build_spectrogram(self) -> None:
        """Build the spectrogram on demand, off the GUI thread."""
        if self._stored is None:
            return
        if self.controller.cached_spectrogram(self._stored.event_id) is None:
            self.spectrogram.clear("Generating spectrogram...")
            self._pending_spectrogram = (
                self.controller.begin_spectrogram(self._stored)
            )
            QTimer.singleShot(150, self._pump_spectrogram)
        else:
            self.spectrogram.set_spectrogram(
                self.controller.cached_spectrogram(self._stored.event_id)
            )

    def _pump_spectrogram(self) -> None:
        pending = self._pending_spectrogram
        if pending is None:
            return
        if not pending.done:
            QTimer.singleShot(120, self._pump_spectrogram)
            return
        self._pending_spectrogram = None
        if pending.ok:
            self.spectrogram.set_spectrogram(pending.spectrogram)
        else:
            self.spectrogram.clear(
                pending.error or "Spectrogram unavailable"
            )

    # ------------------------------------------------------------------
    # Playback
    # ------------------------------------------------------------------
    def toggle_play(self) -> None:
        if not self.player.is_loaded:
            return
        try:
            if self.player.state is PlaybackState.PLAYING:
                self.player.pause()
                self.play_button.setText("Play")
                self._playhead.stop()
            else:
                self.player.play()
                self.play_button.setText("Pause")
                self._playhead.start()
        except PlaybackError as exc:
            # Playback failure must not take the window or the detector down.
            self._status(f"Playback failed: {exc}", error=True)

    def replay(self) -> None:
        self.player.stop()
        try:
            self.player.play()
        except PlaybackError as exc:
            self._status(f"Playback failed: {exc}", error=True)
            return
        self.play_button.setText("Pause")
        self._playhead.start()

    def _on_playhead(self) -> None:
        """Update the marker.  Runs only during playback."""
        try:
            duration = self.player.duration_seconds or self._duration
            position = self.player.position_seconds
        except Exception:  # pragma: no cover - defensive
            return
        if duration > 0:
            fraction = max(0.0, min(1.0, position / duration))
            self.waveform.set_position(fraction)
            if not self.seek_slider.isSliderDown():
                self.seek_slider.blockSignals(True)
                self.seek_slider.setValue(int(fraction * 1000))
                self.seek_slider.blockSignals(False)
            self.position_label.setText(f"{position:.1f} s")
        if self.player.state is not PlaybackState.PLAYING:
            self._playhead.stop()
            self.play_button.setText("Play")

    def _on_seek(self, fraction: float) -> None:
        if not self.player.is_loaded:
            return
        try:
            self.player.seek(fraction * self.player.duration_seconds)
        except PlaybackError as exc:
            self._status(f"Seek failed: {exc}", error=True)
        self._on_playhead()

    def _set_transport_enabled(self, enabled: bool) -> None:
        self.play_button.setEnabled(enabled)
        self.replay_button.setEnabled(enabled)
        self.seek_slider.setEnabled(enabled)

    # ------------------------------------------------------------------
    # Annotation
    # ------------------------------------------------------------------
    def _on_annotation(self, event_id: str, decision: str, label: str,
                       confidence) -> None:
        """Record the user's decision through the controller."""
        notes = self.review_bar.notes_edit.text().strip() or None
        try:
            self.controller.annotate(
                event_id,
                decision=decision,
                label=label or None,
                confidence=confidence,
                notes=notes,
            )
        except Exception as exc:
            self.review_bar.set_feedback(f"Could not save: {exc}", error=True)
            return
        self.review_bar.set_feedback("Annotation saved.")
        self.refresh()
        stored, description = self.controller.select(event_id)
        if stored is not None:
            self._stored = stored
            self._description = description
            self._fill_panels(description)
            self.review_bar.load(stored)

    # ------------------------------------------------------------------
    def _on_live_tick(self) -> None:
        """Notice new events without a full rescan or a restart."""
        if not self.isVisible():
            return
        try:
            marker = self.controller.change_marker()
        except Exception as exc:
            self._status(f"Database unavailable: {exc}", error=True)
            return
        if marker != self._change_marker:
            self._change_marker = marker
            self.refresh()
            self._status(
                f"New event detected - {marker[0]} event(s) recorded"
            )
            # The list is rebuilt, so re-select the current event without
            # reloading its audio, and never interrupt playback.
            current = self.event_list.selected_id()
            if current:
                self.event_list.select_event(current, notify=False)

    def _status(self, message: str, error: bool = False) -> None:
        self.status_label.setText(message)
        if error:
            self.status_label.setStyleSheet(
                f"color: {self.theme.color('danger')};"
            )
        else:
            self.status_label.setStyleSheet("")

    def _update_header_status(self) -> None:
        try:
            stats = self.controller.stats()
            self.header_status.setText(
                f"{stats['events']} event(s)   "
                f"db: {os.path.basename(self.db_path)}"
            )
        except Exception:  # pragma: no cover - defensive
            self.header_status.setText("")

    # ------------------------------------------------------------------
    def showEvent(self, event):
        super().showEvent(event)
        self._update_header_status()

    def keyPressEvent(self, event):
        # A cheap way to jump between events from the keyboard.
        if event.key() in (Qt.Key_Down, Qt.Key_Up):
            index = self.event_list.list.indexOfTopLevelItem(
                self.event_list.list.currentItem()
            )
            step = 1 if event.key() == Qt.Key_Down else -1
            count = self.event_list.list.topLevelItemCount()
            target = max(0, min(count - 1, index + step))
            if target != index and count:
                self.event_list.list.setCurrentItem(
                    self.event_list.list.topLevelItem(target)
                )
            return
        super().keyPressEvent(event)

    def closeEvent(self, event):
        try:
            self.player.close()
        except Exception:  # pragma: no cover - defensive
            pass
        # Stops the separation worker and the model process it owns.  Without
        # this the model would outlive the window by minutes.
        try:
            self.controller.remove_separation_listener(
                self._separation_listener
            )
        except Exception:  # pragma: no cover - defensive
            pass
        try:
            self.controller.close_separation()
        except Exception:  # pragma: no cover - defensive
            pass
        try:
            self.controller.db.close()
        except Exception:  # pragma: no cover - defensive
            pass
        super().closeEvent(event)
