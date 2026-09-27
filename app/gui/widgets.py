"""Qt widgets for the review workstation.

Thin views.  Every widget reads from a :class:`~app.gui.controller.ReviewController`
and holds no database or audio logic, which is what lets the interesting
behaviour be tested without a display.

Two performance rules are load-bearing:

* **Nothing expensive happens on the paint event.**  Waveform and spectrogram
  are QImages built once in a worker thread; painting blits them.
* **No busy waiting.**  The only timers are a playhead, running solely during
  playback, and a low-frequency change marker so new events appear without a
  restart.  There is no per-frame polling anywhere.
"""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QImage, QPainter, QPen
from PyQt5.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QPushButton,
    QSizePolicy,
    QSlider,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..events.database import Decision
from .audioview import ramp_lut
from .formatting import EventRow, format_decision, user_confidence

#: Playhead refresh while playing.  50 ms is smooth enough for a playhead and
#: costs a fraction of a percent; it does not run when paused or stopped.
PLAYHEAD_MS = 50

#: How often to check for new events when nothing is pushing them.
LIVE_POLL_MS = 2000


class WaveformWidget(QWidget):
    """Event waveform with a playback marker and click-to-seek.

    Draws a prebuilt min/max envelope plus a playhead, so a repaint is a handful
    of line segments rather than a re-reduction of the audio.
    """

    position_changed = pyqtSignal(float)   # fraction 0..1
    seek_requested = pyqtSignal(float)     # fraction 0..1
    #: A region was dragged, as two fractions of the event's length.  Emitted
    #: on release, and (0.0, 0.0) when the drag collapsed to a point, which
    #: the window reads as "no region".
    region_changed = pyqtSignal(float, float)

    def __init__(self, theme, parent=None):
        super().__init__(parent)
        self._theme = theme
        self._peaks = np.zeros(0, dtype=np.float32)
        self._troughs = np.zeros(0, dtype=np.float32)
        self._position = 0.0
        self._message = ""
        self.setMinimumHeight(110)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMouseTracking(True)
        self._tooltip = ""
        # Region selection is opt-in, so the default interaction stays
        # click-to-seek: a drag that meant "seek here" must not silently
        # become "separate this part".
        self._selection_enabled = False
        self._selecting = False
        self._anchor = 0.0
        self._region: Optional[tuple] = None

    # ------------------------------------------------------------------
    def set_envelope(self, envelope) -> None:
        if envelope is None:
            self._peaks = np.zeros(0, dtype=np.float32)
            self._troughs = np.zeros(0, dtype=np.float32)
        else:
            self._peaks = np.asarray(envelope.peaks, dtype=np.float32)
            self._troughs = np.asarray(envelope.troughs, dtype=np.float32)
        self.update()

    def set_position(self, fraction: float) -> None:
        fraction = max(0.0, min(1.0, float(fraction)))
        if abs(fraction - self._position) > 1e-4:
            self._position = fraction
            self.update()

    def set_message(self, message: str) -> None:
        self._message = message or ""
        self.update()

    def clear(self) -> None:
        self._peaks = np.zeros(0, dtype=np.float32)
        self._troughs = np.zeros(0, dtype=np.float32)
        self._position = 0.0
        self._message = ""
        self.update()

    # ------------------------------------------------------------------
    # Region selection
    # ------------------------------------------------------------------
    def set_selection_enabled(self, enabled: bool) -> None:
        self._selection_enabled = bool(enabled)
        if not enabled:
            self._selecting = False
        self.setCursor(
            Qt.CrossCursor if enabled else Qt.PointingHandCursor
        )
        self.update()

    @property
    def selection_enabled(self) -> bool:
        return self._selection_enabled

    def set_region(self, start: float, end: float) -> None:
        """Show a region given as fractions, ignoring a collapsed one."""
        if end - start < 1e-4:
            self.clear_region()
            return
        self._region = (max(0.0, min(1.0, start)), max(0.0, min(1.0, end)))
        self.update()

    def clear_region(self) -> None:
        self._region = None
        self._selecting = False
        self.update()

    def region(self) -> Optional[tuple]:
        return self._region

    # ------------------------------------------------------------------
    def _fraction_at(self, x: int) -> float:
        width = max(1, self.width())
        return max(0.0, min(1.0, x / width))

    def mousePressEvent(self, event):
        if not self._peaks.size:
            return
        fraction = self._fraction_at(int(event.position().x()))
        if self._selection_enabled and event.button() == Qt.LeftButton:
            # Drag out a region to separate.  Seeking still works on release
            # of a collapsed drag, which is the least surprising behaviour:
            # a click in selection mode is a zero-width region, and the
            # caller turns that back into "no region".
            self._selecting = True
            self._anchor = fraction
            self._region = (fraction, fraction)
            self.update()
            return
        self.seek_requested.emit(fraction)
        self.set_position(fraction)

    def mouseReleaseEvent(self, event):
        if not self._selecting:
            return
        self._selecting = False
        end = self._fraction_at(int(event.position().x()))
        start, end = sorted((self._anchor, end))
        if end - start < 1e-4:
            self._region = None
            self.region_changed.emit(0.0, 0.0)
        else:
            self._region = (start, end)
            self.region_changed.emit(start, end)
        self.update()

    def mouseMoveEvent(self, event):
        if not self._peaks.size:
            return
        fraction = self._fraction_at(int(event.position().x()))
        if self._selecting:
            start, end = sorted((self._anchor, fraction))
            self._region = (start, end)
            self.update()
            return
        index = int(fraction * (self._peaks.size - 1))
        index = max(0, min(self._peaks.size - 1, index))
        self.setToolTip(
            f"{float(self._troughs[index]):+.3f} .. "
            f"{float(self._peaks[index]):+.3f}"
        )

    # ------------------------------------------------------------------
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        rect = self.rect()
        background = QColor(self._theme.color("bg-elev-1", "#161028"))
        painter.fillRect(rect, background)

        if self._peaks.size == 0:
            painter.setPen(QColor(self._theme.color("text-subtle", "#9698ab")))
            painter.drawText(
                rect, Qt.AlignCenter,
                self._message or "No waveform",
            )
            return

        # Centre line at zero.
        middle = rect.height() / 2.0
        painter.setPen(QPen(QColor(self._theme.color("border", "#3d2d61")), 1))
        painter.drawLine(0, int(middle), rect.width(), int(middle))

        # Map the envelope onto the widget width, column for column.
        count = int(self._peaks.size)
        width = rect.width()
        scale_x = width / max(count, 1)
        peak = float(
            max(abs(self._peaks.max()), abs(self._troughs.min()), 1e-6)
        )
        scale_y = (rect.height() / 2.0 - 2) / peak

        pen = QPen(QColor(self._theme.color("primary", "#12f012")), 1)
        painter.setPen(pen)
        previous = None
        for index in range(count):
            x = index * scale_x
            top = middle - self._peaks[index] * scale_y
            bottom = middle - self._troughs[index] * scale_y
            painter.drawLine(
                int(x), int(top), int(x + max(scale_x, 1.0)), int(top)
            )
            if bottom != top:
                painter.drawLine(
                    int(x), int(bottom), int(x + max(scale_x, 1.0)), int(bottom)
                )
            previous = x

        # Selected region, drawn under the waveform but over the background so
        # the selection is unmistakable.  A/B comparison depends on the user
        # being certain which part of the event they isolated.
        if self._region is not None:
            start, end = self._region
            x0 = int(start * width)
            x1 = int(end * width)
            highlight = QColor(
                self._theme.color("primary", "#12f012")
            )
            highlight.setAlpha(46 if not self._selecting else 76)
            painter.fillRect(x0, 0, max(1, x1 - x0), rect.height(), highlight)
            painter.setPen(
                QPen(QColor(self._theme.color("primary", "#12f012")), 1)
            )
            painter.drawLine(x0, 0, x0, rect.height())
            painter.drawLine(x1, 0, x1, rect.height())

        # Playhead.
        if self._position > 0.0:
            x = int(self._position * width)
            painter.setPen(
                QPen(QColor(self._theme.color("accent", "#00d1ff")), 2)
            )
            painter.drawLine(x, 0, x, rect.height())
        painter.end()


class SpectrogramWidget(QWidget):
    """Optional, lazily built event spectrogram.

    Colours come from the theme pack's own spectral ramp, so the display matches
    the design system's intent rather than an arbitrary colormap.
    """

    def __init__(self, theme, parent=None):
        super().__init__(parent)
        self._theme = theme
        self._image: Optional[QImage] = None
        self._max_hz = 0.0
        self._message = "Spectrogram not generated"
        self.setMinimumHeight(150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._lut = ramp_lut(theme.spectral_ramp())

    def set_spectrogram(self, spectrogram) -> None:
        if spectrogram is None or spectrogram.data.size == 0:
            self._image = None
            self._message = "Spectrogram unavailable"
            self.update()
            return
        data = np.clip(spectrogram.data, 0.0, 1.0)
        # (bins, columns) -> (columns, bins) for an image, then colourise.
        indices = (data.T * 255.0).astype(np.int32)
        rgb = self._lut[np.clip(indices, 0, 255)]
        height, width = rgb.shape[0], rgb.shape[1]
        image = QImage(
            rgb.data, width, height, 3 * width, QImage.Format_RGB888
        ).copy()   # .copy(): the numpy buffer is temporary
        self._image = image
        self._max_hz = float(spectrogram.max_hz)
        self._message = ""
        self.update()

    def clear(self, message: str = "Spectrogram not generated") -> None:
        self._image = None
        self._message = message
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        rect = self.rect()
        painter.fillRect(
            rect, QColor(self._theme.color("bg-elev-1", "#161028"))
        )
        if self._image is None:
            painter.setPen(QColor(self._theme.color("text-subtle", "#9698ab")))
            painter.drawText(rect, Qt.AlignCenter, self._message)
            return
        painter.drawImage(rect, self._image)
        if self._max_hz:
            painter.setPen(QPen(QColor(self._theme.color("text-subtle")), 1))
            painter.drawText(
                6, 14, f"0 - {self._max_hz / 1000:.1f} kHz"
            )


#: Default columns.  Compact on purpose; the wider set is opt-in.
LIST_COLUMNS = ("Time", "Duration", "Decision", "Label")
LIST_COLUMNS_FULL = LIST_COLUMNS + (
    "Onsets", "Presence", "Level dB", "SNR dB", "Reason",
)


class EventListWidget(QWidget):
    """The event browser.

    A tree rather than a list, so the columns actually line up.  The default
    column set stays compact; the technical columns are opt-in, so the primary
    view does not look like a database dump.
    """

    event_selected = pyqtSignal(str)

    def __init__(self, theme, parent=None):
        super().__init__(parent)
        self._theme = theme
        self._rows: list = []
        self._show_detail = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        header = QHBoxLayout()
        title = QLabel("Events")
        title.setObjectName("Heading")
        header.addWidget(title)
        header.addStretch(1)
        self._count = QLabel("0")
        self._count.setObjectName("Subtle")
        header.addWidget(self._count)
        self._detail = QPushButton("Details")
        self._detail.setCheckable(True)
        self._detail.setToolTip(
            "Show onset, presence, level, SNR and segmentation columns"
        )
        self._detail.toggled.connect(self.set_detail_columns)
        header.addWidget(self._detail)
        layout.addLayout(header)

        self.list = QTreeWidget()
        self.list.setColumnCount(len(LIST_COLUMNS))
        self.list.setHeaderLabels(list(LIST_COLUMNS))
        self.list.setRootIsDecorated(False)
        self.list.setAlternatingRowColors(False)
        self.list.setUniformRowHeights(True)
        self.list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.list.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.list.currentItemChanged.connect(self._on_current)
        header_view = self.list.header()
        # The last section stretches, so the fixed widths below are a floor for
        # the other columns rather than a total that overflows the pane.
        header_view.setStretchLastSection(True)
        header_view.setMinimumSectionSize(48)
        layout.addWidget(self.list, 1)

        self._status = QLabel("0 reviewed")
        self._status.setObjectName("Subtle")
        layout.addWidget(self._status)

    # ------------------------------------------------------------------
    #: Wide enough for the formatted values, so nothing is elided.
    _WIDTHS = (76, 74, 88, 116, 60, 70, 70, 64, 86)

    def _apply_column_widths(self) -> None:
        for index, width in enumerate(self._WIDTHS):
            if index < self.list.columnCount():
                self.list.setColumnWidth(index, width)

    def set_detail_columns(self, enabled: bool) -> None:
        self._show_detail = bool(enabled)
        self.list.setColumnCount(
            len(LIST_COLUMNS_FULL) if self._show_detail else len(LIST_COLUMNS)
        )
        self.list.setHeaderLabels(
            list(LIST_COLUMNS_FULL) if self._show_detail else list(LIST_COLUMNS)
        )
        self._apply_column_widths()
        if self._rows:
            self.set_rows(self._rows)

    def set_rows(self, rows: list) -> None:
        self._rows = list(rows)
        selected_id = self.selected_id()
        self.list.blockSignals(True)
        self.list.clear()
        for row in self._rows:
            values = row.full() if self._show_detail else row.compact()
            item = QTreeWidgetItem([str(v) for v in values])
            item.setData(0, Qt.UserRole, row.event_id)
            item.setToolTip(0, row.tooltip())
            self.list.addTopLevelItem(item)
        self.list.blockSignals(False)
        self._count.setText(str(len(self._rows)))
        self._apply_column_widths()
        target = selected_id or (self._rows[0].event_id if self._rows else None)
        if target:
            self.select_event(target, notify=False)

    def select_event(self, event_id: str, notify: bool = True) -> None:
        for index in range(self.list.topLevelItemCount()):
            item = self.list.topLevelItem(index)
            if item.data(0, Qt.UserRole) == event_id:
                self.list.blockSignals(not notify)
                self.list.setCurrentItem(item)
                self.list.blockSignals(False)
                return

    def selected_id(self) -> Optional[str]:
        item = self.list.currentItem()
        return item.data(0, Qt.UserRole) if item is not None else None

    def set_summary(self, text: str) -> None:
        self._status.setText(text)

    def _on_current(self, current, previous):
        if current is None:
            return
        event_id = current.data(0, Qt.UserRole)
        if event_id:
            self.event_selected.emit(event_id)


class FieldPanel(QFrame):
    """A titled group of label/value rows."""

    def __init__(self, title: str, theme, parent=None):
        super().__init__(parent)
        self.setObjectName("Card")
        self._layout = QVBoxLayout(self)
        self._layout.setContentsMargins(12, 10, 12, 10)
        self._layout.setSpacing(4)
        heading = QLabel(title)
        heading.setObjectName("Heading")
        self._layout.addWidget(heading)
        self._body = QVBoxLayout()
        self._body.setSpacing(2)
        self._layout.addLayout(self._body)
        self._rows = []

    def set_fields(self, fields) -> None:
        """Show ``fields``, reusing existing rows.

        Rows are updated in place rather than destroyed and rebuilt.  With
        ~35 rows across four panels, tearing them down and recreating them on
        every event selection cost tens of milliseconds on the GUI thread and
        queued the old widgets for deferred deletion, so repeated selections
        made the window progressively slower.
        """
        fields = list(fields)
        # Grow or shrink only as much as necessary.
        while len(self._rows) < len(fields):
            self._rows.append(self._make_row())
        while len(self._rows) > len(fields):
            row = self._rows.pop()
            self._body.removeWidget(row)
            row.setParent(None)
            row.deleteLater()
        for row, field in zip(self._rows, fields):
            key_label, value_label = self._labels_of(row)
            key_label.setText(str(field.label))
            value_label.setText(str(field.value))

    def _make_row(self) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        key = QLabel("")
        key.setObjectName("Muted")
        key.setFixedWidth(180)
        value = QLabel("")
        value.setWordWrap(True)
        value.setTextInteractionFlags(Qt.TextSelectableByMouse)
        layout.addWidget(key)
        layout.addWidget(value, 1)
        self._body.addWidget(row)
        return row

    @staticmethod
    def _labels_of(row: QWidget) -> tuple:
        labels = row.findChildren(QLabel)
        # Key first, value second, as constructed.
        return labels[0], labels[1]


class ReviewBar(QWidget):
    """The decision controls.

    Uses the backend's Decision enum unchanged.  A label is optional and never
    suggested, and the reviewer's confidence is labelled as theirs so it cannot
    be read as the detector's.
    """

    annotation_saved = pyqtSignal(str, str, str, object)  # id, dec, label, conf

    #: Which decision each button records, from the backend enum.
    BUTTON_DECISIONS = (
        ("Save", Decision.SAVED),
        ("Confirm", Decision.CONFIRMED),
        ("Uncertain", Decision.UNCERTAIN),
        ("Reject", Decision.REJECTED),
    )

    def __init__(self, theme, parent=None):
        super().__init__(parent)
        self._theme = theme
        self._event_id = ""
        self._decision = Decision.UNREVIEWED

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        buttons = QHBoxLayout()
        self._buttons = {}
        for text, decision in self.BUTTON_DECISIONS:
            button = QPushButton(text)
            button.setCheckable(True)
            button.clicked.connect(
                lambda _checked, d=decision: self._choose(d)
            )
            if decision is Decision.REJECTED:
                button.setObjectName("Danger")
            self._buttons[decision] = button
            buttons.addWidget(button)
        buttons.addStretch(1)
        layout.addLayout(buttons)

        hint = QLabel(
            "Saving keeps the event. It is not a claim about what the sound is: "
            "a label is separate, and optional."
        )
        hint.setObjectName("Subtle")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        form = QHBoxLayout()
        form.setSpacing(8)

        self.label_edit = QLineEdit()
        self.label_edit.setPlaceholderText("Label (optional, yours)")
        self.label_edit.returnPressed.connect(self._emit_save)
        form.addWidget(self.label_edit, 2)

        self.confidence = QSlider(Qt.Horizontal)
        self.confidence.setRange(0, 100)
        self.confidence.setValue(0)
        self.confidence.setMaximumWidth(160)
        self.confidence.setToolTip(
            "Your confidence in this review. Separate from the detector's."
        )
        form.addWidget(QLabel("Your confidence:"))
        form.addWidget(self.confidence, 1)

        self._confidence_value = QLabel("not stated")
        self._confidence_value.setObjectName("Subtle")
        self._confidence_value.setFixedWidth(80)
        self.confidence.valueChanged.connect(
            lambda v: self._confidence_value.setText(f"{v}%")
        )
        form.addWidget(self._confidence_value)
        layout.addLayout(form)

        notes_row = QHBoxLayout()
        self.notes_edit = QLineEdit()
        self.notes_edit.setPlaceholderText("Notes (optional)")
        self.notes_edit.returnPressed.connect(self._emit_save)
        notes_row.addWidget(self.notes_edit, 1)
        save = QPushButton("Save annotation")
        save.setObjectName("Primary")
        save.clicked.connect(self._emit_save)
        notes_row.addWidget(save)
        layout.addLayout(notes_row)

        self._feedback = QLabel("")
        self._feedback.setObjectName("Subtle")
        layout.addWidget(self._feedback)

    # ------------------------------------------------------------------
    def load(self, stored) -> None:
        """Populate from a stored event, without inventing anything."""
        self._event_id = stored.event_id if stored else ""
        self._decision = stored.decision if stored else Decision.UNREVIEWED
        for decision, button in self._buttons.items():
            button.setChecked(decision is self._decision)
        self.label_edit.setText(stored.label or "" if stored else "")
        self.notes_edit.setText(stored.notes or "" if stored else "")
        if stored is not None and stored.confidence is not None:
            try:
                value = float(stored.confidence)
                if value <= 1.0:
                    value *= 100.0
                self.confidence.blockSignals(True)
                self.confidence.setValue(int(round(value)))
                self.confidence.blockSignals(False)
                self._confidence_value.setText(user_confidence(stored.confidence))
            except (TypeError, ValueError):
                self._confidence_value.setText("not stated")
        else:
            self._confidence_value.setText("not stated")
        self._feedback.setText("")

    def set_feedback(self, message: str, error: bool = False) -> None:
        self._feedback.setText(message)
        self._feedback.setStyleSheet(
            f"color: {self._theme.color('danger' if error else 'text-muted')};"
        )

    def _choose(self, decision: Decision) -> None:
        if not self._event_id:
            return
        self._decision = decision
        for other, button in self._buttons.items():
            button.setChecked(other is decision)
        self.annotation_saved.emit(
            self._event_id,
            decision.value,
            self.label_edit.text().strip(),
            None,
        )

    def _emit_save(self) -> None:
        if not self._event_id:
            return
        confidence = self.confidence.value() / 100.0
        self.annotation_saved.emit(
            self._event_id,
            self._decision.value,
            self.label_edit.text().strip(),
            confidence,
        )


class SeparationPanel(QWidget):
    """Query input, attempt history, and A/B comparison for separation.

    A thin view, like every other widget here: it emits *requests* and displays
    results, and holds no decision about what may be separated or where output
    goes.  Those belong to :class:`~app.gui.controller.ReviewController`.

    Two deliberate constraints:

    * **No suggested sound classes.**  Query suggestions are the queries the
      user has already run, supplied by the controller.  The backend does not
      know what sounds exist, and a window that hard-coded a list of them
      would put that knowledge in the interface, which the project forbids and
      which would be wrong the moment the detector's vocabulary changes.
    * **Cost is visible.**  Every attempt row shows the realtime ratio, because
      a separation takes seconds per second of audio and a user asking for four
      of them deserves to know that before asking for the fifth.
    """

    separate_requested = pyqtSignal(str, bool)   # query, use_region
    compare_requested = pyqtSignal(str)          # path, or "" for the original
    save_requested = pyqtSignal(str)             # path to save
    selection_toggled = pyqtSignal(bool)

    def __init__(self, theme, parent=None):
        super().__init__(parent)
        self._theme = theme
        self._attempts: list = []
        self._region_seconds: Optional[str] = None
        self._use_region = False
        self._build()

    def _build(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        query_row = QHBoxLayout()
        query_row.addWidget(QLabel("Query"))
        self.query_box = QComboBox()
        self.query_box.setEditable(True)
        self.query_box.setInsertPolicy(QComboBox.NoInsert)
        self.query_box.setPlaceholderText(
            "what to isolate, in your own words"
        )
        self.query_box.setMinimumWidth(280)
        self.query_box.currentTextChanged.connect(self._on_query_changed)
        query_row.addWidget(self.query_box, 1)

        self.separate_button = QPushButton("Isolate")
        self.separate_button.clicked.connect(self._on_separate)
        query_row.addWidget(self.separate_button)
        layout.addLayout(query_row)

        region_row = QHBoxLayout()
        self.region_button = QPushButton("Select a region")
        self.region_button.setCheckable(True)
        self.region_button.toggled.connect(self._on_region_toggled)
        region_row.addWidget(self.region_button)
        self.region_label = QLabel("Whole event")
        self.region_label.setObjectName("Muted")
        region_row.addWidget(self.region_label, 1)
        layout.addLayout(region_row)

        self.status_label = QLabel("")
        self.status_label.setObjectName("Muted")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        self.attempts_list = QListWidget()
        self.attempts_list.setSelectionMode(QAbstractItemView.SingleSelection)
        self.attempts_list.currentRowChanged.connect(self._on_attempt_selected)
        layout.addWidget(self.attempts_list, 1)

        buttons = QHBoxLayout()
        self.original_button = QPushButton("Compare: original")
        self.original_button.clicked.connect(
            lambda: self.compare_requested.emit("")
        )
        buttons.addWidget(self.original_button)

        self.isolated_button = QPushButton("Compare: isolated")
        self.isolated_button.clicked.connect(self._on_compare_isolated)
        buttons.addWidget(self.isolated_button)

        self.enhanced_button = QPushButton("Compare: enhanced")
        self.enhanced_button.clicked.connect(self._on_compare_enhanced)
        buttons.addWidget(self.enhanced_button)

        buttons.addStretch(1)
        self.save_button = QPushButton("Save isolated...")
        self.save_button.clicked.connect(self._on_save)
        buttons.addWidget(self.save_button)
        layout.addLayout(buttons)

        self._sync_buttons()

    # ------------------------------------------------------------------
    def query(self) -> str:
        return self.query_box.currentText().strip()

    def _on_query_changed(self, _text: str) -> None:
        self._sync_buttons()

    def _on_separate(self) -> None:
        self.separate_requested.emit(self.query(), self._use_region)

    def _on_region_toggled(self, checked: bool) -> None:
        self._use_region = bool(checked)
        if not checked:
            self._region_seconds = None
            self.region_label.setText("Whole event")
        self.selection_toggled.emit(bool(checked))
        self._sync_buttons()

    def _on_attempt_selected(self, _row: int) -> None:
        self._sync_buttons()

    def _on_compare_isolated(self) -> None:
        attempt = self.selected_attempt()
        if attempt is not None and attempt.isolated_path:
            self.compare_requested.emit(attempt.isolated_path)

    def _on_compare_enhanced(self) -> None:
        attempt = self.selected_attempt()
        if attempt is not None and attempt.has_enhanced:
            self.compare_requested.emit(attempt.enhanced_path)

    def _on_save(self) -> None:
        attempt = self.selected_attempt()
        if attempt is not None and attempt.isolated_path:
            self.save_requested.emit(attempt.isolated_path)

    # ------------------------------------------------------------------
    def selected_attempt(self):
        row = self.attempts_list.currentRow()
        if row < 0 or row >= len(self._attempts):
            return None
        return self._attempts[row]

    def set_queries(self, queries) -> None:
        """Offer previously used queries as suggestions."""
        current = self.query()
        self.query_box.blockSignals(True)
        self.query_box.clear()
        self.query_box.addItems(list(queries))
        self.query_box.setCurrentText(current)
        self.query_box.blockSignals(False)

    def set_attempts(self, attempts) -> None:
        """Replace the attempt list, keeping the selection when possible."""
        previous = self.selected_attempt()
        self._attempts = list(attempts)
        self.attempts_list.clear()
        for attempt in self._attempts:
            self.attempts_list.addItem(attempt.summary())
        if self._attempts:
            index = 0
            if previous is not None:
                for position, attempt in enumerate(self._attempts):
                    if attempt.name == previous.name:
                        index = position
                        break
            self.attempts_list.setCurrentRow(index)
        self._sync_buttons()

    def set_region(self, start_seconds: Optional[float],
                   end_seconds: Optional[float]) -> None:
        if start_seconds is None or end_seconds is None:
            self._region_seconds = None
            self.region_label.setText("Whole event")
        else:
            self._region_seconds = f"{start_seconds:.2f}s to {end_seconds:.2f}s"
            self.region_label.setText(self._region_seconds)
        self._sync_buttons()

    def set_status(self, text: str) -> None:
        self.status_label.setText(text or "")

    def set_availability(self, available: bool, reason: Optional[str]) -> None:
        """Enable or disable the controls, and say why when disabled.

        A missing model is stated in full rather than hidden behind a greyed
        button, because the fix is an installation step the user has to know
        about.
        """
        self.query_box.setEnabled(bool(available))
        self.separate_button.setEnabled(bool(available))
        if not available and reason:
            self.status_label.setText(reason)
        self._sync_buttons()

    def _sync_buttons(self) -> None:
        attempt = self.selected_attempt()
        has_audio = bool(attempt and attempt.isolated_path)
        self.original_button.setEnabled(bool(attempt))
        self.isolated_button.setEnabled(has_audio)
        self.enhanced_button.setEnabled(bool(attempt and attempt.has_enhanced))
        self.save_button.setEnabled(has_audio)
        self.separate_button.setEnabled(bool(self.query()))
        if has_audio and self._region_seconds is None and not self._use_region:
            self.region_label.setText("Whole event")
