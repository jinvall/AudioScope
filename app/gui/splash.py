"""Splash screen: start the stream, then press any key.

The Android sender gives up after a fixed number of connection attempts and
never reconnects on its own, so whoever starts the system has to start the
stream too. Making that a question of ordering - and a race the operator
inadvertently loses - is exactly the wrong design.

So the application asks for it explicitly instead. The splash appears first and
states what to do on the phone; the operator starts the stream whenever they
like and presses any key when audio is arriving. The review window then opens
onto a database that is already being written.

Deliberately *not* done here: starting the phone app over adb. That would make
this system depend on a USB or wireless-debugging link, and those links time
out on their own.
"""

from __future__ import annotations

import os
from typing import Optional

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QVBoxLayout,
    QWidget,
)

#: The mark the desktop launcher uses, so the splash and the icon agree.
LOGO = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "srp-css-theme-pack", "assets", "lary.png",
)


class SplashWindow(QWidget):
    """Instructions, the mark, and a wait for any key.

    Dismissed by any key press or a click, so the operator confirms with the
    gesture they have already made - reaching for the keyboard - rather than
    having to find a button first.
    """

    dismissed = pyqtSignal()

    def __init__(self, theme, db_path: str = "events.db",
                 source: str = "network", port: int = 8190,
                 capture_log: Optional[str] = None, parent=None):
        super().__init__(parent)
        self._theme = theme
        self._texts: list = []
        self._dismissed = False
        # Where the capture process reports what it has received.  The counter
        # belongs to that process, so the number is read from its log rather
        # than guessed at from here.
        self._capture_log = capture_log
        self.setWindowTitle("Audio Microscope")
        self.setFixedSize(760, 520)
        self.setFocusPolicy(Qt.StrongFocus)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(36, 28, 36, 28)
        layout.setSpacing(16)

        header = QHBoxLayout()
        header.setSpacing(16)
        logo = QLabel()
        pixmap = QPixmap(LOGO)
        if not pixmap.isNull():
            logo.setPixmap(
                pixmap.scaled(
                    72, 72, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
            )
        header.addWidget(logo)

        titles = QVBoxLayout()
        titles.setSpacing(2)
        name = QLabel("Audio Microscope")
        name.setObjectName("Title")
        titles.addWidget(name)
        subtitle = QLabel("Acoustic event review")
        subtitle.setObjectName("Muted")
        titles.addWidget(subtitle)
        header.addLayout(titles, 1)
        layout.addLayout(header)

        layout.addWidget(self._rule())

        layout.addWidget(self._step(
            "1",
            "Open the streaming app on your phone and press Start.",
            f"It connects to this machine on port {port} and sends audio "
            "continuously.",
        ))
        layout.addWidget(self._step(
            "2",
            "Wait until audio is arriving.",
            "The window below reports how much has been received.",
        ))
        layout.addWidget(self._step(
            "3",
            "Then press any key.",
            "The review window opens and fills as events are detected.",
        ))

        layout.addStretch(1)

        self._status = QLabel("waiting for the audio stream")
        self._status.setObjectName("Heading")
        layout.addWidget(self._status)

        self._detail = QLabel(f"database: {os.path.abspath(db_path)}")
        self._detail.setObjectName("Subtle")
        layout.addWidget(self._detail)

        self._capture_note = QLabel("")
        self._capture_note.setObjectName("Subtle")
        self._capture_note.setVisible(False)
        layout.addWidget(self._capture_note)
        if not capture_log:
            self._capture_note.setText("capture is not running")
            self._capture_note.setVisible(True)

        prompt = QLabel("press any key, or click anywhere, to continue")
        prompt.setObjectName("Muted")
        prompt.setAlignment(Qt.AlignCenter)
        layout.addWidget(prompt)

        self.setStyleSheet(self._style())

    # ------------------------------------------------------------------
    def _rule(self) -> QFrame:
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(
            f"color: {self._theme.color('border', '#3d2d61')};"
        )
        return line

    def _step(self, number: str, title: str, detail: str) -> QWidget:
        self._texts.append(f"{title} {detail}")
        row = QFrame()
        row.setObjectName("Card")
        layout = QHBoxLayout(row)
        layout.setContentsMargins(14, 10, 14, 10)
        layout.setSpacing(14)

        badge = QLabel(number)
        badge.setObjectName("Title")
        badge.setFixedWidth(18)
        badge.setAlignment(Qt.AlignTop | Qt.AlignHCenter)
        layout.addWidget(badge)

        text = QVBoxLayout()
        text.setSpacing(2)
        head = QLabel(title)
        head.setObjectName("Heading")
        head.setWordWrap(True)
        text.addWidget(head)
        tail = QLabel(detail)
        tail.setObjectName("Subtle")
        tail.setWordWrap(True)
        text.addWidget(tail)
        layout.addLayout(text, 1)
        return row

    def _style(self) -> str:
        c = self._theme.color
        return f"""
        QWidget {{ background-color: {c('bg', '#0f0a18')};
                   color: {c('text', '#f3f3f7')}; }}
        QFrame#Card {{ background-color: {c('surface', '#1a1230')};
                       border: 1px solid {c('border', '#3d2d61')};
                       border-radius: {self._theme.px('radius-md', 12)}px; }}
        """

    # ------------------------------------------------------------------
    def poll_capture_log(self) -> tuple:
        """Read the real receive counter out of the capture process's log.

        Capture runs in a separate process, so its counters are not visible from
        here.  Its log is the honest channel: it reports the seconds received
        for the connected client every few seconds.  Returns
        ``(seconds, connected)``, and ``(0.0, False)`` when nothing has arrived
        yet or there is nothing to read.
        """
        path = self._capture_log
        if not path or not os.path.exists(path):
            return 0.0, False
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as handle:
                lines = handle.readlines()[-80:]
        except OSError:
            return 0.0, False
        seconds, connected = 0.0, False
        for line in lines:
            if "client connected" in line:
                connected = True
            marker = "s received"
            if marker in line:
                head, _, tail = line.partition(marker)
                digits = ""
                for char in reversed(head):
                    if char.isdigit() or char == ".":
                        digits = char + digits
                    else:
                        break
                try:
                    seconds = max(seconds, float(digits))
                except ValueError:
                    pass
        return seconds, connected

    def set_stream_state(self, received_seconds: float,
                         connected: bool) -> None:
        """Report whether audio is actually arriving.

        The operator is being asked to press a key at the right moment, so the
        screen has to tell them whether the moment has arrived rather than
        leaving them to guess.  The number is the real seconds the capture
        process reports receiving, never a prediction or a substitute.
        """
        if received_seconds > 0.5:
            self._status.setText(
                f"receiving audio - {received_seconds:.1f} s so far"
            )
            self._status.setStyleSheet(
                f"color: {self._theme.color('primary', '#12f012')};"
            )
        elif connected:
            self._status.setText("connected - waiting for audio")
            self._status.setStyleSheet(
                f"color: {self._theme.color('warning', '#ffb020')};"
            )
        else:
            self._status.setText("waiting for the audio stream")
            self._status.setStyleSheet(
                f"color: {self._theme.color('text-muted', '#c8c9d4')};"
            )

    def _step_texts(self) -> list:
        """The instruction lines, for tests and for future accessibility."""
        return list(self._texts)

    @property
    def was_dismissed(self) -> bool:
        return self._dismissed

    def dismiss(self) -> None:
        if self._dismissed:
            return
        self._dismissed = True
        self.dismissed.emit()
        self.close()

    # ------------------------------------------------------------------
    def keyPressEvent(self, event):
        # Any key, including Escape: the point is a deliberate confirm.
        self.dismiss()

    def mousePressEvent(self, event):
        self.dismiss()

    def showEvent(self, event):
        super().showEvent(event)
        self.raise_()
        self.activateWindow()
        self.setFocus()
