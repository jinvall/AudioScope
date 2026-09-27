"""Shared fixtures."""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import AppConfig  # noqa: E402
from app.audio.wavio import probe_wav, write_wav_atomic  # noqa: E402


@pytest.fixture
def config() -> AppConfig:
    """A validated default configuration."""
    return AppConfig().validate()


@pytest.fixture
def tone():
    """A mono float32 sine at 1 kHz."""

    def _make(freq: float = 1000.0, seconds: float = 1.0, rate: int = 48000,
              amplitude: float = 0.5, phase: float = 0.0) -> np.ndarray:
        t = np.arange(int(seconds * rate)) / rate
        return (amplitude * np.sin(2 * np.pi * freq * t + phase)).astype(
            np.float32
        )

    return _make


@pytest.fixture
def silence():
    def _make(seconds: float = 1.0, rate: int = 48000) -> np.ndarray:
        return np.zeros(int(seconds * rate), dtype=np.float32)

    return _make


@pytest.fixture
def wav_file(tmp_path, tone):
    """A real WAV file on disk, for the file-source path."""
    path = str(tmp_path / "input.wav")
    write_wav_atomic(path, tone(440.0, 0.5), 48000)
    return path
