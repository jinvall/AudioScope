"""Lossless WAV reading and writing.

AGENTS.md section 20 requires WAV as the evidence format, and section 2.1
requires the original to be preserved unmodified.  Both push towards one
decision: **the canonical on-disk format is mono float32 WAV**, which stores
samples exactly, with no clipping and no dithering.

Writes are atomic (docs/DATA_AND_STORAGE.md section 9): data goes to a
temporary file in the same directory and is renamed into place only after a
successful close.  An interrupted write therefore leaves a ``.part`` file
rather than a truncated file that looks valid.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from typing import Optional

import numpy as np
import soundfile as sf

# Evidence is stored as 32-bit float so that nothing is clipped or quantised.
# PCM subtypes would be smaller but lossy for a monitoring tool.
EVIDENCE_SUBTYPE = "FLOAT"
EVIDENCE_FORMAT = "WAV"


@dataclass(frozen=True)
class WavInfo:
    """What a file actually contains, as read back from disk."""

    path: str
    sample_rate: int
    channels: int
    frames: int
    subtype: str
    duration: float

    @property
    def is_mono(self) -> bool:
        return self.channels == 1


# ----------------------------------------------------------------------
# Reading
# ----------------------------------------------------------------------
def read_wav(
    path: str,
    start: int = 0,
    frames: Optional[int] = None,
    mono: bool = True,
) -> np.ndarray:
    """Read a region of a WAV file as float32.

    ``start``/``frames`` are in frames of the *file's* sample rate.  With
    ``mono=True`` a multi-channel file is downmixed by averaging, which is the
    internal mono processing path from AGENTS.md section 6.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with sf.SoundFile(path, "r") as handle:
        if start < 0:
            raise ValueError("start must not be negative")
        if start > len(handle):
            raise ValueError(
                f"start {start} is past end of file ({len(handle)} frames)"
            )
        handle.seek(start)
        block = handle.read(
            frames if frames is not None else -1,
            dtype="float32",
            always_2d=True,
        )
    data = block[:, 0] if mono else block.T.copy()
    return np.ascontiguousarray(data, dtype=np.float32)


def read_wav_mono(path: str) -> np.ndarray:
    """Read an entire file as mono float32."""
    return read_wav(path, 0, None, mono=True)


def probe_wav(path: str) -> WavInfo:
    """Read a file's real format from its header."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with sf.SoundFile(path, "r") as handle:
        frames = len(handle)
        return WavInfo(
            path=os.path.abspath(path),
            sample_rate=int(handle.samplerate),
            channels=int(handle.channels),
            frames=int(frames),
            subtype=str(handle.subtype),
            duration=frames / handle.samplerate if handle.samplerate else 0.0,
        )


# ----------------------------------------------------------------------
# Writing
# ----------------------------------------------------------------------
def write_wav_atomic(
    path: str,
    samples: np.ndarray,
    sample_rate: int,
    subtype: str = EVIDENCE_SUBTYPE,
) -> str:
    """Write mono float samples to ``path`` atomically.

    Returns the final path.  The file is only renamed into place after the
    handle is closed cleanly, so a crash mid-write cannot leave a truncated
    file that later looks like valid evidence.
    """
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    data = np.asarray(samples)
    if data.ndim not in (1, 2):
        raise ValueError(
            f"expected mono (frames,) or (frames, channels) audio, got "
            f"shape {data.shape}"
        )
    # libsndfile's Python bindings take (frames, channels), which is also the
    # natural layout here, so no transposition is needed.
    data = np.ascontiguousarray(data, dtype=np.float32)
    channels = 1 if data.ndim == 1 else int(data.shape[1])

    directory = os.path.dirname(os.path.abspath(path))
    os.makedirs(directory, exist_ok=True)

    tmp_path = None
    try:
        fd, tmp_path = tempfile.mkstemp(
            dir=directory, prefix=os.path.basename(path) + ".", suffix=".part"
        )
        os.close(fd)
        with sf.SoundFile(
            tmp_path,
            "w",
            samplerate=sample_rate,
            channels=channels,
            subtype=subtype,
            format=EVIDENCE_FORMAT,
        ) as out:
            out.write(data)
        os.replace(tmp_path, path)
        tmp_path = None
    finally:
        if tmp_path is not None and os.path.exists(tmp_path):
            # The write failed; drop the partial file so no corrupt evidence
            # is left lying around.
            os.unlink(tmp_path)
    return path


class WavWriter:
    """Append-only WAV writer for the continuous recorder.

    Holds the file open across many blocks and rolls over to a new file every
    ``chunk_frames``.  Also writes atomically per chunk: each chunk is built
    in a temporary file and renamed, so a chunk on disk is always complete.
    """

    def __init__(
        self,
        directory: str,
        sample_rate: int,
        chunk_frames: int,
        name_template: str = "rec_{index:06d}.wav",
        start_index: int = 0,
    ) -> None:
        if chunk_frames <= 0:
            raise ValueError("chunk_frames must be positive")
        self.directory = os.path.abspath(directory)
        self.sample_rate = int(sample_rate)
        self.chunk_frames = int(chunk_frames)
        self.name_template = name_template
        self.index = int(start_index)
        os.makedirs(self.directory, exist_ok=True)

        self._buffer: list[np.ndarray] = []
        self._buffered = 0
        self._current: Optional[str] = None
        self.chunks_written: list[str] = []

    # ------------------------------------------------------------------
    @property
    def current_path(self) -> Optional[str]:
        return self._current

    @property
    def buffered_frames(self) -> int:
        return self._buffered

    def path_for(self, index: int) -> str:
        return os.path.join(self.directory, self.name_template.format(index=index))

    def append(self, samples: np.ndarray) -> Optional[str]:
        """Append mono samples, rolling over when the chunk is full.

        Returns the path of a completed chunk if this call finished one, else
        None.
        """
        block = np.asarray(samples, dtype=np.float32).ravel()
        if block.size == 0:
            return None
        self._buffer.append(block)
        self._buffered += block.size
        if self._buffered >= self.chunk_frames:
            return self._flush_chunk()
        return None

    # ------------------------------------------------------------------
    def _flush_chunk(self) -> str:
        data = np.concatenate(self._buffer) if self._buffer else np.zeros(0, np.float32)
        # Do not exceed the configured chunk length.
        data = data[: self.chunk_frames]
        path = self.path_for(self.index)
        write_wav_atomic(path, data, self.sample_rate)
        self.index += 1
        self.chunks_written.append(path)
        self._current = path

        # Carry the overshoot into the next chunk so no audio is dropped.
        leftover = data.size and data.size < self._buffered
        if leftover:
            tail = np.concatenate(self._buffer)[data.size:]
            self._buffer = [np.ascontiguousarray(tail, dtype=np.float32)]
            self._buffered = int(tail.size)
        else:
            self._buffer = []
            self._buffered = 0
        return path

    def close(self) -> Optional[str]:
        """Write whatever is buffered and return its path, if any."""
        if self._buffered == 0:
            return None
        return self._flush_chunk()

    def __enter__(self) -> "WavWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
