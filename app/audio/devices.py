"""Input device discovery.

Phase 1 task 1 (tasks/PHASE_01_AUDIO.md): list input devices, identify the
default, and allow selection by name or index.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import sounddevice as sd


class AudioDeviceError(RuntimeError):
    """Raised when no usable input device can be found or opened."""


@dataclass(frozen=True)
class DeviceInfo:
    """An input device PortAudio can open."""

    index: int
    name: str
    max_input_channels: int
    default_sample_rate: float
    is_default: bool

    @property
    def supports_mono(self) -> bool:
        return self.max_input_channels >= 1

    def describe(self) -> str:
        return (
            f"[{self.index}] {self.name}  "
            f"in={self.max_input_channels}  "
            f"{self.default_sample_rate:.0f} Hz"
            + ("  (default)" if self.is_default else "")
        )

    def __str__(self) -> str:
        return self.describe()


def list_input_devices() -> list[DeviceInfo]:
    """Every device PortAudio reports an input for.

    PortAudio lists ALSA, PulseAudio and JACK devices; some advertise 128
    channels and are really "route everything" devices.  All are returned -
    choosing between them is the caller's job, and hiding them would make the
    device list a lie.
    """
    default_index = _default_input_index()
    devices: list[DeviceInfo] = []
    for index, entry in enumerate(sd.query_devices()):
        if int(entry["max_input_channels"]) <= 0:
            continue
        devices.append(
            DeviceInfo(
                index=index,
                name=str(entry["name"]),
                max_input_channels=int(entry["max_input_channels"]),
                default_sample_rate=float(entry["default_samplerate"]),
                is_default=index == default_index,
            )
        )
    return devices


def _default_input_index() -> Optional[int]:
    try:
        index, _ = sd.default.device[0]
        return None if index is None else int(index)
    except Exception:  # pragma: no cover - depends on host audio config
        return None


def default_input_device() -> DeviceInfo:
    """The system's default input, as PortAudio reports it."""
    for device in list_input_devices():
        if device.is_default:
            return device
    devices = list_input_devices()
    if not devices:
        raise AudioDeviceError("no input devices found")
    return devices[0]


def resolve_device(selector: str | int | None) -> DeviceInfo:
    """Resolve a device selector to a concrete device.

    ``selector`` may be None (system default), an integer index, or a name.
    Names are matched case-insensitively, first by exact match then by
    substring, so ``"hw:0,0"`` or ``"ALC897"`` both work.
    """
    devices = list_input_devices()
    if not devices:
        raise AudioDeviceError(
            "no input devices available - is an audio device connected?"
        )

    if selector is None:
        for device in devices:
            if device.is_default:
                return device
        return devices[0]

    if isinstance(selector, int):
        for device in devices:
            if device.index == selector:
                return device
        raise AudioDeviceError(
            f"input device index {selector} not found; "
            f"available: {[d.index for d in devices]}"
        )

    text = str(selector).strip()
    if not text:
        for device in devices:
            if device.is_default:
                return device
        return devices[0]

    # Numeric string: treat as an index.
    if re.fullmatch(r"-?\d+", text):
        return resolve_device(int(text))

    lowered = text.lower()
    for device in devices:
        if device.name.lower() == lowered:
            return device
    matches = [d for d in devices if lowered in d.name.lower()]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        names = ", ".join(f"{d.index}:{d.name}" for d in matches)
        raise AudioDeviceError(
            f"device name '{selector}' is ambiguous; candidates: {names}"
        )
    raise AudioDeviceError(
        f"no input device matching '{selector}'; available:\n  "
        + "\n  ".join(d.describe() for d in devices)
    )


def can_open(device_index: int, sample_rate: int, channels: int = 1) -> bool:
    """Whether PortAudio can open this device at the requested settings.

    Used to decide whether the internal 48 kHz rate can be honoured directly
    or whether a conversion is needed (docs/AUDIO_PIPELINE.md section 1).
    """
    try:
        sd.check_input_settings(
            device=device_index,
            channels=channels,
            dtype="float32",
            samplerate=sample_rate,
        )
        return True
    except Exception:
        return False
