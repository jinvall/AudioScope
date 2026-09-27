#!/usr/bin/env python3
"""Reference client for the Audio Microscope / AMP network source.

Python equivalent of the Android sender.  It exists so the wire format can be
exercised from a workstation and so there is one concrete implementation to
port to Kotlin.

The format matches ``/home/jason/amp/server/audio_receiver.py`` and
``app/audio/network.py``:

1. connect to TCP port 8190
2. optionally send one newline-terminated JSON config object
3. stream **raw, unframed** signed 16-bit little-endian PCM at 44 100 Hz, mono

The server sends nothing back.  Close the connection to end the stream.

Examples
--------
    python tools/android_stream_client.py --server 10.0.0.147 --file input.wav
    python tools/android_stream_client.py --server 10.0.0.147 --tone 440 --realtime
    python tools/android_stream_client.py --server 10.0.0.147 --file in.wav \\
        --config '{"amplification":1.0}'
"""

from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time

import numpy as np

DEFAULT_PORT = 8190
WIRE_SAMPLE_RATE = 44100
WIRE_CHANNELS = 1
BYTES_PER_SECOND = WIRE_SAMPLE_RATE * WIRE_CHANNELS * 2


def to_s16le(samples: np.ndarray) -> bytes:
    """Convert float samples in [-1, 1] to signed 16-bit little-endian PCM."""
    clipped = np.clip(np.asarray(samples, dtype=np.float32), -1.0, 1.0)
    scaled = np.round(clipped * 32767.0).astype("<i2")
    return scaled.tobytes()


def load_audio(path: str) -> tuple[np.ndarray, int]:
    """Load a WAV as mono float32, returning (samples, source_rate)."""
    sys.path.insert(
        0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    from app.audio.resample import StreamingResampler, ratio_for
    from app.audio.wavio import probe_wav, read_wav

    info = probe_wav(path)
    data = read_wav(path, 0, None, mono=True)
    if info.sample_rate != WIRE_SAMPLE_RATE:
        up, down = ratio_for(info.sample_rate, WIRE_SAMPLE_RATE)
        data = StreamingResampler(up, down).process(data)
    return data, WIRE_SAMPLE_RATE


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--server", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--file", default=None, help="WAV file to stream")
    parser.add_argument("--tone", type=float, default=None,
                        help="stream a sine at this frequency instead")
    parser.add_argument("--amplitude", type=float, default=0.3)
    parser.add_argument("--seconds", type=float, default=10.0,
                        help="length of a generated tone")
    parser.add_argument("--rate", type=int, default=WIRE_SAMPLE_RATE,
                        help="rate for a generated tone")
    parser.add_argument("--config", default=None,
                        help='JSON config line, e.g. \'{"amplification":1.0}\'')
    parser.add_argument("--chunk", type=int, default=8192,
                        help="samples per write")
    parser.add_argument("--realtime", action="store_true",
                        help="pace the stream at wall-clock speed")
    args = parser.parse_args(argv)

    if args.file:
        samples, rate = load_audio(args.file)
        print(f"streaming {args.file}: {samples.size} samples -> {rate} Hz s16le")
    elif args.tone is not None:
        rate = args.rate
        count = int(args.seconds * rate)
        t = np.arange(count) / rate
        samples = (args.amplitude * np.sin(2 * np.pi * args.tone * t)).astype(
            np.float32
        )
        print(f"streaming {args.tone} Hz tone: {count} samples @ {rate} Hz s16le")
    else:
        parser.error("one of --file or --tone is required")

    try:
        sock = socket.create_connection((args.server, args.port), timeout=10)
    except OSError as exc:
        print(f"error: cannot connect to {args.server}:{args.port}: {exc}",
              file=sys.stderr)
        return 1

    with sock:
        sock.settimeout(10)
        if args.config is not None:
            # Must be a single line: the newline is the frame terminator.
            line = json.dumps(json.loads(args.config), separators=(",", ":"))
            if "\n" in line:
                print("error: --config must not contain a newline", file=sys.stderr)
                return 2
            sock.sendall(line.encode("utf-8") + b"\n")
            print(f"sent config: {line}")

        sent = 0
        started = time.monotonic()
        index = 0
        while index < samples.size:
            block = samples[index:index + args.chunk]
            sock.sendall(to_s16le(block))
            index += block.size
            sent += block.size
            if args.realtime:
                drift = (sent / rate) - (time.monotonic() - started)
                if drift > 0:
                    time.sleep(drift)

    print(f"sent {sent} samples ({sent / rate:.2f} s), stream complete")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
