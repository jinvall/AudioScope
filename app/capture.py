"""``python -m app.capture`` - live capture from a device or the network.

Phase 1 acceptance (tasks/PHASE_01_AUDIO.md) is::

    microphone -> capture -> ring buffer -> recording -> playback

This command runs that chain for real, and is what ``./run.sh`` invokes.

Two input modes:

* ``--source device``  - a local input via PortAudio (default)
* ``--source network`` - a TCP stream from an Android device on port 8090,
  which also reserves ports 8060-8064 for the session
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from typing import Optional

from .audio.capture import DeviceCapture
from .audio.devices import AudioDeviceError, list_input_devices, resolve_device
from .audio.network import (
    RESERVED_PORTS,
    STREAM_PORT,
    WIRE_SAMPLE_RATE,
    NetworkError,
    NetworkSource,
)
from .config import AppConfig
from .pipeline import AudioPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.capture",
        description="Live audio capture -> ring buffer -> WAV recording.",
    )
    parser.add_argument(
        "-s", "--source", choices=("device", "network"), default="network",
        help="where the audio comes from. Defaults to network - the phone on "
             f"port {STREAM_PORT} - because that is this project's source. The "
             "local input is a host microphone, which on this machine is the "
             "HDA Intel PCH ALC897 analog path and carries audio only while "
             "scrcpy is running; with nothing playing it delivers silence, and "
             "a capture that silently records nothing looks like a quiet room. "
             "Use --source device deliberately, not by default.",
    )
    parser.add_argument(
        "-d", "--device", default=None,
        help="input device index, name, or substring; omit for system default",
    )
    parser.add_argument(
        "-o", "--output", default="recordings",
        help="directory for recording chunks (default: recordings)",
    )
    parser.add_argument("-c", "--config", default=None, help="JSON config file")
    parser.add_argument(
        "--seconds", type=float, default=None,
        help="stop automatically after this many seconds",
    )
    parser.add_argument(
        "--port", type=int, default=STREAM_PORT,
        help=f"network stream port (default: {STREAM_PORT})",
    )
    parser.add_argument(
        "--reserve", default=",".join(str(p) for p in RESERVED_PORTS),
        help="ports to reserve while the stream is active "
             f"(default: {','.join(str(p) for p in RESERVED_PORTS)})",
    )
    parser.add_argument(
        "--no-reserve", action="store_true",
        help="do not reserve the auxiliary ports",
    )
    parser.add_argument("--list-devices", action="store_true",
                        help="list input devices and exit")
    parser.add_argument(
        "--no-record", action="store_true", help="do not write recording chunks"
    )
    parser.add_argument(
        "--no-analysis", action="store_true",
        help="skip the analysis stage (it is on by default; it never blocks "
             "capture)",
    )
    parser.add_argument(
        "--no-detection", action="store_true",
        help="skip event detection and storage (on by default)",
    )
    parser.add_argument(
        "--events", default="events",
        help="directory for stored events (default: events)",
    )
    parser.add_argument(
        "--db", default=os.environ.get("AUDIOMICROSCOPE_DB", "events.db"),
        help="SQLite index of detected events, for review and annotation "
             "(default: events.db; empty string disables indexing)",
    )
    parser.add_argument(
        "--wait-client", type=float, default=None,
        help="network mode: pause this many seconds to report on clients; "
             "never gates startup",
    )
    parser.add_argument(
        "--status", action="store_true",
        help="print client/config status while streaming",
    )
    return parser


def _load_config(args: argparse.Namespace) -> AppConfig:
    config = AppConfig.load(args.config) if args.config else AppConfig()
    if args.device is not None and args.source == "device":
        config.audio.device = args.device
    config.record.directory = args.output
    return config.validate()


def _print_devices() -> int:
    try:
        devices = list_input_devices()
    except Exception as exc:
        print(f"error: cannot enumerate devices: {exc}", file=sys.stderr)
        return 1
    if not devices:
        print("no input devices found")
        return 1
    print("input devices:")
    for device in devices:
        print("  " + device.describe())
    return 0


def _run_device(config: AppConfig, args: argparse.Namespace) -> int:
    try:
        source = DeviceCapture(config, on_error=lambda m: print(f"  ! {m}"))
    except AudioDeviceError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"device       : {source.output_device}")
    print(f"native rate  : {source.native_sample_rate} Hz")
    if source.is_converting:
        print(f"converting   : {source.native_sample_rate} Hz -> "
              f"{config.sample_rate} Hz (once, at the input boundary)")
    else:
        print(f"internal rate: {config.sample_rate} Hz (direct, no conversion)")
    return _drive(config, source, args)


def _run_network(config: AppConfig, args: argparse.Namespace) -> int:
    reserved = ()
    if args.reserve:
        reserved = tuple(int(p) for p in args.reserve.split(",") if p.strip())

    def status(message: str) -> None:
        print(f"  {message}")

    try:
        source = NetworkSource(
            config,
            port=args.port,
            reserve_ports=not args.no_reserve,
            reserved=reserved or RESERVED_PORTS,
            on_status=status,
        )
    except NetworkError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    print(f"stream port  : {args.port} (TCP)")
    if not args.no_reserve:
        print(f"reserving    : {', '.join(str(p) for p in (reserved or RESERVED_PORTS))}")
    print(f"wire format  : s16le {WIRE_SAMPLE_RATE} Hz mono (raw, unframed)")
    print(f"internal     : float32 {config.sample_rate} Hz mono")
    print("waiting for the Android client (it may connect at any time)...")

    try:
        source.start()
    except NetworkError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    # The server does not need a client in order to start.  It listens and
    # accepts whenever the phone turns up, so a missing client is a normal
    # idle state, not a startup failure.  --wait-client is a diagnostic that
    # reports on the state after a pause; it never gates startup.
    if args.wait_client:
        if source.wait_for_client(timeout=args.wait_client):
            for client in source.clients:
                print(f"streaming    : {client.address}")
                if client.config:
                    print(f"  config     : {json.dumps(client.config, sort_keys=True)}")
                    if client.amplification is not None:
                        print("  note       : 'amplification' is recorded, not "
                              "applied to the stored audio (apply it at "
                              "playback instead)")
        else:
            print("no client yet; still listening. The pipeline will start and "
                  "record nothing until audio arrives.")

    return _drive(config, source, args)


def _drive(config: AppConfig, source, args: argparse.Namespace) -> int:
    os.makedirs(args.output, exist_ok=True)
    print(f"recording to : {os.path.abspath(args.output)}")
    print(f"ring buffer  : {config.buffer.seconds:g} s "
          f"({config.capacity_frames} frames)")
    print("-" * 60)
    print("Ctrl-C to stop.")

    pipeline = AudioPipeline(
        config, source, record=not args.no_record, record_directory=args.output
    )
    worker = None
    if not args.no_analysis:
        worker = pipeline.enable_analysis()
        print(f"analysis     : {config.analysis.frame_ms:g} ms frame / "
              f"{config.analysis.hop_ms:g} ms hop "
              f"({config.analysis_frames_per_second:.0f} frames/s)")

    detection = None
    detected: list = []
    if not args.no_detection:
        detection = pipeline.enable_detection(
            on_event=detected.append,
            event_root=args.events,
            database=args.db or None,
        )
        det = config.detection
        print(f"detection    : onset {det.onset_threshold:g} / "
              f"continuation {det.continuation_threshold:g}, "
              f"release {det.release_timeout:g}s, "
              f"min {det.min_duration:g}s, max {det.max_duration:g}s")
        print(f"events       : {os.path.abspath(args.events)}/<date>/<event_id>/")

    stopping = {"flag": False}

    def handle_signal(signum, frame):  # pragma: no cover - signal path
        stopping["flag"] = True
        print("\nstopping...")

    try:
        signal.signal(signal.SIGINT, handle_signal)
        signal.signal(signal.SIGTERM, handle_signal)
    except ValueError:
        pass  # not on the main thread

    pipeline.start()
    deadline = time.monotonic() + args.seconds if args.seconds else None

    # Periodic status so the run is observable, not silent.
    next_report = time.monotonic() + 5.0
    try:
        while pipeline.running and not stopping["flag"]:
            time.sleep(0.1)
            now = time.monotonic()
            if deadline and now >= deadline:
                break
            if now >= next_report:
                next_report = now + 5.0
                if args.status:
                    for client in getattr(source, "clients", []):
                        print(f"    {client.describe()}")
                s = pipeline.stats
                rec = pipeline.recorder.stats if pipeline.recorder else None
                extra = ""
                if rec:
                    extra = (f", {len(rec.chunks_written)} chunk(s)"
                             f"{'' if rec.is_contiguous else ' NOT CONTIGUOUS'}")
                print(f"  {s.audio_seconds:8.1f}s audio  "
                      f"{s.realtime_ratio:5.2f}x realtime  "
                      f"ring={pipeline.ring.frames_available} frames{extra}")
    except KeyboardInterrupt:  # pragma: no cover
        # SIGINT can still arrive before the handler above is installed (during
        # imports), so this is the backstop that guarantees a clean shutdown.
        print("\nstopping...")

    stats = pipeline.stop()
    summary = pipeline.summary()
    print("-" * 60)
    print(f"captured     : {stats.audio_seconds:.1f} s "
          f"({stats.frames} frames in {stats.blocks} blocks)")
    overflows = getattr(source, "input_overflows", 0)
    if overflows:
        print(f"  WARNING: {overflows} input overflow(s) - PortAudio could not "
              f"service its buffer in time, so audio was dropped and the "
              f"recording is not contiguous")
    rec = summary.get("recorder")
    if rec:
        print(f"chunks       : {rec['chunk_count']} file(s) in "
              f"{os.path.abspath(args.output)}")
        for path in rec["chunks_written"]:
            print(f"  {path}")
        if not rec["is_contiguous"]:
            print(f"  WARNING: {rec['blocks_dropped']} block(s) / "
                  f"{rec['frames_dropped']} frame(s) dropped - the recording "
                  "is not contiguous")
        for error in rec["errors"]:
            print(f"  ERROR: {error}")
    if worker is not None:
        a = summary["analysis"]
        print(f"analysis     : {a['frames']} frames, "
              f"{a['mean_frame_ms']:.3f} ms/frame, "
              f"{a['realtime_ratio']:.4f}x realtime "
              f"({'within budget' if a['realtime_ratio'] < 1.0 else 'OVER BUDGET'})")
        print(f"noise floor  : {worker.noise_floor_db:.1f} dBFS")
        if not a["is_contiguous"]:
            print(f"  WARNING: analysis dropped {a['blocks_dropped']} block(s); "
                  f"{a['gap_frames']} frame(s) are marked as gapped")
        for error in a["errors"]:
            print(f"  analysis error: {error}")
    if detection is not None:
        d = summary["detection"]
        print(f"detection    : {d['events_detected']} event(s) detected, "
              f"{d['events_stored']} stored, "
              f"{d['mean_frame_ms']:.3f} ms/frame")
        for event in detected:
            print(f"  {event.summary()}")
            print(f"      evidence: {', '.join(event.evidence[:4]) or 'none'}")
            print(f"      query   : {event.separation_query}")
        writer = pipeline._writer
        if writer is not None:
            ps = writer.stats.to_dict()
            print(f"database    : {os.path.abspath(args.db)}  "
                  f"{ps['stored']} indexed, "
                  f"{ps['dropped_queue_full']} dropped from the index, "
                  f"max {ps['max_latency_ms']:.1f} ms write")
        if d["events_unwritable"]:
            print(f"  WARNING: {d['events_unwritable']} event(s) could not be "
                  f"written")
        for error in d["write_errors"][:5]:
            print(f"  detection error: {error}")
    for error in stats.errors:
        print(f"  pipeline error: {error}")
    return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    if args.list_devices:
        return _print_devices()

    try:
        config = _load_config(args)
    except Exception as exc:
        print(f"error: invalid configuration: {exc}", file=sys.stderr)
        return 2

    if args.source == "network":
        return _run_network(config, args)
    return _run_device(config, args)


if __name__ == "__main__":
    raise SystemExit(main())
