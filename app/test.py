"""``python -m app.test`` - drive the pipeline from a WAV file.

docs/AUDIO_PIPELINE.md section 13 / AGENTS.md section 23: the same analysis
path must work from a file so development is repeatable without a live
microphone.

This module runs the *real* pipeline (:class:`app.pipeline.AudioPipeline`)
with a :class:`~app.audio.recorder.FileSource` in place of the device, so what
is verified here is the code that runs in production, not a test-only copy.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from .audio.recorder import FileSource
from .config import AppConfig
from .pipeline import AudioPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.test",
        description=(
            "Run the Audio Microscope audio pipeline over a WAV file. "
            "Exercises capture, ring buffer, and recording without a microphone."
        ),
    )
    parser.add_argument("input", help="input WAV file")
    parser.add_argument(
        "-o", "--output",
        default="test_output",
        help="directory for the recording chunks (default: test_output)",
    )
    parser.add_argument(
        "-c", "--config", default=None,
        help="path to a JSON configuration file",
    )
    parser.add_argument(
        "--seconds", type=float, default=None,
        help="stop after this much audio instead of reading the whole file",
    )
    parser.add_argument(
        "--block-size", type=int, default=None,
        help="capture block size in frames",
    )
    parser.add_argument(
        "--chunk-seconds", type=float, default=None,
        help="recording chunk length in seconds",
    )
    parser.add_argument(
        "--realtime", action="store_true",
        help="pace the file read at wall-clock speed",
    )
    parser.add_argument(
        "--no-record", action="store_true",
        help="do not write recording chunks",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="print the run summary as JSON",
    )
    return parser


def _load_config(args: argparse.Namespace) -> AppConfig:
    config = AppConfig.load(args.config) if args.config else AppConfig()
    if args.block_size:
        config.audio.block_size = args.block_size
    if args.chunk_seconds:
        config.record.chunk_seconds = args.chunk_seconds
    return config.validate()


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if not os.path.exists(args.input):
        print(f"error: no such file: {args.input}", file=sys.stderr)
        return 2

    try:
        config = _load_config(args)
    except Exception as exc:
        print(f"error: invalid configuration: {exc}", file=sys.stderr)
        return 2

    try:
        source = FileSource(args.input, config, realtime=args.realtime)
    except Exception as exc:
        print(f"error: cannot read {args.input}: {exc}", file=sys.stderr)
        return 2

    limit_frames = None
    if args.seconds:
        limit_frames = int(args.seconds * source.native_sample_rate)

    print(f"input        : {args.input}")
    print(f"format       : {source.native_sample_rate} Hz, "
          f"{source.duration_seconds:.2f} s")
    print(f"internal     : {config.sample_rate} Hz mono float32")
    print(f"ring buffer  : {config.buffer.seconds:g} s "
          f"({config.capacity_frames} frames)")
    print(f"recording to : {os.path.abspath(args.output)}"
          f"{'' if not args.no_record else ' (disabled)'}")
    print("-" * 60)

    # Cap the read so --seconds works.
    if limit_frames is not None:
        original = source.read

        def limited(frames=None):
            remaining = limit_frames - source._position
            if remaining <= 0:
                return None
            return original(min(frames or remaining, remaining))

        source.read = limited  # type: ignore[method-assign]

    started = time.monotonic()
    pipeline = AudioPipeline(
        config, source, record=not args.no_record, record_directory=args.output
    )
    pipeline.start()
    pipeline.wait()
    stats = pipeline.stop()
    elapsed = time.monotonic() - started

    summary = pipeline.summary()
    if args.json:
        print(json.dumps(summary, indent=2))
    else:
        rec = summary.get("recorder")
        print(f"blocks       : {stats.blocks}")
        print(f"audio        : {stats.audio_seconds:.2f} s "
              f"({stats.frames} frames)")
        print(f"wall clock   : {elapsed:.2f} s")
        print(f"speed        : {stats.audio_seconds / elapsed:.1f}x realtime"
              if elapsed > 0 else "speed        : n/a")
        print(f"ring buffer  : {summary['ring_frames_held']} frames held, "
              f"{summary['ring_overruns']} overrun frames")
        if rec:
            print(f"chunks       : {rec['chunk_count']} -> "
                  f"{os.path.abspath(args.output)}")
            for path in rec["chunks_written"]:
                size = os.path.getsize(path) if os.path.exists(path) else 0
                print(f"  {path}  ({size} bytes)")
            if not rec["is_contiguous"]:
                print(f"  WARNING: recording is not contiguous - "
                      f"{rec['blocks_dropped']} blocks / "
                      f"{rec['frames_dropped']} frames were dropped")
            for error in rec["errors"]:
                print(f"  ERROR: {error}")
        if stats.errors:
            print("pipeline errors:")
            for error in stats.errors:
                print(f"  {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
