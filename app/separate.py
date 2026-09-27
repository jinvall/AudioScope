"""``python -m app.separate`` - query-based source separation.

Phase 4 (tasks/PHASE_04_SEPARATION.md).  This replaces the stub that exited 3
without writing anything.

    ./venv/bin/python -m app.separate input.wav --query "a whisper" -o out.wav
    ./venv/bin/python -m app.separate input.wav -q footsteps --event event_000042

The separation itself is AudioSep's: a CLAP text encoder turns the query into
an embedding, and a ResUNet30 estimates the matching waveform from the
mixture.  This module owns everything around it - validation of the request,
region selection, output validation, optional enhancement, metadata, and the
metrics that make CPU cost visible.

What this deliberately does *not* do is filter.  A bandpass, a spectral gate
or a denoiser is not source separation (docs/SOURCE_SEPARATION.md section 16),
so if the model is unavailable this exits non-zero and writes no audio at all,
rather than producing something that looks like a result.

Exit codes
    0  separation written
    2  bad usage or invalid configuration
    3  the model is not available (see --preflight for why)
    4  separation did not produce a usable result
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Optional

from .config import AppConfig
from .separation.jobs import JobOrigin, SeparationJob, SeparationResult
from .separation.separator import build_separator
from .separation.store import SeparationStore

EXIT_OK = 0
EXIT_USAGE = 2
EXIT_UNAVAILABLE = 3
EXIT_FAILED = 4

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.separate",
        description="Isolate a queried sound from a recording using AudioSep "
                    "(a real separation model, CPU only).",
    )
    parser.add_argument("input", nargs="?", help="input WAV file")
    parser.add_argument(
        "-q", "--query",
        help='what to isolate, e.g. "a whisper", "footsteps", "scraping"',
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="output WAV path (default: isolated.wav, or inside the event's "
             "separation directory when --event is used)",
    )
    parser.add_argument(
        "--start", type=float, default=None,
        help="region start in seconds (default: whole file)",
    )
    parser.add_argument(
        "--duration", type=float, default=None,
        help="region length in seconds (default: to end of file)",
    )
    parser.add_argument(
        "--event", default=None, metavar="EVENT_ID",
        help="separate this event instead of writing a single file: the output "
             "goes to a fresh separation_NNN directory beside the event, so "
             "repeated attempts never overwrite each other",
    )
    parser.add_argument(
        "--events-root", default=None,
        help=f"event tree to search (default: {os.path.join(PROJECT_ROOT, 'events')})",
    )
    parser.add_argument(
        "--no-enhance", action="store_true",
        help="write only the raw model output, with no enhanced.wav",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="print the result metadata as JSON on stdout",
    )
    parser.add_argument(
        "--preflight", action="store_true",
        help="report whether separation can run, and why not if it cannot",
    )
    parser.add_argument(
        "-c", "--config", default=None, help="JSON config file",
    )
    return parser


def _fail(message: str, code: int) -> int:
    print(f"error: {message}", file=sys.stderr)
    return code


def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    # Validate the request first, so the error is about the request rather
    # than about the model.
    try:
        config = AppConfig.load(args.config) if args.config else AppConfig()
    except Exception as exc:
        return _fail(f"invalid configuration: {exc}", EXIT_USAGE)

    separation_config = config.separation
    if args.no_enhance:
        separation_config.enable_enhancement = False

    # ------------------------------------------------------------------
    # Preflight: report the environment without doing any work.
    # ------------------------------------------------------------------
    separator = build_separator(separation_config, PROJECT_ROOT)
    if args.preflight:
        reason = separator.unavailable_reason()
        if reason:
            print("source separation is NOT available:", file=sys.stderr)
            print(f"  {reason}", file=sys.stderr)
            return EXIT_UNAVAILABLE
        print("source separation is available.")
        print(f"  model:      {separation_config.checkpoint_path}")
        print(f"  model dir:  {separation_config.model_dir}")
        print(f"  interpreter:{separation_config.venv_python}")
        print(f"  sample rate:{separation_config.model_sample_rate} Hz")
        return EXIT_OK

    if not args.input:
        return _fail("an input WAV is required (or use --preflight)", EXIT_USAGE)
    if not args.query:
        return _fail("--query is required", EXIT_USAGE)
    if not args.query.strip():
        return _fail("--query must not be empty", EXIT_USAGE)
    if args.start is not None and args.start < 0:
        return _fail("--start must not be negative", EXIT_USAGE)
    if args.duration is not None and args.duration <= 0:
        return _fail("--duration must be positive", EXIT_USAGE)
    if args.event and args.output:
        return _fail(
            "--event and --output are mutually exclusive: --event writes into "
            "the event's own separation_NNN directory",
            EXIT_USAGE,
        )
    if not os.path.exists(args.input):
        return _fail(f"input file does not exist: {args.input}", EXIT_USAGE)

    reason = separator.unavailable_reason()
    if reason:
        print("error: source separation is not available:", file=sys.stderr)
        print(f"  {reason}", file=sys.stderr)
        print("", file=sys.stderr)
        print("  Check with: python -m app.separate --preflight", file=sys.stderr)
        return EXIT_UNAVAILABLE

    # ------------------------------------------------------------------
    # Where the output goes.
    # ------------------------------------------------------------------
    store = SeparationStore(
        root=os.path.abspath(
            args.events_root or os.path.join(PROJECT_ROOT, "events")
        )
    )
    output_directory = ""
    event_id = args.event or os.path.splitext(os.path.basename(args.input))[0]

    if args.event:
        day = store.find_day(args.event)
        if day is None:
            return _fail(
                f"event {args.event!r} was not found under {store.root}",
                EXIT_USAGE,
            )
        output_directory = store.event_directory(day, args.event)
        origin = JobOrigin.EVENT
    else:
        origin = (
            JobOrigin.MANUAL_REGION
            if (args.start is not None or args.duration is not None)
            else JobOrigin.EVENT
        )

    try:
        job = SeparationJob(
            event_id=event_id,
            source_path=os.path.abspath(args.input),
            query=args.query.strip(),
            output_directory=output_directory,
            output_path=os.path.abspath(args.output) if args.output else None,
            model_name="AudioSep",
            start_seconds=args.start,
            duration_seconds=args.duration,
            origin=origin,
        )
    except ValueError as exc:
        return _fail(str(exc), EXIT_USAGE)

    # ------------------------------------------------------------------
    # Run it.  The model load is the long part on a cold start, so say so
    # rather than letting the terminal look hung.
    # ------------------------------------------------------------------
    if not separator.available():
        return _fail(str(separator.unavailable_reason()), EXIT_UNAVAILABLE)

    print(
        f"separating: {job.query!r} from {os.path.basename(job.source_path)}",
        file=sys.stderr,
    )
    print(
        "the first run loads the model, which takes about 35 seconds",
        file=sys.stderr,
    )
    started = time.time()
    result: SeparationResult
    try:
        result = separator.separate(job)
    finally:
        close = getattr(separator, "close", None)
        if callable(close):
            close()
    wall = time.time() - started

    if not result.ok:
        return _report_failure(result, wall, args.json)

    _report_success(result, wall, args.json)
    return EXIT_OK


def _report_success(
    result: SeparationResult, wall: float, as_json: bool
) -> None:
    if as_json:
        print(json.dumps(result.to_metadata(), indent=2, sort_keys=True))
        return

    job = result.job
    print("", file=sys.stderr)
    print(f"  query     : {job.query}", file=sys.stderr)
    print(f"  model     : {job.model_name}", file=sys.stderr)
    print(f"  input     : {result.input_seconds:.2f} s", file=sys.stderr)
    print(f"  processed : {result.processing_seconds:.2f} s "
          f"(wall {wall:.2f} s)", file=sys.stderr)
    if result.realtime_ratio:
        print(
            f"  ratio     : {result.realtime_ratio:.2f}x realtime",
            file=sys.stderr,
        )
    print(f"  isolated  : {result.isolated_path}", file=sys.stderr)
    if result.enhanced_path:
        gain = result.enhancement.get("gain")
        gain_text = f" (gain {gain:.3f})" if isinstance(gain, float) else ""
        print(f"  enhanced  : {result.enhanced_path}{gain_text}", file=sys.stderr)
    if result.metadata_path:
        print(f"  metadata  : {result.metadata_path}", file=sys.stderr)
    for warning in result.warnings:
        print(f"  note      : {warning}", file=sys.stderr)


def _report_failure(
    result: SeparationResult, wall: float, as_json: bool
) -> int:
    """Report a failure honestly and name a single exit code for it.

    Once availability is confirmed, every failure is the same thing to a
    caller: no usable result.  Whether the model crashed, timed out, or
    returned output that failed validation, the answer is "there is nothing
    to listen to", and inventing a distinction here would only make the exit
    codes harder to act on.
    """
    if as_json:
        print(json.dumps(result.to_metadata(), indent=2, sort_keys=True))
    else:
        print("", file=sys.stderr)
        print(f"  query     : {result.job.query}", file=sys.stderr)
        print(f"  elapsed   : {wall:.2f} s", file=sys.stderr)
        if result.error:
            print(f"  failed    : {result.error}", file=sys.stderr)
        for warning in result.warnings:
            print(f"  note      : {warning}", file=sys.stderr)
        print("", file=sys.stderr)
        print(
            "  The original recording is untouched.  No audio was written.",
            file=sys.stderr,
        )
    return EXIT_FAILED


if __name__ == "__main__":
    raise SystemExit(main())
