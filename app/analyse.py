"""``python -m app.analyse`` - run the analysis stage over a WAV file.

docs/AUDIO_PIPELINE.md section 13: file input must enter the same analysis path
as live input.  This runs the real :class:`~app.analysis.worker.AnalysisWorker`
through the real :class:`~app.pipeline.AudioPipeline`, so what is inspected here
is the code that runs in production.

Useful for checking the noise floor against a known environment, confirming the
feature set, and measuring the CPU cost the analysis stage actually adds.

Examples
--------
    python -m app.analyse input.wav
    python -m app.analyse input.wav --json
    python -m app.analyse input.wav --frames 20
    python -m app.analyse input.wav --seconds 5
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Optional

from .audio.recorder import FileSource
from .audio.wavio import probe_wav
from .config import AppConfig
from .pipeline import AudioPipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m app.analyse",
        description="Continuous acoustic analysis over a WAV file.",
    )
    parser.add_argument("input", help="input WAV file")
    parser.add_argument("-c", "--config", default=None, help="JSON config file")
    parser.add_argument(
        "--frames", type=int, default=8,
        help="how many sample analysis frames to print (default: 8)",
    )
    parser.add_argument(
        "--seconds", type=float, default=None,
        help="analyse only this much audio",
    )
    parser.add_argument(
        "--tail", action="store_true",
        help="print the last frames instead of the first",
    )
    parser.add_argument(
        "--bands", action="store_true",
        help="include per-band energy and per-band SNR in the printed frames",
    )
    parser.add_argument(
        "--spectrogram", default=None,
        help="write a spectrogram image (.pgm works anywhere; .png needs "
             "matplotlib)",
    )
    parser.add_argument(
        "--json", action="store_true",
        help="print everything as JSON",
    )
    return parser


def _load_config(args: argparse.Namespace) -> AppConfig:
    return AppConfig.load(args.config) if args.config else AppConfig().validate()


def _format_frame(index: int, result, show_bands: bool) -> list[str]:
    f = result.features
    n = result.floor
    lines = [
        f"  frame {index:>6}  t={result.audio_time:7.2f}s  "
        f"start={result.start_sample}",
        f"    level    rms {f.rms_db:7.1f} dBFS   peak {f.peak_db:7.1f} dBFS   "
        f"crest {f.crest_factor:5.2f}",
        f"    floor    {n.overall_floor_db:7.1f} dBFS   "
        f"snr {n.overall_snr_db:6.1f} dB   "
        f"{'gated' if n.transient_detected else '     '}",
        f"    spectral centroid {f.spectral_centroid_hz:7.0f} Hz   "
        f"bandwidth {f.spectral_bandwidth_hz:7.0f} Hz   "
        f"rolloff {f.spectral_rolloff_hz:7.0f} Hz",
        f"             flatness {f.spectral_flatness:.4f}   "
        f"flux {'n/a' if f.spectral_flux is None else f'{f.spectral_flux:.4f}'}   "
        f"zcr {f.zero_crossing_rate:.4f}   dc {f.dc_offset:+.5f}",
    ]
    if show_bands:
        lines.append("    bands (energy dB / snr dB):")
        for band, state in zip(f.bands, n.bands):
            lines.append(
                f"      {band.label:>10}  {band.db:8.1f}  {state.snr_db:7.1f}"
            )
    if result.gap:
        lines.append("    NOTE: audio was dropped before this frame")
    return lines


def main(argv: Optional[list[str]] = None) -> int:
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
        info = probe_wav(args.input)
    except Exception as exc:
        print(f"error: cannot read {args.input}: {exc}", file=sys.stderr)
        return 2

    analysis = config.analysis
    # In --json mode nothing but the JSON may go to stdout, or the output
    # cannot be parsed.
    if not args.json:
        print(f"input        : {args.input}")
        print(f"format       : {info.sample_rate} Hz, {info.channels} ch, "
              f"{info.duration:.2f} s")
        print(f"framing      : {analysis.frame_ms:g} ms frame / "
              f"{analysis.hop_ms:g} ms hop "
              f"({config.analysis_frame_frames} samples, "
              f"{config.analysis_frames_per_second:.0f} frames/s, "
              f"{analysis.overlap_ratio * 100:.0f}% overlap)")
        print(f"window       : {analysis.window}")
        print(f"noise floor  : rise {analysis.floor_rise_db_per_sec:g} dB/s, "
              f"fall {analysis.floor_fall_db_per_sec:g} dB/s, "
              f"gate {analysis.transient_gate_db:g} dB, "
              f"sustain {analysis.floor_sustain_sec:g} s")
        print("-" * 68)

    source = FileSource(args.input, config)
    if args.seconds:
        limit = int(args.seconds * source.native_sample_rate)
        original = source.read

        def limited(frames=None):
            remaining = limit - source._position
            if remaining <= 0:
                return None
            return original(min(frames or remaining, remaining))

        source.read = limited  # type: ignore[method-assign]

    pipeline = AudioPipeline(config, source, record=False)
    # Lossless and full history: an offline run must analyse the whole file
    # and must be able to report any frame in it.
    worker = pipeline.enable_analysis(
        lossless=True, history_size=int(info.duration * 200) + 1024
    )
    pipeline.start()
    pipeline.wait()
    pipeline.stop()

    results = worker.history or worker.results()
    stats = worker.stats
    payload = {
        "input": os.path.abspath(args.input),
        "sample_rate": info.sample_rate,
        "channels": info.channels,
        "duration": info.duration,
        "framing": {
            "frame_ms": analysis.frame_ms,
            "hop_ms": analysis.hop_ms,
            "frame_samples": config.analysis_frame_frames,
            "hop_samples": config.analysis_hop_frames,
            "frames_per_second": config.analysis_frames_per_second,
        },
        "noise_floor_db": worker.noise_floor_db,
        "stats": stats.to_dict(),
        "frame_count": stats.frames,
        "frames_reported": len(results),
    }

    if args.spectrogram:
        if not args.json:
            print(f"spectrogram  : writing {args.spectrogram}")
        from .analysis.spectrogram import write_pgm

        image = worker.spectrogram.image()
        path = args.spectrogram
        is_png = path.lower().endswith(".png")
        if is_png:
            try:
                import matplotlib
                matplotlib.use("Agg")
                import matplotlib.pyplot as plt
            except ImportError:
                # Fall back rather than fail: a display feature should not
                # need a plotting library to produce output.
                path = path[:-4] + ".pgm"
                is_png = False
                if not args.json:
                    print("  matplotlib not installed; writing PGM instead")
        if is_png:
            fig, ax = plt.subplots(figsize=(12, 4))
            extent = [
                -worker.spectrogram.span_seconds, 0.0,
                0.0, worker.spectrogram.frequencies[-1],
            ]
            ax.imshow(
                image.T, origin="lower", aspect="auto", extent=extent,
                vmin=0.0, vmax=1.0, cmap="magma",
            )
            ax.set_xlabel("seconds ago")
            ax.set_ylabel("Hz")
            fig.tight_layout()
            fig.savefig(path, dpi=120)
            plt.close(fig)
        else:
            write_pgm(path, image.T, worker.spectrogram.frequencies)
        payload["spectrogram"] = os.path.abspath(path)
        payload["spectrogram_shape"] = list(image.shape)
        payload["spectrogram_filled_columns"] = worker.spectrogram.filled_columns

    if args.json:
        show = results[-args.frames:] if args.tail else results[:args.frames]
        payload["frames"] = [r.to_dict() for r in show]
        print(json.dumps(payload, indent=2))
        return 0

    print(f"frames       : {len(results)}  "
          f"({stats._audio_seconds:.2f} s of audio)")
    print(f"cpu          : {stats.analysis_seconds * 1000:.1f} ms total, "
          f"{stats.mean_frame_ms:.3f} ms/frame")
    print(f"realtime     : {stats.realtime_ratio:.4f}x "
          f"({'within budget' if stats.realtime_ratio < 1.0 else 'OVER BUDGET'}; "
          f"<1.0 leaves headroom for the GUI and separation)")
    print(f"dropped      : {stats.blocks_dropped} block(s), "
          f"{stats.gap_frames} frame(s) marked as gapped")
    if stats.errors:
        print("errors:")
        for error in stats.errors:
            print(f"  {error}")
    print(f"noise floor  : {worker.noise_floor_db:.1f} dBFS")
    print("-" * 68)

    show = results[-args.frames:] if args.tail else results[:args.frames]
    if args.frames > 0:
        if show:
            label = "last" if args.tail else "first"
            print(f"{label} {len(show)} frame(s):")
            for result in show:
                print("\n".join(_format_frame(result.index, result, args.bands)))
        else:
            print("no frames produced - the input is shorter than one frame")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
