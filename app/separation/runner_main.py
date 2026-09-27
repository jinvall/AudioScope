"""The AudioSep model process.  **Run as a script, not imported.**

Why a separate file instead of a module in this package:

* It executes under ``venv-sep/bin/python`` (torch, librosa, transformers),
  while the application runs under ``venv/bin/python`` (PyQt5, sounddevice).
  The two environments have genuinely incompatible numpy and transformers
  pins - the model needs numpy 1.x and transformers 4.28 to load its own
  checkpoint - so this process cannot import anything from ``app``.
* It therefore takes no configuration object and holds no application state.
  Everything it needs arrives as one JSON request per line.

Protocol
--------
One JSON object per line on stdin, one JSON object per line on stdout.
Stdout is a protocol channel, so model and library output is redirected to
stderr; the parent captures stderr and keeps it as diagnostics.

Requests:

    {"op": "load",    "model_dir": ..., "config_yaml": ..., "checkpoint_path": ...}
    {"op": "separate", "input": ..., "query": ..., "output": ...,
     "sample_rate": 32000, "threads": 3}
    {"op": "ping"}
    {"op": "shutdown"}

Responses always carry ``"ok"``.  A failure carries ``"error"`` and leaves the
model loaded, because a bad query should not cost the next job a 35 s load.

The model is loaded once and kept resident: the load is 35 s and a
separation is 25 s, so reloading per job would dominate every other cost.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback

_MODEL = None
_MODEL_DIR = None


def _log(message: str) -> None:
    """Diagnostics go to stderr; stdout is reserved for the protocol."""
    print(f"[audiosep-runner] {message}", file=sys.stderr, flush=True)


def _load_model(model_dir: str, config_yaml: str, checkpoint_path: str):
    """Build the AudioSep model exactly as its own pipeline does.

    This is the documented inference path: ``build_audiosep`` from the
    repository's ``pipeline.py``, which composes the CLAP text encoder with
    the separation network and loads the trained weights.  No step of the
    separation is reimplemented here.
    """
    global _MODEL, _MODEL_DIR

    import torch

    # torch threads are set before any parallel work begins.  This is the
    # budget that keeps one core for capture (SeparationConfig.torch_threads).
    threads = int(os.environ.get("AUDIOSEP_THREADS", "3"))
    torch.set_num_threads(max(1, threads))
    # The live pipeline is prioritised over separation, so this process also
    # yields the CPU at the scheduler level.
    try:
        nice = int(os.environ.get("AUDIOSEP_NICE", "10"))
        os.nice(nice)
    except (OSError, ValueError):
        pass

    if _MODEL is not None and _MODEL_DIR == model_dir:
        return _MODEL

    started = time.time()
    if model_dir not in sys.path:
        sys.path.insert(0, model_dir)
    from pipeline import build_audiosep

    device = torch.device("cpu")  # AGENTS.md 2.2: CPU is the required path.
    model = build_audiosep(
        config_yaml=os.path.join(model_dir, config_yaml),
        checkpoint_path=os.path.join(model_dir, checkpoint_path),
        device=device,
    )
    _MODEL = model
    _MODEL_DIR = model_dir
    _log(
        f"model loaded in {time.time() - started:.1f}s "
        f"(threads={threads}, rss="
        f"{_rss_mb():.0f} MB)"
    )
    return model


def _rss_mb() -> float:
    try:
        import psutil

        return psutil.Process().memory_info().rss / 1e6
    except Exception:
        return 0.0


def _separate(request: dict) -> dict:
    """Run one query against one file, writing the model's own output.

    The separation itself is AudioSep's, unmodified: the CLAP encoder embeds
    the query, the network estimates the target waveform, and the estimate is
    written as PCM_16 at the model's 32 kHz.

    Note the output is *not* post-processed here.  Enhancement is a separate,
    optional step in the application (AGENTS.md section 17) and must never
    overwrite this file.
    """
    import numpy as np
    import soundfile as sf
    import torch

    model = _load_model(
        request["model_dir"],
        request.get("config_yaml", "config/audiosep_base.yaml"),
        request.get("checkpoint_path", "checkpoint/audiosep_base_4M_steps.ckpt"),
    )

    sample_rate = int(request.get("sample_rate", 32000))
    input_path = request["input"]
    query = request["query"]
    output_path = request["output"]

    # Load and resample once, at the model boundary.  This is the only
    # resample in the path; the application's own files stay at 48 kHz.
    import librosa

    mixture, _ = librosa.load(input_path, sr=sample_rate, mono=True)
    if mixture.size == 0:
        raise ValueError(f"input file is empty: {input_path}")

    started = time.time()
    with torch.no_grad():
        condition = model.query_encoder.get_query_embed(
            modality="text",
            text=[query],
            device=torch.device("cpu"),
        )
        estimate = model.ss_model(
            {
                "mixture": torch.Tensor(mixture)[None, None, :],
                "condition": condition,
            }
        )["waveform"]
        estimate = estimate.squeeze(0).squeeze(0).data.cpu().numpy()
    elapsed = time.time() - started

    if estimate.size == 0:
        raise ValueError("model returned an empty waveform")
    if not np.isfinite(estimate).all():
        # Written anyway, as NaN, so the failure is visible in the artefact
        # and the validator can reject it on evidence rather than on a
        # trust that the model behaved.
        _log("WARNING: model output contains non-finite samples")

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    sf.write(output_path, estimate, sample_rate, subtype="PCM_16")

    input_seconds = mixture.size / float(sample_rate)
    return {
        "ok": True,
        "processing_seconds": elapsed,
        "input_seconds": input_seconds,
        "realtime_ratio": (
            elapsed / input_seconds if input_seconds > 0 else None
        ),
        "sample_rate": sample_rate,
        "output_frames": int(estimate.size),
        "model_rss_mb": _rss_mb(),
    }


def handle(request: dict) -> dict:
    op = request.get("op")
    if op == "ping":
        return {"ok": True, "loaded": _MODEL is not None, "rss_mb": _rss_mb()}
    if op == "load":
        _load_model(
            request["model_dir"],
            request.get("config_yaml", "config/audiosep_base.yaml"),
            request.get("checkpoint_path", "checkpoint/audiosep_base_4M_steps.ckpt"),
        )
        return {"ok": True, "rss_mb": _rss_mb()}
    if op == "separate":
        return _separate(request)
    if op == "shutdown":
        return {"ok": True}
    return {"ok": False, "error": f"unknown op {op!r}"}


def main() -> int:
    # stdout is about to become stderr, and the protocol moves to its own
    # descriptor, so nothing printed from here on can corrupt a response.
    import warnings

    warnings.filterwarnings("ignore")
    _claim_protocol()

    _log(f"runner started (pid {os.getpid()})")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as exc:
            _reply({"ok": False, "error": f"bad request JSON: {exc}"})
            continue
        if request.get("op") == "shutdown":
            _reply({"ok": True})
            return 0
        try:
            _reply(handle(request))
        except Exception as exc:  # noqa: BLE001 - reported, never swallowed
            _log("request failed:\n" + traceback.format_exc())
            _reply({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
    return 0


#: The protocol's own file descriptor, duplicated from stdout before stdout is
#: aimed at stderr.  See :func:`_claim_protocol`.
_PROTOCOL = None


def _claim_protocol() -> None:
    """Move the protocol off stdout, and point stdout at stderr.

    The model and its libraries print freely: AudioSep announces the loaded
    checkpoint on stdout, as do transformers and timm.  A response is one JSON
    object per line, so a single extra line makes the parent's parse fail and
    a perfectly good separation is reported as a protocol error.

    Rather than trying to silence output that is not ours to silence, the
    protocol gets its own descriptor and stdout is aimed at stderr, where the
    parent already collects diagnostics.  Anything anybody prints is then
    harmless by construction, which is the only kind of safe against a
    dependency's print statements.
    """
    global _PROTOCOL
    _PROTOCOL = os.fdopen(os.dup(1), "w", buffering=1)
    os.dup2(2, 1)           # raw fd 1 now writes to stderr
    sys.stdout = sys.stderr  # and so does everything printing to sys.stdout


def _reply(payload: dict) -> None:
    _PROTOCOL.write(json.dumps(payload) + "\n")
    _PROTOCOL.flush()


if __name__ == "__main__":
    raise SystemExit(main())
