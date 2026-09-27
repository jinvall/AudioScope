"""Supervision of the AudioSep model process.

The model does not live in this process.  It lives in a child process
started with ``venv-sep/bin/python``, reached over a newline-delimited JSON
protocol (:mod:`app.separation.runner_main`).

This is the single most important structural decision in Phase 4, and the
reasoning is in ``SeparationConfig``'s docstring: the model costs 4.5 GB and
wants every core, while the audio callback must never be late.  Process
isolation means that even a model that ignores its thread cap cannot starve
capture, and a model that leaks memory cannot take the application down with
it.  A worker thread inside this process could promise neither.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Optional

from ..config import SeparationConfig


class ModelUnavailable(RuntimeError):
    """The model process cannot be started, and the reason is knowable.

    Raised instead of a bare ``FileNotFoundError`` so the GUI and the CLI can
    report *which* prerequisite is missing.  Separation is optional
    functionality; a missing model must degrade to a clear message, never to
    a fabricated result.
    """


@dataclass
class ModelStatus:
    """Observable state of the model process, for the GUI and for tests."""

    running: bool = False
    loaded: bool = False
    pid: Optional[int] = None
    rss_mb: float = 0.0
    load_seconds: float = 0.0
    last_used: float = 0.0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "running": self.running,
            "loaded": self.loaded,
            "pid": self.pid,
            "rss_mb": round(self.rss_mb, 1),
            "load_seconds": round(self.load_seconds, 2),
            "idle_seconds": (
                0.0 if not self.last_used else time.time() - self.last_used
            ),
            "errors": list(self.errors[-5:]),
        }


class ModelProcess:
    """Owns the child process that holds the model.

    Thread-safe, and deliberately serialising: one lock guards both the
    request/response exchange and the process handle, so a second caller
    cannot interleave a line into the protocol.  ``SeparationConfig.workers``
    stays at 1 - queueing several jobs into one model process would buy
    nothing, because the model itself is the bottleneck.
    """

    def __init__(
        self,
        config: SeparationConfig,
        project_root: Optional[str] = None,
    ) -> None:
        self.config = config
        self.root = os.path.abspath(project_root or os.getcwd())
        self.status = ModelStatus()
        self._proc: Optional[subprocess.Popen] = None
        self._lock = threading.RLock()
        self._stderr_tail: list[str] = []

    # ------------------------------------------------------------------
    # Paths
    # ------------------------------------------------------------------
    def resolve(self, path: str) -> str:
        return path if os.path.isabs(path) else os.path.join(self.root, path)

    def preflight(self) -> list[str]:
        """Everything wrong with the installation, as readable messages.

        Checked before any process is started so the user gets "install the
        model venv" rather than a traceback from deep inside torch.
        """
        problems: list[str] = []
        python = self.resolve(self.config.venv_python)
        if not os.path.exists(python):
            problems.append(
                f"separation virtualenv not found: {python}\n"
                f"  create it with: python3.10 -m venv venv-sep && "
                f"pip install -r venv-sep-requirements.txt"
            )
        model_dir = self.resolve(self.config.model_dir)
        for name, value in (
            ("model directory", model_dir),
            ("config", os.path.join(model_dir, self.config.config_yaml)),
            ("checkpoint", os.path.join(model_dir, self.config.checkpoint_path)),
            (
                "query encoder",
                os.path.join(model_dir, self.config.query_encoder_checkpoint),
            ),
        ):
            if not os.path.exists(value):
                problems.append(f"missing {name}: {value}")
        return problems

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------
    def available(self) -> bool:
        return not self.preflight()

    def ensure_started(self) -> None:
        """Start the process if it is not already running."""
        with self._lock:
            if self._proc is not None and self._proc.poll() is None:
                return

            problems = self.preflight()
            if problems:
                raise ModelUnavailable(
                    "source separation is not available:\n  - "
                    + "\n  - ".join(problems)
                )

            env = dict(os.environ)
            env["AUDIOSEP_THREADS"] = str(self.config.torch_threads)
            env["AUDIOSEP_NICE"] = str(self.config.nice_level)
            env["OMP_NUM_THREADS"] = str(self.config.torch_threads)
            env["MKL_NUM_THREADS"] = str(self.config.torch_threads)
            # torch must not try to grab the machine before we cap it.
            env["CUDA_VISIBLE_DEVICES"] = ""
            env["PYTHONUNBUFFERED"] = "1"

            script = os.path.join(
                os.path.dirname(os.path.abspath(__file__)), "runner_main.py"
            )
            # The working directory is the model repository, not the project
            # root, and that is load-bearing rather than cosmetic: AudioSep's
            # CLAP text encoder resolves its own weights through the relative
            # path `checkpoint/music_speech_audioset_epoch_15_esc_89.98.pt`,
            # so it finds them only when run from the repository root - which
            # is how its own inference example runs it.  The failure otherwise
            # reads as "pretrained weights not found" while the file is
            # sitting right there, 2.3 GB of it.
            model_dir = self.resolve(self.config.model_dir)
            # stderr to a pipe, drained by a reader thread: the model is
            # chatty, and an unread pipe would eventually block it.
            self._proc = subprocess.Popen(
                [self.resolve(self.config.venv_python), script],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=model_dir,
                env=env,
            )
            self.status = ModelStatus(running=True, pid=self._proc.pid)
            threading.Thread(
                target=self._drain_stderr,
                args=(self._proc,),
                daemon=True,
                name="audiosep-stderr",
            ).start()

    def _drain_stderr(self, proc: subprocess.Popen) -> None:
        try:
            for line in proc.stderr:
                self._stderr_tail.append(line.rstrip())
                del self._stderr_tail[:-50]
        except Exception:
            pass

    def stderr_tail(self, lines: int = 20) -> list[str]:
        return list(self._stderr_tail[-lines:])

    def stop(self, timeout: float = 5.0) -> None:
        """Ask the process to exit, then insist."""
        with self._lock:
            proc = self._proc
            self._proc = None
            self.status = ModelStatus()
        if proc is None or proc.poll() is not None:
            return
        try:
            self._write(proc, {"op": "shutdown"})
            proc.wait(timeout=timeout)
        except Exception:
            proc.kill()
            try:
                proc.wait(timeout=timeout)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Protocol
    # ------------------------------------------------------------------
    def _write(self, proc: subprocess.Popen, request: dict) -> None:
        assert proc.stdin is not None
        proc.stdin.write(json.dumps(request) + "\n")
        proc.stdin.flush()

    def _read(self, proc: subprocess.Popen, timeout: float) -> dict:
        """Read one response line, enforcing the timeout.

        A thread plus a queue is used rather than a blocking readline: the
        deadline has to be real, because a wedged torch process must not hold
        the worker thread forever.
        """
        lines: "queue.Queue[Optional[str]]" = queue.Queue(maxsize=1)

        def _reader() -> None:
            """Read one JSON response, skipping anything that is not JSON.

            The runner already keeps the protocol on its own descriptor, so
            this should never be needed.  It is here because the alternative
            is worse: a dependency printing one line to stdout would cost a
            35-second model load and a separation, and a skip is free.
            """
            try:
                assert proc.stdout is not None
                for _ in range(64):
                    line = proc.stdout.readline()
                    if not line:
                        lines.put(None)
                        return
                    stripped = line.strip()
                    if not stripped:
                        continue
                    if stripped.startswith("{"):
                        lines.put(stripped)
                        return
                    self._stderr_tail.append(
                        f"[runner stdout] {stripped[:200]}"
                    )
                    del self._stderr_tail[:-50]
                lines.put(None)
            except Exception:
                lines.put(None)
        thread = threading.Thread(target=_reader, daemon=True)
        thread.start()
        try:
            line = lines.get(timeout=timeout)
        except queue.Empty:
            raise TimeoutError(
                f"model process did not respond within {timeout:.0f}s"
            )
        if not line:
            raise RuntimeError(
                "model process exited unexpectedly:\n  "
                + "\n  ".join(self.stderr_tail(10))
            )
        return json.loads(line)

    def request(self, request: dict, timeout: Optional[float] = None) -> dict:
        """Send one request and return the parsed response.

        A dead process is restarted on the next call rather than here, so a
        crash costs the caller one failed job, not a permanently broken
        separator.
        """
        with self._lock:
            self.ensure_started()
            proc = self._proc
            assert proc is not None
            try:
                self._write(proc, request)
                response = self._read(
                    proc, timeout or self.config.job_timeout_seconds
                )
            except (BrokenPipeError, TimeoutError, RuntimeError, json.JSONDecodeError) as exc:
                # The exchange failed; the process is no longer trustworthy.
                self._kill_locked()
                raise RuntimeError(
                    f"model process failed: {exc}\n  "
                    + "\n  ".join(self.stderr_tail(10))
                ) from exc
            self.status.running = True
            self.status.loaded = True
            if response.get("rss_mb"):
                self.status.rss_mb = float(response["rss_mb"])
            return response

    def _kill_locked(self) -> None:
        proc, self._proc = self._proc, None
        if proc is not None and proc.poll() is None:
            proc.kill()
            try:
                proc.wait(timeout=5.0)
            except Exception:
                pass
        self.status = ModelStatus(errors=self.status.errors)

    def load(self) -> dict:
        """Load the model ahead of the first job, if requested."""
        return self.request(
            {
                "op": "load",
                "model_dir": self.resolve(self.config.model_dir),
                "config_yaml": self.config.config_yaml,
                "checkpoint_path": self.config.checkpoint_path,
            },
            timeout=max(120.0, self.config.job_timeout_seconds),
        )

    def separate(
        self,
        input_path: str,
        query: str,
        output_path: str,
        timeout: Optional[float] = None,
    ) -> dict:
        """One separation.  Raises on model failure; never fakes output."""
        return self.request(
            {
                "op": "separate",
                "model_dir": self.resolve(self.config.model_dir),
                "config_yaml": self.config.config_yaml,
                "checkpoint_path": self.config.checkpoint_path,
                "input": os.path.abspath(input_path),
                "query": query,
                "output": os.path.abspath(output_path),
                "sample_rate": self.config.model_sample_rate,
            },
            timeout=timeout,
        )

    def mark_used(self) -> None:
        self.status.last_used = time.time()

    def idle_seconds(self) -> float:
        if not self.status.last_used:
            return 0.0
        return time.time() - self.status.last_used

    def idle_expiry_due(self) -> bool:
        limit = self.config.model_idle_unload_seconds
        return bool(limit) and self.status.loaded and self.idle_seconds() >= limit
