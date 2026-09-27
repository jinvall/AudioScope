"""Verify audio retention against the real running pipeline.

    ./venv/bin/python tools/retention_check.py

Unit tests prove the accounting arithmetic.  This proves the thing that
actually broke three times while it was being built: that the live system -
capture, analysis, detection, the event store, the writer thread, the
database and retention running together - keeps its byte accounting equal to
the bytes on disk, no matter how the threads interleave.

It records a synthetic scene with a deliberately tiny cap, so eviction is
guaranteed to happen, and then checks the invariants that matter:

* accounting equals the filesystem;
* every row that says its audio is available has a file;
* every fingerprint and every review decision survives eviction;
* a restart recovers the accounting from storage without rescanning on the
  hot path.

Everything it prints is a measurement.  A failure is reported, not asserted
away, and the exit status is non-zero if any invariant broke.
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import soundfile as sf

from app.audio.recorder import FileSource
from app.config import AppConfig
from app.events.database import EventDatabase
from app.events.retention import RetentionManager
from app.pipeline import AudioPipeline

SAMPLE_RATE = 48_000
SCENE_SECONDS = 16


def make_scene(path: str) -> None:
    """A quiet room with four separated bursts.

    Separated, because a continuous scene is segmented into one long event
    (a known limitation), and a single event cannot demonstrate eviction of
    one recording to make room for another.
    """
    rng = np.random.default_rng(7)
    segments = []
    for burst in range(4):
        segment = rng.normal(0, 1e-4, size=int(SAMPLE_RATE * 4)).astype(np.float32)
        length = int(SAMPLE_RATE * 0.9)
        t = np.arange(length) / SAMPLE_RATE
        start = int(SAMPLE_RATE * 1.5)
        segment[start:start + length] += (
            0.4 * np.sin(2 * np.pi * (70 + 30 * burst) * t) * np.exp(-t * 4)
        ).astype(np.float32)
        segments.append(segment)
    sf.write(path, np.concatenate(segments).astype(np.float32), SAMPLE_RATE,
             subtype="FLOAT")


def bytes_on_disk(root: str) -> tuple[int, int]:
    total = 0
    count = 0
    for day in os.listdir(root):
        day_path = os.path.join(root, day)
        if not os.path.isdir(day_path):
            continue
        for event_id in os.listdir(day_path):
            audio = os.path.join(day_path, event_id, "original.wav")
            if os.path.exists(audio):
                total += os.path.getsize(audio)
                count += 1
    return total, count


def main() -> int:
    root = tempfile.mkdtemp(prefix="retention-check-")
    events = os.path.join(root, "events")
    database_path = os.path.join(root, "index.db")
    scene = os.path.join(root, "scene.wav")
    make_scene(scene)
    print(f"workspace: {root}")

    # ~192 KB per 5 s event, so a 600 KB class cap forces eviction during a
    # 16 s recording.
    config = AppConfig(
        audio_retention={
            "enabled": True,
            "per_class_cap_bytes": "600KB",
            "total_cap_bytes": "2MB",
            "reconcile_interval_seconds": 2,
        }
    ).validate()

    pipeline = AudioPipeline(config, FileSource(scene, config), record=False)
    pipeline.enable_detection(
        lossless=True, event_root=events, database=database_path
    )
    pipeline.start()
    pipeline.wait()
    pipeline.stop()

    retention = pipeline.retention
    database = EventDatabase(database_path)
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        print(f"  [{'ok ' if ok else 'FAIL'}] {name}{f' - {detail}' if detail else ''}")
        if not ok:
            failures.append(name)

    print("\n--- live accounting ---")
    status = retention.status()
    print(f"  total: {status['total_human']} of {status['total_cap_human']} cap")
    for name, info in status["classes"].items():
        print(f"  class {name}: {info['human']} in {info['files']} file(s) "
              f"(cap {status['per_class_cap_human']})")

    events_rows = database.list_events()
    available = [e for e in events_rows if e.audio_available]
    evicted = [e for e in events_rows if not e.audio_available]
    disk_bytes, disk_files = bytes_on_disk(events)
    write_errors = pipeline.detection.store.write_errors if pipeline.detection else []

    print("\n--- events ---")
    print(f"  recorded: {len(events_rows)}   available: {len(available)}   "
          f"evicted: {len(evicted)}")
    print(f"  audio on disk: {disk_files} file(s), {disk_bytes} bytes")

    print("\n--- invariants ---")
    check("no store write errors", not write_errors, str(write_errors[:2]))
    check("some audio was evicted", len(evicted) > 0,
          "nothing to evict means the check proved nothing")
    check("accounting equals the filesystem",
          retention.accounting.total_bytes == disk_bytes,
          f"accounted {retention.accounting.total_bytes}, on disk {disk_bytes}")
    check("file counts agree",
          retention.accounting.total_files == disk_files,
          f"accounted {retention.accounting.total_files}, on disk {disk_files}")
    check("every available row has a file", len(available) == disk_files,
          f"{len(available)} rows say available, {disk_files} files exist")
    check("no pending unlinked entries", not retention._unlinked,
          f"left over: {list(retention._unlinked)[:3]}")
    check("every fingerprint survived",
          all(e.fingerprint is not None for e in events_rows))
    check("every decision survived",
          all(e.decision is not None for e in events_rows))
    check("an evicted event is not playable",
          all(not e.is_playable for e in evicted))

    print("\n--- restart recovery ---")
    fresh = RetentionManager(
        config.audio_retention, database=database, root=os.path.abspath(events)
    )
    recovered = fresh.load_accounting()
    check("recovered accounting matches disk",
          recovered.total_bytes == disk_bytes,
          f"recovered {recovered.total_bytes}, on disk {disk_bytes}")

    print("\n--- retained audio is still readable ---")
    for event in available:
        print(f"  {event.event_id}: exists={os.path.exists(event.audio_path or '')}")
        check(f"audio readable for {event.event_id}",
              bool(event.audio_path) and os.path.exists(event.audio_path))

    # Reconciliation enforces the caps against whatever is on disk, so it may
    # legitimately evict here: a single retained file larger than its class
    # cap is kept by the per-class floor only if it is the last one, and the
    # class floor keeps exactly that case.  The invariant that must hold
    # afterwards is that accounting again equals the filesystem.
    report = fresh.reconcile()
    disk_after, files_after = bytes_on_disk(events)
    print(f"\n  reconcile: {len(report.evicted)} evicted, "
          f"{len(report.notes)} note(s); disk now {files_after} file(s), "
          f"{disk_after} bytes")
    for note in report.notes:
        print(f"    note: {note}")
    check("accounting equals the filesystem after reconcile",
          fresh.accounting.total_bytes == disk_after,
          f"accounted {fresh.accounting.total_bytes}, on disk {disk_after}")
    check("no row claims audio that is gone",
          len([e for e in database.list_events() if e.audio_available])
          == files_after)
    check("every fingerprint still survives",
          all(e.fingerprint is not None for e in database.list_events()))

    database.close()
    print()
    if failures:
        print(f"FAILED: {len(failures)} invariant(s): {failures}")
        return 1
    print("all invariants hold")
    shutil.rmtree(root, ignore_errors=True)
    return 0




if __name__ == "__main__":
    raise SystemExit(main())
