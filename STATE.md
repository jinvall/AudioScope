# Audio Microscope — State

As of 2026-09-27. Single source of truth for what exists and what does not.

## Status

| Phase | Name | Status |
|---|---|---|
| 1 | Audio Foundation | **done** |
| 2 | Analysis | **done** |
| 3 | Events, fingerprints, database | **done** |
| 4 | Source Separation | **done** |
| 5 | GUI / review workstation | **done** |
| 6 | Integration | effectively done via the single launcher |

**Byte-based audio retention is built** (not a numbered phase): stored audio is
capped at 500 MB per classification and 5 GB in total, both configurable, with
eviction that prefers redundant recordings over the oldest ones. Fingerprints,
measurements and review history are kept indefinitely; only the audio file is
disposable. See `docs/DATA_AND_STORAGE.md` section 14 and
`tools/retention_check.py`.

**Phase 4 is done and proven with the real model.** AudioSep runs on CPU in its
own process (`third_party/AudioSep`, `venv-sep`), reached over a line protocol.
The CLI works, the **Separate** tab works, A/B playback and save work, and a
manually selected waveform region goes through the same job path as a whole
event. `tools/separation_check.py` proves the whole AGENTS.md section 26 chain:
the separated audio correlates **-0.013** with the input mixture, where a
filtered copy would be ≈1.0. See `docs/SOURCE_SEPARATION.md` section 17.

**Timestamps are shown in system time.** They are stored in UTC, which is
right, and converted for display - showing UTC put events up to a day out and,
near midday, in the wrong day entirely.

**The continuous-recording chunk length is adjustable from the running GUI**,
on the Live tab, via a control file the capture process polls. The file being
written is neither cut short nor extended, and the panel shows the length
actually in force rather than the one requested.

**A button clears the audio of every unreviewed event**, keeping all 232
fingerprints, measurements and annotations, and always confirming first. Events
a human has judged are unreachable by it, and separation output is kept.

**Similar events are clickable.** The Similar tab lists the closest stored
events as a table; clicking one opens it in the window, so it can be judged,
played or separated without hunting for it in the list. The event number is
now the first column of the event list, since it is the identifier the CLI,
the metadata and the similarity results all use.

**After an extraction, a Back button returns to the original event** and
disappears as soon as any other event is selected.

**A selected region of an event can be extracted as an event of its own.**
Drag on the waveform or the spectrogram, then "Extract selection": the
selection becomes a new event with its own audio, measurements and provenance,
and the parent is never modified. This is for the case where one long event
holds the sound worth keeping together with a car going past and somebody
honking. See `docs/DATA_AND_STORAGE.md` section 15.

**The GUI has a Live tab** that verifies the capture connection and monitors
the incoming audio: state, sender address, bytes and seconds received, the
wire format, whether the sample rate was *declared* or *assumed*, a level
meter and a scrolling level trace. The capture process publishes it to
`<recordings>/live_status.json` and the window polls that, because the two are
separate processes and the window otherwise cannot ask whether audio is
arriving right now. Verified against a real phone on 8190.

**682 tests passing**, stable across repeated runs.

## Port allocation

| Port | Purpose |
|---|---|
| 8190 | listener for the Android device (raw s16le PCM) |
| 8060-8064 | reserved and held by this app while streaming |

Both are defaults and both are configurable. `STREAM_PORT` and `RESERVED_PORTS`
live in `app/audio/network.py`; the launcher exposes `--port` and
`AUDIOMICROSCOPE_SOURCE` / `AUDIOMICROSCOPE_DB`.

Ports 8090-8099 belong to the AMP receiver (`/home/jason/amp/server/`) and are
never used here.

## Launching

Single entry point — capture plus the review window, both feeding `events.db`:

```bash
cd /home/jason/audioscope
./launch.sh                 # or: ./venv/bin/python -m app
```

Also installed as a desktop icon (**Audio Microscope**, using
`app/gui/srp-css-theme-pack/assets/lary.png`) at
`~/.local/share/applications/audio-microscope.desktop`.

```bash
./launch.sh --no-splash      # skip the splash, open the window directly
./launch.sh --no-capture      # review only, no capture
./launch.sh --capture-only    # capture, no window
./launch.sh --source device   # local microphone instead of the phone
./launch.sh --db other.db --events ev --recordings rec
```

## The startup sequence that matters

The Android sender **stops retrying after 30 attempts at 2 s intervals and never
reconnects on its own.** So:

1. `./launch.sh` — server binds 8190, reserves 8060-8064, opens the splash
2. On the phone: open the app, press Start
3. The splash shows the real received-seconds counter
4. **Press any key** — the review window opens and fills as events arrive

Starting the phone first is the common failure: it finds nothing listening, dies
after ~60 s, and then looks permanently broken.

## Environment specifics

- Interpreter: `./venv/bin/python` (system site-packages; PyQt5 comes from there)
- **PyQt5 5.15.10 present; QtMultimedia absent** — so playback reuses the
  backend `AudioPlayer` (sounddevice), which was the better outcome anyway
- **matplotlib absent** — the spectrogram is a `QImage` built from a colour LUT
- Tests: `QT_QPA_PLATFORM=offscreen ./venv/bin/python -m pytest tests/ -q`
- `tools/gui_snapshot.py` renders the window to a PNG with no display needed

## Architecture

```
phone (44.1k s16le) ──TCP 8190──▶ NetworkSource ──▶ 48k float32 mono
                                                          │
   mic (48k) ─────────────────────────────▶ DeviceCapture │
                                                          ▼
                                    RingBuffer (30 s, absolute clock)
                                                          │
                                              ┌───────────┴───────────┐
                                              ▼                       ▼
                                    analysis (100 fps)          RawRecorder ──▶ recordings/
                                     features + noise floor          15 min chunks
                                              │
                                    detection (onset, segmentation,
                                      classification, fingerprints)
                                              │
                                    EventStore ──▶ events/<date>/<event_NNNNNN>/
                                                      original.wav + metadata.json
                                              │
                                    EventWriter (background thread)
                                              │
                                    EventDatabase (SQLite, WAL) ◀── GUI review
```

Threading: capture → analysis → detection, each on its own queue; persistence on
a fourth thread. The GUI never writes detector data and performs no analysis.

## Layout

```
app/
  launch.py          single entry point (capture child + GUI)
  capture.py         live capture CLI (device | network)
  pipeline.py        source -> ring buffer -> recorder, plus attach helpers
  analyse.py         analysis CLI
  separate.py        Phase 4 stub (refuses honestly)
  __main__.py        dispatch
  config.py          all tunables, validated
  audio/             capture, devices, network, ringbuffer, resample, wavio,
                     recorder, playback
  analysis/          frames, features, noise_floor, worker, spectrogram
  detection/         candidate, tracker, classifier, worker
  events/            event, store, fingerprint, database, persistence,
                     retention (byte-capped audio), cli
  gui/               theme, audioview, controller, formatting, widgets,
                     main_window, splash, app
  gui/srp-css-theme-pack/   design system (19 files, 2.5 MB)
  separation/        model process, separator interface, jobs, worker,
                     separation_NNN store, validation, enhancement
third_party/AudioSep/  vendored separation model + 2.4 GB of weights
venv-sep/            torch 2.5.1+cpu environment for the model process only
docs/                16 documents, see PASSDOWN.md for the map
tasks/               6 phase task files
tools/               gui_snapshot.py, retention_check.py,
                     separation_check.py
tests/               18 files, 560 tests
launch.sh            thin wrapper: resolves the interpreter, execs python -m app
audio-microscope.desktop
```

## What works, verified

- Phone → 8190 → 44.1k converted once → 48k float32 → recorded bit-exactly
- Local microphone at 1.0x realtime
- Adaptive noise floor that converges to the measured room level
- Segmentation into events with three boundary types
- 10 rule-based labels with evidence; `confidence` deliberately always `null`
- Compact versioned fingerprints, stable and comparable
- SQLite index + append-only human annotations
- GUI: event list, inspector, waveform, on-demand spectrogram, playback, review

## Known limitations

State the honest version, not the reassuring one.

1. **Continuous ambient sound becomes one long event.** The detector is
   deliberately broad; a 0.4 s release does not segment steady ambient audio.
   On real phone audio this yields 10-28 s overlapping spans. Fixing it needs a
   segmentation-aware tracker, not a lower threshold.
2. **Short bursts now read as "sustained".** With proper segmentation, a
   0.2 s burst has presence ≈ 1.0 inside its own event. A footstep *sequence*
   still scores low (0.10) because its gaps live inside one event. The whisper
   rule's "sustained" means "relative to this event's own span" — a real
   limitation of the interaction, not a solved problem.
3. **The wire sample rate is assumed, not negotiated.** The sender can fall
   back to 16 kHz / 8 kHz for Bluetooth and wired-headset sources and its config
   line carries no rate. A declared rate is now believed; otherwise 44 100 is
   assumed and `status()` reports `wire_rate_origin: assumed`. Getting this
   wrong would be silent, so it is always stated.
4. **Input overflow on this host.** The ALC297/PulseAudio path drops roughly
   5 samples per 10 s with nothing running, 15-20 with analysis. Block size is
   not the cause (256→42, 512→20, 1024→30, 4096→62 overflows per 3 runs). Counted
   and reported; the audio really is missing.
5. **A signal present 100% of the time is invisible** to an adaptive floor.
6. **Similarity is a distance, not a classifier**, validated only on synthetic
   audio.
7. **The whisper classifier has never heard a real whisper.**
8. **Separation is expensive and single-threaded by design.** One worker, one
   model, ~3-9x realtime on 4 cores, 4.5 GB resident, 35 s to load. Two users
   cannot separate at once, and the model is unloaded after 15 idle minutes, so
   a re-run pays the load again. Long events must be separated by region
   (`max_input_seconds`, default 30 s).
9. **The Compare/Save row sits below the fold** of the Separate panel at
   1280x820 and needs a scroll. The panel's layout is unchanged from when it was
   written; only the wiring was added.
10. **Column widths in the GUI are fixed floors**, not responsive; light mode is
    parsed and applied but not visually reviewed in depth.
11. **Audio retention evicts whole files only.** A long event is never split
    across the cap, because a truncated recording that looks complete is worse
    than none. A single file larger than its class cap is kept and the overage
    reported, rather than the last copy of a class being destroyed.
12. **The GUI shows retention as a state, not a panel.** An evicted event reads
    as "audio evicted" in the list and inspector, and playback explains the
    reason. There is no retention status display showing per-class usage.
13. **The GUI must be told the events tree** (`--events`, or
    `AUDIOMICROSCOPE_EVENTS`). It defaults to the project's `events`, which is
    wrong for any capture run that used a different root. The launcher passes it
    through, but a hand-built window can still get it wrong, and the
    consequence is an attempt landing in an unrelated event's directory.
14. **The two venvs are a real operational cost.** `venv` (PyQt5, numpy 2.x)
    cannot import torch at all: the model needs numpy 1.x and transformers 4.28
    to load its own checkpoint. Separation needs both environments, and the
    model's weights are 3.4 GB that are not in the repository.
15. **Only one process can hold port 8190.** The app *is* the listener - the
    phone streams only to a receiver that is already running, and it gives up
    after 30 retries and never reconnects - so a second capture, or a
    verification run, silently takes the stream away and the phone has to be
    restarted. Worth knowing before diagnosing a "broken" sender.
16. **An extracted selection is not always classified.** The selection is
    re-analysed, and a short quiet transient - a single click is tens of
    milliseconds - does not always trip the detector. The audio is saved
    either way, with the duration, level and provenance, and the record says
    no classification was measured rather than inventing one. So an extracted
    click may have no fingerprint and will not appear in similarity search.
17. **The level trace is a level, not a waveform.** It is one RMS value per
    analysis frame, so it shows when audio arrived and how loud it was; it
    cannot show the shape of a transient. The spectrogram and event waveform
    are the tools for that.
18. **The model has never been judged on this project's real audio.** It is
    proven on synthetic probes and on real captured events, and the output
    provably is not a copy or a filter. Whether it isolates a real whisper, a
    real footstep or a real knock *well* is untested, and the classifier
    upstream of it (limitation 7) has never heard a whisper either. A listener,
    not a test, has to judge quality.
