# Audio Microscope — Passdown

Handover for a new session. Read this, then `STATE.md`, then the relevant
section of `AGENTS.md` (the project's rules, not its history).

Paths assume `/home/jason/audioscope`. Use absolute paths; the environment
sometimes reports a workspace root of `/`.

---

## 1. First five minutes

```bash
cd /home/jason/audioscope
./venv/bin/python -m pytest tests/ -q 2>&1 | tail -3   # expect: 413 passed
```

Run the GUI tests with `QT_QPA_PLATFORM=offscreen` or they need a display.

To see it:

```bash
./launch.sh
# then on the phone: open the app, press Start
# then in the window: press any key
```

To look without a display, render the window to PNG:

```bash
QT_QPA_PLATFORM=offscreen ./venv/bin/python tools/gui_snapshot.py \
    --db events.db --out /tmp/shot.png --tab Source
```

---

## 2. What the project is

A Linux desktop application that listens continuously, detects and classifies
acoustic events, stores them losslessly with a reviewable human annotation, and
is intended to let a person isolate a queried sound and listen to it.

Phases 1, 2, 3 and 5 are built. **Phase 4 (source separation) is not.** That is
the only missing piece of the mission, and it is the whole of the
`isolate → hear` half.

---

## 3. The one operational thing that bites

**The Android sender gives up after 30 connection attempts (2 s apart) and never
reconnects on its own.** It calls `stopSelf()` and goes quiet.

So the order is: **server first, phone second.** If the phone starts first it
finds nothing listening, dies after about a minute, and then looks permanently
"not connecting" — which has cost real time already. The splash screen exists
specifically to make this ordering explicit rather than a race.

Ports: listener **8190**, reserved **8060-8064**. The splash names the port, so
if the phone is configured for a different one you will see it.

Check the phone's settings:

```bash
adb shell "run-as com.amp.streamer sh -c 'cat /data/data/com.amp.streamer/shared_prefs/*.xml'"
```

Restart the sender:

```bash
adb shell am force-stop com.amp.streamer
adb shell am start -n com.amp.streamer/.MainActivity
```

**Do not** wire adb into the application. A USB or wireless-debugging link times
out on its own, which is why the splash asks for a keypress instead.

---

## 4. What the sender actually does

From `/home/jason/amp/android-app/app/src/main/java/com/amp/streamer/`:

- Mono s16le PCM via `AudioRecord`; disables AGC and noise suppression
- Rate is **44 100 normally, but falls back to 16 000 then 8 000** for
  `BLUETOOTH_SCO` and `WIRED_HEADSET` sources
- Tries sources in order: UNPROCESSED, VOICE_RECOGNITION, WIRED_HEADSET, USB,
  VOICE_COMMUNICATION, BLUETOOTH_SCO, CAMCORDER, MIC, DEFAULT
- Sends **one JSON config line** then raw unframed s16le in 1024-byte reads
- Config keys: `amplification`, `breathing_sensitivity`, `breathing_cooldown`,
  `segment_duration_min` — displayed verbatim as data, never interpreted
- Before sending: preGain → 5-band EQ → amplification, with tanh soft-clip
- Retries 30× at 2 s, then `stopSelf()`

Two consequences already handled in the code:

- **The sample rate is not on the wire.** A declared rate is believed
  (`sample_rate` / `rate` / `samplerate`); otherwise 44 100 is assumed and
  `status()` says `wire_rate_origin: assumed`. This matters because a fallback
  to 16 kHz read as 44.1 kHz would silently corrupt every duration, spectrum and
  fingerprint.
- **"Original" means as received**, not raw mic output. The Source box says so.

---

## 5. Architecture, briefly

```
phone ──8190──▶ NetworkSource ──┐
                               ├─▶ 48k float32 mono ─▶ RingBuffer
mic ───────────▶ DeviceCapture ─┘                          │
                                                            ├─▶ RawRecorder
                                                            └─▶ analysis
                                                                    │
                                                              detection
                                                                    │
                                                  EventStore ─▶ events/<date>/
                                                                event_NNNNNN/
                                                                    │
                                                    EventWriter (thread)
                                                                    │
                                                    EventDatabase (SQLite)
                                                                    │
                                                                  GUI
```

Each stage owns a queue and a thread. Priority: capture, then analysis, then UI,
then writer, then separation. The GUI reads the database and appends
annotations; it never writes detector data and never analyses.

Two things are worth not breaking:

- **`EventTracker` holds a completed event for one frame** before emitting it,
  because merging cannot be decided until the next detection is known. That is
  the only latency the tracker adds.
- **Segmentation boundaries must survive merging.** An event terminated by
  `acoustic_change`, `onset_cluster` or `max_duration` is never merged into its
  neighbour, and the prohibition lives on the *event* rather than on a
  tracker-wide flag. A tracker-wide flag is cleared when the next event begins,
  and the boundary is then silently undone — which is exactly what defeated
  `max_duration` before.

---

## 6. The quiet-lesson that matters most

An adaptive noise floor cannot see a sound that is present most of the time.
Phase 2 originally used the **mean** of accepted sub-window levels; a whisper
occupying 71% of a window scored presence 0.00 at every amplitude from 7 dB to
24 dB above the noise, because the mean is by construction the average level of
whatever occupied the window.

The estimator is now a **low quantile (0.20)** of the sub-window distribution
in linear power, with a 0.1 s sub-window:

| sub-window | whisper presence | footsteps | empty room | phantom band SNR |
|---|---|---|---|---|
| 0.50 s (mean-based, old) | 0.11 | 0.04 | 0.00 | 0.02 dB |
| 0.10 s (quantile, now) | 0.65 | 0.04 | 0.00 | 0.19 dB |

If you touch the noise floor, re-measure that table. It is the difference
between whisper detection working and not.

---

## 7. Document map

| File | What it covers |
|---|---|
| `AGENTS.md` | project rules — read before changing anything |
| `STATE.md` | current state, ports, launch, known limitations |
| `docs/PROJECT_SPEC.md` | the original brief |
| `docs/ARCHITECTURE.md` | stages, threads, queues, priorities |
| `docs/AUDIO_PIPELINE.md` | formats, framing, features, noise floor, segmentation |
| `docs/ANALYSIS.md` | Phase 2 as built, with measurements |
| `docs/DETECTION_IMPLEMENTATION.md` | Phase 3 segmentation and classification |
| `docs/EVENT_DATABASE.md` | fingerprints, schema, annotation model, costs |
| `docs/GUI.md` | Phase 5 review workstation |
| `docs/NETWORK_SOURCE.md` | the phone protocol, ports, client |
| `docs/DATA_AND_STORAGE.md` | event directories, metadata, retention |
| `docs/DETECTION_AND_CLASSIFICATION.md` | the Phase 3 *specification* |
| `docs/ROADMAP.md` | phase status (its numbering differs from `tasks/`) |
| `docs/UI_SPEC.md` | the GUI *specification* |
| `docs/SOURCE_SEPARATION.md` | the Phase 4 *specification* |
| `docs/PERFORMANCE.md` | the performance *specification* |

Files marked *specification* predate the implementation and are left as written.
Two deliberate divergences from them are recorded in
`docs/EVENT_DATABASE.md`: confidence is `null` rather than a percentage, and
event colours follow the review decision rather than a sound class.

---

## 8. Conventions worth keeping

- **Confidence is always `null`.** A rule-based score is not a calibrated
  probability. Scores are reported as `label_scores` plus evidence and a margin.
  Do not add a percentage to make a field look populated.
- **Preserve measurements rather than re-deriving them.** Fingerprints and
  metadata exist so a reviewer can see *why* two events were considered similar.
- **Missing measurements read as absent, not as zero.** The backend's −120 dBFS
  floor displays as "unavailable".
- **Genericity is enforced by test**, not convention: a sound-class word in the
  detector core or the GUI fails the build
  (`test_no_application_specific_terms_in_the_detector_core`,
  `test_gui_modules_contain_no_sound_class_vocabulary`).
- **Never delete a rejected event.** It is a negative example.
- **Atomic writes.** Audio goes to a temporary file and is renamed; a partially
  written event never looks complete.
- **Comment the *why*.** Several decisions here look arbitrary until you read
  the comment: the 0.20 quantile, the 0.1 s sub-window, the full-window
  requirement before a change can be claimed. Those comments are load-bearing.

---

## 9. Where the next session should probably go

Phase 4, in this order:

1. Read `docs/SOURCE_SEPARATION.md` and `OVERVIEW.md` sections 18-21 and 62.
   The model must be real; the brief is explicit that filtering is not
   separation, and `app/separate.py` already refuses honestly rather than
   faking output.
2. Decide the model and check it fits the CPU budget. Analysis (0.63) plus
   detection (0.65) is already ~1.3 ms/frame — near 1.0 of a core. Separation
   will not fit alongside that, so something has to give: a lower analysis
   rate, fewer analysed bands, or more cores. **Pick deliberately and say which.**
3. Keep it on the writer thread model, not the detection thread.
4. `python -m app.separate input.wav --query "a whisper" -o out.wav` currently
   exits 3 and writes nothing. That is the contract to replace.

Two smaller things worth picking up:

- Event segmentation for continuous ambient sound (limitation 1 in `STATE.md`).
  It needs a tracker that reasons about structure, not a lower threshold.
- The `presence` interaction (limitation 2). A burst now looks sustained
  because presence is measured within one event.

---

## 10. Housekeeping

Before finishing, confirm you left nothing running:

```bash
pgrep -af "app\.capture|venv/bin/python -m app"
ss -ltn | grep -E ':(806[0-4]|8190)\b'
```

Strays hold 8060-8064 and the next run dies with `Address already in use`.
Note `pkill -f "app\.capture"` will **not** match the launcher, whose command
line is `python -m app`. Kill that by PID.

Untracked artefacts you may want to clean: `events.db`, `events/`, `recordings/`.
