# GUI — Event Review Workstation

Phase 5. The human-review surface for the event, fingerprint and database
system built in Phase 3.

    python -m app.gui                       # review events.db
    python -m app.gui --db events.db --theme light
    ./run.sh gui

    app/gui/
        theme.py       SRP design tokens -> Qt styling
        audioview.py   waveform and spectrogram maths (no Qt)
        controller.py  the only accessor to the backend
        formatting.py  human-readable presentation
        widgets.py     Qt views
        main_window.py assembly, playback, live updates
        app.py         entry point

Everything the GUI shows is read from the existing backend. It starts no
detection, recomputes no fingerprint, and opens no SQL of its own.

---

## 1. Core principle: the GUI is a client

    Audio input -> detection -> event store -> fingerprint -> SQLite
                                                              |
                                                              v
                                                          GUI review
                                                              |
                                              decision + label + notes

One event model (`app/events/event.py`), one persistence model
(`app/events/database.py`), one decision vocabulary (`Decision`). The GUI is a
view over those plus a controller; it is not a second copy of any of them.

The one place the GUI *writes* is `controller.annotate`, which calls
`EventDatabase.annotate`. Detector measurements are never touched — a test
asserts a decision leaves the fingerprint and the detector's own label intact.

---

## 2. Framework

**PyQt5 5.15.10**, already present in the environment.

Why, and what was rejected:

* **PyQt5** was already installed, so it adds no dependency. Qt gives a real
  desktop toolkit on Linux, and `QAbstractItemView` gives a properly aligned
  table for the event list.
* **Qt Multimedia is not installed** in this environment, so it could not have
  been used for playback. This turned out to be the right outcome anyway: the
  backend's `AudioPlayer` (sounddevice) is reused instead of adding a second
  decoding path, as the brief asks.
* **A browser-based UI** would have needed a server or a desktop wrapper, and
  would have added a runtime the project does not otherwise need.
* No new dependency was added for charting: the spectrogram is a `QImage` built
  from a colour lookup table, so matplotlib is not needed.

New dependencies: **none**.

---

## 3. Theme-pack integration

`app/gui/srp-css-theme-pack/` is the design system, and it is the single source
of truth. `theme.py` **parses its CSS at runtime** rather than copying colours
into Python, so editing `srp-theme.css` changes the application without touching
any code.

Two details that took reading the pack to get right:

* The pack names the same concept two ways — `--srp-bg` in the theme file and
  `--color-bg` in the token file. Lookups use the short name and resolve through
  those prefixes.
* Its cascade is: the base block defines every token, and the optional light
  flavour overrides **colours only**. Parsing only the requested flavour left
  light mode with no radii or spacing, because the light block never declares
  them. The loader mirrors the cascade: base first, light overlay second.

Qt does not read CSS variables, so `build_stylesheet` translates the palette
into Qt style-sheet syntax, and `Theme.color/px/spectral_ramp` provide colours
for the places QSS has no hook (waveform stroke, spectrogram ramp).

The theme pack's own **spectral gradient** drives the spectrogram, so the display
uses the ramp the design system intended rather than an arbitrary colormap.

Both flavours resolve: 60 tokens, `bg #0f0a18` dark and `#f5f6fb` light, with
the shared metrics intact.

---

## 4. Event list

Four columns by default — **Time, Duration, Decision, Label** — in a tree, so
they actually line up. A **Details** toggle adds onsets, presence, level, SNR
and segmentation reason, so the default view does not look like a database dump
and the technical values are still one click away.

Rows carry a tooltip with the event id, label, decision and status. Events are
newest first and the list is reloaded when the database's change marker moves,
so **new events appear without restarting**.

---

## 5. Inspector

Tabs: **Event, Acoustics, Source, Review, Details, Spectrogram, Similar**.

* **Event** — identity, duration, how much audio is stored, whether the span is
  complete, and the current decision.
* **Acoustics** — mean level, level variation, peak SNR, SNR variation,
  centroid and its variation, flatness, presence, onsets and onset rate, from
  the **stored fingerprint**. This is the §17 summary, not a vector dump; the
  raw fields are summarised (mean/std/min/max) and the vector length is listed
  in Details.
* **Source** — where the audio came from: source name, input format, source and
  internal rate, channels, stored format, the audio file and how much of the
  span survived. Any settings the sender reported are shown verbatim as data,
  with no interpretation attached.
* **Review** — the decision, label, notes, reviewer confidence, annotation
  count, and the detector's own label and confidence side by side.
* **Details** — versions, segmentation reason, format, merged detections,
  truncation, source name.
* **Spectrogram / Similar** — built on demand (section 8).

The source is also shown in the header under the title, because "which input did
this come from" is the first question about a capture.

---

## 6. Human review

The four buttons map to the backend `Decision` enum unchanged:

    Save -> Decision.SAVED        Confirm  -> Decision.CONFIRMED
    Uncertain -> Decision.UNCERTAIN  Reject -> Decision.REJECTED

No competing vocabulary is invented; a test asserts no value outside the enum is
offered. Under the buttons the window says, in words:

> Saving keeps the event. It is not a claim about what the sound is: a label is
> separate, and optional.

Label, notes and a **Your confidence** slider are separate inputs, and the
current decision's button shows as checked. The detector's confidence is never
merged with the reviewer's: the backend reports it as `not calibrated`, and the
GUI prints that rather than a number, because a rule-based score is not a
calibrated probability.

Rejected events are kept. Reject is a review decision, not a delete; there is
no destructive action in this GUI at all.

---

## 7. Audio and waveform

Playback reuses `app/audio/playback.py`. It plays on its own PortAudio callback
thread, so `play()` returns immediately and detection is untouched.

Transport: play/pause, replay, a seek slider, position and duration readouts,
loop and volume. The playhead follows the player and the waveform can be
clicked to seek.

The waveform is a **min/max envelope** (`audioview.build_envelope`), not a mean:
a mean renders a quiet or AC-coupled signal as a flat line and hides exactly
the transients a reviewer is looking for. A test asserts a single full-scale
impulse survives as a peak.

The envelope is **fixed at 1200 columns**, so its cost is independent of how
long the event is, and it is built in a worker thread on selection and cached.

---

## 8. Background work and CPU

Nothing expensive runs on the GUI thread. The controller dispatches audio
decode, envelope and spectrogram to throwaway threads and returns a `Pending`
immediately; the window renders what is ready and refreshes when a result lands.

A thread per task is deliberate: with a pool, one long decode would occupy a
slot and delay the next task. With a thread each, the GUI never waits and tasks
cannot queue behind each other.

There are exactly two timers, and no busy waiting:

| timer | interval | cost |
| --- | --- | --- |
| Playhead | 50 ms, **only while playing** | negligible |
| Live change marker | 2 s, one cheap aggregate query | **0.42% of a core** |

The live check calls `EventDatabase.change_marker()` — `COUNT(*)` and
`MAX(created_at)` — and rebuilds the list only when that tuple changes. A viewer
that re-listed every event on a timer would be a full-table scan per tick.

**Nothing is generated speculatively.** No waveform is built for events nobody
selects, and no spectrogram is computed until its tab is opened. Opening the
Spectrogram tab is the only thing that triggers it, and the result is cached per
event.

### Measured

| | |
| --- | --- |
| Live marker poll | 0.042 ms per 2 s → **0.42% of a core** |
| Event selection, logic | **8.2 ms** |
| Waveform render, when actually painted | ~12 ms |
| Window repaint | 0.4 ms |
| Fingerprint use | none — read from the store |

Event selection was **48.5 ms** before two fixes: panel rows are now updated in
place instead of being destroyed and rebuilt, and the audio stream is only torn
down if one is actually loaded. Teardown of a PortAudio stream on every
selection was the larger cost.

The detector is not involved in any of this, and the GUI performs no analysis
that belongs to the backend: the spectrogram reuses the backend's
`FeatureExtractor` and framing, so it is the same measurement the detector
makes, not a second display-only pipeline.

---

## 9. Error handling

| situation | behaviour |
| --- | --- |
| No database | refuses to start, with the command to create one |
| Missing audio | "Audio unavailable" in the waveform, transport disabled, **metadata and annotation still work** |
| Unreadable audio | distinguished from missing, shown in the inspector, review still possible |
| Playback failure | reported in the status bar; detection and the window keep running |
| Database failure | visible error, current state preserved |
| New events arriving | list refreshes; **playback is not interrupted** |
| Missing theme pack | warning, unstyled Qt, no crash |

Missing and corrupt audio are separate exception types precisely so the window
can say which, and so an event without audio remains fully reviewable.

---

## 10. Testing

`tests/test_gui.py`, 48 tests, headless via `QT_QPA_PLATFORM=offscreen`.

The split is the point: most behaviour lives in Qt-free modules and is tested
directly, and the widget tests check the window is wired to them and does not
redo their work.

Covered: theme parsing and both flavours; missing-pack degradation; colour
normalisation; human formatting including that a null confidence is reported as
absent rather than 0%; envelope transients and width bounds; spectrogram shape;
missing vs corrupt audio; controller listing, filtering, selection, annotation
persistence, notes-only edits, label clearing, similarity, caching and eviction;
the marker being cheap and change-detecting; widget construction, selection
updating the inspector, annotation reaching the database, missing audio being
tolerated, the spectrogram **not** being built until its tab is opened,
selection not regenerating a fingerprint, live tick reloading only on change,
and a check that the GUI modules contain no sound-class vocabulary.

The last one is enforced by a test, not by convention: a term like `footstep` or
`breathing` in the GUI source fails the build. The detector has the equivalent
check.

---

## 11. Known limitations

* **No `import`/A-B playback.** The original design calls for Original /
  Isolated / Enhanced views with A/B switching. Only Original exists, because
  separation is not implemented; the other two have no audio to show.
* **No manual region selection or separation panel.** Both need a separation
  model.
* **Separation and its history are absent** for the same reason.
* **The live poll is a timer, not a push.** The brief allows the smallest
  integration; a 2 s poll of one aggregate is that, and it costs 0.4% of a core.
  The in-process alternative — an `EventWriter.on_stored` callback — is the
  natural upgrade if capture and review ever run in one process.
* **Not visually verified at every window size.** Built and rendered at
  1280x820; the splitter is stretchable, but the column widths are fixed
  floors rather than responsive.
* **Light mode is parsed and applied, not visually reviewed** in depth.
* **Event selection does not scroll the spectrogram or waveform across
  generations** of the same event, because only one generation exists.
