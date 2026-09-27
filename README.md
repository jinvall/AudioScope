# Audio Microscope

A CPU-capable Linux desktop application for continuous acoustic monitoring, event detection, source separation, playback, and evidence recording.

## Core Capability

Audio Microscope continuously listens to an audio input and can detect and investigate:

- whispers
- speech
- footsteps
- movement
- rubbing
- scraping
- impacts
- mechanical sounds
- electrical interference
- unknown sounds

The application is designed around:

    detect
      ↓
    capture
      ↓
    classify
      ↓
    isolate
      ↓
    hear
      ↓
    save

## Important

Detection alone is not the goal.

The application must be able to produce actual isolated audio that the user can hear.

## Requirements

- Linux
- CPU-capable audio processing
- microphone/audio input, **or** a network audio stream
- Python 3.10+
- source-separation model *(Phase 4)*

GPU acceleration is optional. CPU operation is mandatory.

## Install

    python3 -m venv venv
    ./venv/bin/pip install -r requirements.txt

`sounddevice` bundles PortAudio, so no system `portaudio` package is needed.

## Start

    ./run.sh                      # list devices, then capture live audio
    ./run.sh --list-devices       # just list input devices
    ./run.sh --seconds 30         # capture 30 s, then write WAV chunks

## Audio Sources

Audio can enter from a local device or from an Android device over the network.

**Local device**

    ./run.sh --device "hw:0,0"
    ./run.sh --device 0

**Network (primary)**

The primary source is a raw PCM stream from an Android device on port **8190**.
While streaming, ports **806[0-4]-806[0-4]** are reserved by the application.

    ./run.sh --source network

The server starts and runs with **no client connected**; the phone may connect
at any time. The wire format is the one the AMP receiver already speaks: an
optional JSON config line, then raw unframed signed 16-bit little-endian PCM at
44 100 Hz mono, converted once to 48 kHz float32 internally.

Full protocol, client example and operational notes: **`docs/NETWORK_SOURCE.md`**.

Port range is deliberately disjoint from the AMP receiver's 8090-8099.

## Test WAV

    python -m app.test input.wav

Runs the real pipeline over a file, so file mode cannot drift from live mode.

## Events

Detection and classification run automatically during capture and write one
directory per event:

    events/2026-09-27/event_000000/
        metadata.json      classification, evidence, measurements, provenance
        original.wav       the unmodified captured audio

Inspect them with:

    python -m app.events
    python -m app.events --verbose
    python -m app.events --json
    python -m app.events --event event_000003

Every event has a timestamp, a duration, its actual captured audio, and either a
classification or `unknown`. **Confidence is always `null`** — see
`docs/DETECTION_IMPLEMENTATION.md` for why, and for the measured evidence
behind every threshold.

Skip the stage with `--no-detection`.

## Analysis

Continuous analysis runs automatically during capture. To inspect it over a
file:

    python -m app.analyse input.wav
    python -m app.analyse input.wav --frames 20 --bands
    python -m app.analyse input.wav --json
    python -m app.analyse input.wav --spectrogram out.pgm

Gives per-frame features (RMS, peak, crest, zero-crossing rate, DC, spectral
centroid/bandwidth/rolloff/flatness/flux, nine band energies) plus the adaptive
noise floor and signal-to-noise ratio, overall and per band. Costs about a tenth
of a core and never blocks capture.

See **`docs/ANALYSIS.md`**.

## Review Workstation

    python -m app.gui
    python -m app.gui --db events.db --theme light

A desktop GUI over the event, fingerprint and database system. Browse events,
inspect their waveform and metadata, play and seek the audio, and record a
decision with an optional label and notes. Uses the existing
`EventDatabase` and `Decision` enum; the detector is never involved.

Built on PyQt5, already present, and styled from the existing SRP theme pack in
`app/gui/srp-css-theme-pack/`. No new dependencies.

    python tools/gui_snapshot.py --db events.db --out shot.png

renders the window to a PNG without needing a screenshot tool.

See **`docs/GUI.md`**.

## Manual Separation

    python -m app.separate input.wav \
        --query "a whisper" \
        --output isolated.wav

> **Not implemented yet** (Phase 4). This command validates the request and
> then reports plainly that no separation model is wired up. It writes **no**
> output file, because a filtered copy of the input is not source separation
> (AGENTS.md section 2.4).

## Project Structure

    app/
        config.py            all tunables, validated in one place
        capture.py           python -m app.capture   (live device or network)
        test.py              python -m app.test      (file -> same pipeline)
        separate.py          python -m app.separate  (Phase 4)
        pipeline.py          source -> ring buffer -> recorder
        analyse.py           python -m app.analyse (Phase 2)

        analysis/
            frames.py         block-size-independent framing
            features.py       per-frame DSP measurements
            noise_floor.py    adaptive floor and SNR
            worker.py         the analysis thread
            spectrogram.py    rolling spectrogram / waveform buffers

        events/
            event.py         the stored event and its metadata
            store.py         per-event directory, WAV + metadata.json
            fingerprint.py   compact versioned fingerprint + distance
            database.py      SQLite index, annotations, similarity lookup
            persistence.py   background writer, off the detection thread
            cli.py           python -m app.events

        gui/
            theme.py         SRP design tokens -> Qt styling
            audioview.py     waveform and spectrogram maths (no Qt)
            controller.py    the only accessor to the backend
            formatting.py    human-readable presentation
            widgets.py       Qt views
            main_window.py   assembly, playback, live updates
            app.py           entry point

        audio/
            devices.py       input device discovery
            capture.py       PortAudio capture source
            network.py       8190 stream source + 806[0-4]-806[0-4] reservation
            ringbuffer.py    thread-safe rolling buffer, absolute sample clock
            resample.py      streaming 44.1k <-> 48k polyphase resampler
            wavio.py         lossless float32 WAV, atomic writes
            recorder.py      chunked continuous recording
            playback.py      play / pause / stop / seek / loop / volume

        analysis/            Phase 2
        detection/           Phase 3
        separation/          Phase 4
        events/              Phase 3
        gui/                 Phase 5

    docs/
        PROJECT_SPEC.md              ARCHITECTURE.md
        AUDIO_PIPELINE.md            DETECTION_AND_CLASSIFICATION.md
        DETECTION_IMPLEMENTATION.md  Phase 3 segmentation and classification
        EVENT_DATABASE.md           fingerprints, storage, annotation
        GUI.md                       Phase 5 review workstation
        SOURCE_SEPARATION.md         UI_SPEC.md
        DATA_AND_STORAGE.md          PERFORMANCE.md
        TEST_PLAN.md                 ROADMAP.md
        NETWORK_SOURCE.md            audio input over 8190 / 806[0-4]-806[0-4]

    tasks/
        PHASE_01_AUDIO.md            PHASE_02_ANALYSIS.md
        PHASE_03_EVENTS.md           PHASE_04_SEPARATION.md
        PHASE_05_UI.md               PHASE_06_INTEGRATION.md

    tools/
        android_stream_client.py     reference sender for the 8090 protocol

    tests/                          400 tests

## Development

Read:

    AGENTS.md          the project rules
    STATE.md           what is built, what is not, ports, launch
    PASSDOWN.md        handover notes and the traps

before modifying the project.

The agent must follow the requirements in AGENTS.md.

Run the tests:

    ./venv/bin/python -m pytest tests/ -q

## Design Principle

Preserve the original.

Analyze copies.

Separate asynchronously.

Never sacrifice live capture to perform expensive processing.

## Current Status

**Phases 1-3 and 5 are complete and verified. Phase 4 (source separation) is
not started.**

Working: device discovery, live capture at 48 kHz float32 mono, a 30 s
thread-safe ring buffer with absolute addressing and pre-roll extraction, the
8190 network source with 806[0-4]-806[0-4] reservation, chunked lossless WAV recording
(verified bit-exact), playback, file mode through the same pipeline, and
continuous feature extraction with an adaptive noise floor at roughly a tenth of
a core.

The network path was verified against the real Android device on this host: it
connected on its own, sent its config, and streamed 32.7 s of audio that was
recorded and checked.

Not yet built: source separation (Phase 4). The review GUI exists, so the
`isolate -> hear` half of the chain is not end-to-end yet.

## Design Notes

**Why the capture callback is nearly empty.** PortAudio's callback runs on a
real-time thread. It only converts to the internal format and appends to a
bounded queue. Disk writes, analysis and GUI work all happen on other threads,
so a slow disk can never stall capture (AGENTS.md section 2.3).

**Why the ring buffer is addressed absolutely.** Events need audio from
*before* the detector noticed them. The buffer therefore keeps a sample clock
that only ever increases, and reads request a range on that clock. A pre-roll
that cannot be satisfied is *reported*, never quietly shortened.

**Why `start()` is idempotent.** The CLI starts a source so it can accept a
client before the pipeline exists, then the pipeline calls `start()` again. A
source that re-bound its ports on the second call would fail against itself.

**Why unknown config keys are rejected.** Silently ignoring an unrecognised key
is how you end up running with a setting you believe you changed.

**Why a second network client is refused, not mixed.** Two TCP connections carry
independent jitter, so adding one chunk from each would combine samples captured
at different instants. The result would look plausible and be wrong. Refusing is
honest; a bad mix is not.

**Why `amplification` from the device is recorded, not applied.** It is a gain.
Applying it at capture time would bake it into the evidence permanently, so it
is stored in metadata and left for playback to apply.

**Why presence is the central measurement.** Three footsteps in a quiet room and
a sustained whisper have the same *mean* level and the same *mean* spectrum,
because the gaps pull the average down in both cases. Averaging cannot tell
them apart; the fraction of frames genuinely above the noise floor can, and
measures 0.04 against 0.50-0.65.
