# Audio Microscope — Manual

**Version:** as of 2026-09-27
**Applies to:** the working tree at `/home/jason/audioscope`

This is the operating manual. It covers what the system is, how to install and
run it, what every control does, every command and setting, and what to do
when something is wrong. It is written to be read by the person operating the
system, not by the person changing it — for design intent see `docs/`, and
for the current state of the work see `STATE.md`.

Every measured number in this manual was measured on the development host
(4 cores, 15 GB RAM, no GPU) and is labelled as such.

---

## Table of Contents

1. [What Audio Microscope Is](#1-what-audio-microscope-is)
2. [Requirements and Installation](#2-requirements-and-installation)
3. [First Run: The Ordering That Matters](#3-first-run-the-ordering-that-matters)
4. [How the System Works](#4-how-the-system-works)
5. [The Review Workstation](#5-the-review-workstation)
6. [Live Capture Monitor](#6-live-capture-monitor)
7. [Source Separation](#7-source-separation)
8. [Selecting a Region and Extracting an Event](#8-selecting-a-region-and-extracting-an-event)
9. [Event Review and Annotation](#9-event-review-and-annotation)
10. [Recordings, Events and Storage](#10-recordings-events-and-storage)
11. [Command-Line Reference](#11-command-line-reference)
12. [Configuration Reference](#12-configuration-reference)
13. [Files and Directories](#13-files-and-directories)
14. [Verification and Testing](#14-verification-and-testing)
15. [Troubleshooting](#15-troubleshooting)
16. [Performance and Measured Costs](#16-performance-and-measured-costs)
17. [Known Limitations](#17-known-limitations)
18. [Glossary](#18-glossary)
19. [Index](#19-index)

---

## 1. What Audio Microscope Is

Audio Microscope is a desktop application for monitoring and reviewing
**acoustic events** in an environment. It listens continuously, decides on its
own what is worth recording, and lets you go back and listen to what it decided
was worth recording.

It exists because the interesting question about a sound in a room is rarely
"was there a sound". It is "what was it, and what else was happening at the
time". A forty-second recording containing a mouse click, a car going past and
somebody honking is not a useful artefact. Three separate events — the click,
the traffic, the horn — are.

The system does four things:

| | |
|---|---|
| **Capture** | Listens continuously and never blocks on anything else. |
| **Detect and measure** | Decides what is an event, and measures it. |
| **Review** | Lets you listen, look, label, reject and search what was recorded. |
| **Isolate** | Pulls one queried sound out of a recording, so you can hear the click without the traffic. |

The source of audio is a **phone streaming to this machine over TCP** (§3).
Nothing here requires a GPU, a network connection to the internet, or any
service.

Two ideas run through the whole design and are worth knowing before you start:

- **Evidence is never destroyed silently.** Original audio is written losslessly
  and is only ever removed by an explicit, configured, reported policy (§10.4).
  A failed operation leaves the original exactly as it was.
- **Nothing is invented.** A confidence number, a classification or a
  measurement appears only if it was actually computed. Where a value does not
  exist, the interface says so rather than filling the gap.

---

## 2. Requirements and Installation

### 2.1 Host requirements

| | |
|---|---|
| OS | Linux (developed and verified on Ubuntu) |
| CPU | 4 cores or more recommended. 2 works, but separation becomes slow |
| RAM | 2 GB for capture and review; **+5 GB if you use source separation** |
| Disk | Audio is bounded by configuration (§10.4); 10 GB is comfortable |
| GPU | **Not used and not required** |
| Display | Any X11 or Wayland session for the review window |
| Sound card | Only for `--source device`. The default source needs none |

### 2.2 The main environment

The application runs from a virtual environment in the project root:

```bash
cd /home/jason/audioscope
python3 -m venv venv
./venv/bin/pip install -r requirements.txt
```

`requirements.txt` installs numpy, scipy, soundfile, sounddevice and pytest.

> **The review window also needs PyQt5, which is not in `requirements.txt`.**
> On the development host it was already present, so the file was never
> updated. On a fresh machine install it explicitly:
>
> ```bash
> ./venv/bin/pip install PyQt5
> ```
>
> Without it, capture still works and the review window reports that PyQt5
> could not be imported. See §15.

### 2.3 The separation environment (optional)

Source separation needs a **second** virtual environment. This is not
optional complexity — the model requires older versions of two libraries than
the main environment can provide, so they genuinely cannot coexist:

| | main `venv` | separation `venv-sep` |
|---|---|---|
| Python | 3.12 | 3.10 |
| numpy | 2.x | 1.x |
| transformers | — | 4.28.1 |
| torch | — | 2.5.1+cpu |
| PyQt5 | yes | no |

```bash
python3.10 -m venv venv-sep
./venv-sep/bin/pip install torch==2.5.1+cpu torchaudio==2.5.1+cpu torchvision==0.20.1+cpu \
    --index-url https://download.pytorch.org/whl/cpu
./venv-sep/bin/pip install "numpy<2" librosa pyyaml "transformers==4.28.1" \
    "huggingface_hub==0.16.4" "tokenizers<0.14" lightning torchlibrosa \
    safetensors h5py ftfy braceexpand timm pandas wandb webdataset hdf5 wget
```

Then the model itself: the source code and **3.4 GB of weights**, which are
downloaded rather than authored and are not in this repository:

```bash
git clone --depth 1 https://github.com/Audio-AGI/AudioSep third_party/AudioSep
cd third_party/AudioSep
# fetch into checkpoint/ :
#   audiosep_base_4M_steps.ckpt
#   music_speech_audioset_epoch_15_esc_89.98.pt
```

Check it at any time:

```bash
./venv/bin/python -m app.separate --preflight
```

### 2.4 The desktop icon

An installed launcher lives at `~/.local/share/applications/audio-microscope.desktop`
and points at `launch.sh` in this directory. Its `Exec`, `Path` and `Icon`
lines contain absolute paths, so **they must be edited per machine** if you
move the project. The icon image is
`app/gui/srp-css-theme-pack/assets/lary.png`.

---

## 3. First Run: The Ordering That Matters

**The app is the listener. The phone streams to a receiver that is already
running.** Start them in this order:

1. **Start the app first.** `./launch.sh`, or the desktop icon.
2. **Then start the stream on the phone.**

If the phone is started first it will fail to connect, retry, and **give up
after about 30 attempts (2 seconds apart) and never reconnect**. You then have
to restart the phone app — restarting the desktop app will not help.

The window opens on a splash that says so, and reports the real receive
counter while it waits.

To run the two independently:

```bash
./launch.sh --no-capture    # review window only, no listening
./launch.sh --capture-only  # listening only, no window
```

---

## 4. How the System Works

### 4.1 The pipeline

```
phone (or sound card)
    ↓  TCP, raw s16le 44100 Hz mono          [network.py]
conversion once, at the input boundary
    ↓  float32 48000 Hz mono
ring buffer, 30 s
    ├→ analysis          40 ms frames, 10 ms hops → 100 frames/s
    ├→ detection         onset/continuation tracking, change-point segmentation
    │      ↓
    │   event → written to disk, fingerprinted, stored in SQLite
    └→ recording         continuous 15-minute chunks
```

### 4.2 What is measured

Every 40 ms frame, on both the audio and the spectral side:

| Measure | What it tells you |
|---|---|
| **Adaptive noise floor** | The running estimate of the background level, from which louder sounds are measured. It rises slowly and falls quickly, so a sudden sound does not immediately become part of the background |
| **Spectral flux** | How much the spectrum changes between frames. An onset shows as a spike |
| **Spectral flatness** | How noise-like a spectrum is, from tonal to flat. Separates a tone from hiss |
| **Spectral centroid** | The centre of mass of the spectrum in Hz — a rough brightness |
| **CREST factor** | Peak over RMS. High for transients, low for steady sound |
| **SNR** | How far the sound sits above the floor, in dB |
| **Presence** | How continuously the sound occupies its own event |
| **Band energy** | Level in each of nine bands, 20 Hz to 24 kHz |

Segmentation uses a **change point** — a moment where the character of the
sound changes distinctly — so an event can end at a real boundary rather than
at an arbitrary timeout. Detection and analysis together cost about 1.3 ms per
frame (§16), and run on their own threads.

**Pre-roll / post-roll** are the 5 seconds of audio kept either side of an
event, so it is heard in context rather than as a fragment.

### 4.3 Two processes, and why

The launcher starts **capture** as a child process and **the review window**
in its own. They communicate only through files:

| | |
|---|---|
| `events.db` (SQLite) | events, fingerprints, annotations, decisions |
| `events/<date>/event_NNNNNN/` | lossless audio and metadata |
| `recordings/` | continuous chunks, the capture log, and the live status file |

Closing the review window **stops capture**. There is no way to have the
window open and capture stopped other than `--no-capture`.

### 4.4 Audio format

| Stage | Format |
|---|---|
| On the wire | s16le, 44 100 Hz, mono, raw and unframed |
| Everything after capture | float32, 48 000 Hz, mono |
| Stored evidence | float32 WAV, unmodified |
| Separation output | 16-bit WAV at the model's 32 000 Hz |

The conversion happens **once**, at the input boundary. Nothing downstream ever
sees 16-bit or 44.1 kHz.

> **The wire rate is assumed, not declared.** The sender does not state its
> sample rate, and raw PCM has no header, so 44 100 is taken on trust. The Live
> tab shows `rate assumed` so you can see that. If the phone switches to a
> Bluetooth source it may drop to 16 kHz or 8 kHz, and **nothing in the audio
> would reveal it**. See §15.

---

## 5. The Review Workstation

### 5.1 Layout

```
┌─────────────────┬──────────────────────────────────────────────┐
│  Event list     │  Title: event id, duration, decision, source │
│  (left column)  │  Waveform          ← select regions here     │
│                 │  Transport: play, seek, loop, select, extract │
│                 │  Tabs: Event · Acoustics · Source · Review   │
│                 │        Details · Spectrogram · Similar        │
│                 │        Separate · Live                        │
│                 │  Tab contents                                │
│                 │  Save / Confirm / Uncertain / Reject          │
│                 │  Label · Confidence · Notes                   │
├─────────────────┴──────────────────────────────────────────────┤
│  Status line                                                       │
└────────────────────────────────────────────────────────────────┘
```

### 5.2 The event list

Columns are **Time, Duration, Decision, Label**. The decision is shown as
`new` until you judge the event. Right-click or use the filter at the top-left
to narrow the list to *Not reviewed*, *Saved*, *Confirmed*, *Uncertain* or
*Rejected*.

The list polls the database while capture is running, so new events appear
without a restart.

### 5.3 Transport

| Control | Effect |
|---|---|
| **Play / Pause** | Plays the loaded audio. Enabled once audio is loaded |
| **Replay** | Stops and plays from the beginning |
| **Position** | Current playback time; the slider seeks |
| **Loop** | Repeats playback |
| **Select** | Arms region selection on the waveform and spectrogram |
| **Extract selection** | Saves the selected region as a new event (§8) |
| **Volume** | Playback volume; does not affect stored audio |

Clicking the waveform seeks. With **Select** armed, dragging selects instead.

### 5.4 Tabs

| Tab | What it shows |
|---|---|
| **Event** | Identity, time, duration, stored audio, completeness, decision |
| **Acoustics** | Level, noise floor, SNR, spectral centroid, bands, modulation |
| **Source** | Where the audio came from: device or sender, wire format, rate and whether it was assumed, sender config |
| **Review** | The annotation controls in full, with the note about what saving does and does not mean |
| **Details** | Every field stored for the event, grouped |
| **Spectrogram** | Frequency content over time, generated on first open for this event |
| **Similar** | The five closest stored events by fingerprint distance |
| **Separate** | Query input, attempt history, A/B comparison, save (§7) |
| **Live** | Connection verification and a live level trace (§6) |

The **Spectrogram**, **Similar** and **Separate** tabs do their work only when
first opened, so opening the window never starts the model.

### 5.5 Text size

Text is small by default in some environments. Two controls:

- **A- / A+ buttons** in the header, top right. They take effect immediately,
  from 70% to 250% of the base size (17 px). The current percentage is shown
  beside them. This is a per-session setting.
- **`--font-scale`** on the launcher or the GUI command, or the
  `AUDIOMICROSCOPE_FONT_SCALE` environment variable, for a permanent change.

Column widths in the event list and the minimum window width scale with it.

---

## 6. Live Capture Monitor

The **Live** tab answers one question: **is the phone actually sending audio
right now?** Capture publishes it to `recordings/live_status.json` and the
window polls that file.

### 6.1 The states

| State | Meaning | What to do |
|---|---|---|
| **Waiting for the phone** | The app is listening; nothing has connected | Start the stream on the phone. If the phone was started first, restart it — it has already given up |
| **Connected, no audio yet** | The socket is open, nothing has arrived | Usually transient at the start of a connection |
| **Receiving audio** | Audio is arriving | Nothing. This is the healthy state |
| **Stalled** | The connection is open and no audio has arrived recently. **Nothing is being recorded** | Check the phone: it may have frozen, lost network, or the stream may have stopped |
| **No update from capture** | The status file has not been rewritten for several seconds | Capture is not running, or the window is not connected to the right recordings directory |
| **Capture stopped** | The capture process finished | — |

The state is colour-coded: green for receiving, amber for waiting, red for
stalled or unavailable.

### 6.2 What the panel also shows

| | |
|---|---|
| **sender** | Address the phone connected from |
| **received** | Audio seconds and bytes so far |
| **wire** | Wire format, and the rate |
| **rate** | The rate, and whether the sender **declared** it or it is **assumed** |
| **level** | Current level in dBFS, with the peak |
| The trace | A scrolling level trace of the incoming audio |
| The footer | Where the status file is, and the sender's config keys |

The level trace is one RMS value per analysis frame: it shows **when** audio
arrived and **how loud** it was. It cannot show the shape of a transient — use
the Spectrogram tab or the event waveform for that.

### 6.3 If it says "Capture is not publishing status"

Either capture is not running, or the window is looking in the wrong place.
The footer shows the exact path it is reading. If you started capture with a
different `--recordings`, start the window with the same one.

---

## 7. Source Separation

Separation isolates a queried sound from a mixed recording: `a whisper`,
`footsteps`, `a person moving`, `electrical interference` — any phrase at all.

This is **source separation by a real model**, not filtering. A bandpass, a
spectral gate or a denoiser applied to the input is not separation, and the
output here does not correlate with the input (measured: **−0.013**, where a
filtered copy would be about 1.0).

### 7.1 In the window

1. Select an event.
2. Open the **Separate** tab.
3. Type a query. Previously used queries are offered as suggestions.
4. Optionally arm **Select** and drag a region, to separate only part of the event.
5. Press **Isolate**.
6. Wait. See the cost table in §7.4 — this is the part people misread as broken.
7. When it finishes, the attempt appears in the list. Select it and use
   **Compare: original**, **Compare: isolated**, **Compare: enhanced**, or
   **Save isolated…**.

### 7.2 What it produces

```
event_000042/
  original.wav
  separation_001/
    isolated.wav      the model's output. Authoritative
    enhanced.wav      optional, conservative gain and DC removal
    region.wav        only for a selected region: the exact audio separated
    metadata.json     query, timings, cost, validation result
  separation_002/      a second query on the same event
```

**Attempts never overwrite each other.** Separating the same event three
different ways gives three directories, so you can compare them.

The output is validated before it is offered: existence, readability, non-zero
duration, valid sample rate, finite samples, and not catastrophically clipped.
A rejected file is **deleted rather than shown**, because a clipped or
non-finite separation is not playable evidence.

A **quiet** result is not a failure. If the query names something genuinely
absent, a near-silent file is the correct answer and is reported as such.

### 7.3 Comparing and saving

**Compare: original** / **isolated** / **enhanced** load that audio into the
transport. **Save isolated…** writes a copy wherever you choose.

`isolated.wav` is never modified by enhancement. The two are separate files so
you can hear exactly what the enhancement did.

### 7.4 What it costs

Measured on the development host, per separation:

| | |
|---|---|
| First run of a session | **+35 s** to load the model |
| Realtime ratio, steady state | **≈ 3–9× realtime** (5.7× typical) |
| Resident memory | **4.5 GB**, in a separate process |
| Model rate | 32 kHz |

So a 5-second event takes roughly 25–45 seconds, and a 30-second event several
minutes. The panel shows a measured estimate for the selection you have drawn,
and disables **Isolate** while a job is running.

**Longer than 30 seconds cannot be separated** as one job
(`separation.max_input_seconds`). Select a region instead.

The model is unloaded after 15 idle minutes, so a session that pauses pays the
35-second load again.

### 7.5 From the command line

```bash
# is it installed?
./venv/bin/python -m app.separate --preflight

# one file, one query
./venv/bin/python -m app.separate input.wav -q "a whisper" -o isolated.wav

# a region, in seconds
./venv/bin/python -m app.separate input.wav -q scraping --start 1.2 --duration 2.0

# an event, into its own separation_NNN directory
./venv/bin/python -m app.separate input.wav -q footsteps --event event_000042

# machine-readable result
./venv/bin/python -m app.separate input.wav -q "a knock" --json
```

Exit codes: `0` written · `2` usage or configuration · `3` model unavailable
· `4` no usable result. Nothing is written on a non-zero exit.

---

## 8. Selecting a Region and Extracting an Event

An event is a span the detector chose, and on a continuous scene that span can
contain the sound worth keeping together with a car going past and somebody
honking. This feature lets you keep just the part you care about.

### 8.1 How

1. Select an event and let its audio load.
2. Press **Select** in the transport (or "Select a region" on the Separate
   tab — they are the same control).
3. Drag on the **waveform** or the **spectrogram**. The region appears on both,
   because they show the same event on the same timeline.
4. Press **Extract selection**.

The selection becomes a new event with its own audio, duration, level and
provenance. **The parent is never modified.** The new event opens immediately
so you can listen to what you kept.

### 8.2 What you get

The selection is re-analysed by the same pipeline that analyses live audio, so
the result is a real event: measured, classified and fingerprinted — not a
renamed file.

If the detector does not fire inside the selection, the selection is **still
saved**, with its audio, duration, level and provenance, and with **no
classification and no fingerprint** because none was measured. The window says
so. A short transient — a single click is tens of milliseconds — often does not
trip the detector, and an extracted click may therefore be absent from
similarity search.

The saved event records where it came from:

```json
"derivation": {
  "derived_from_event_id": "event_000042",
  "selection_start_seconds": 8.0,
  "selection_end_seconds": 9.5,
  "found_start_seconds": null,
  "found_end_seconds": null
}
```

A null `derived_from_event_id` means the event came from the live stream.

### 8.3 Notes

- You can only extract from an event that still has audio. If audio retention
  has evicted it, the fingerprint and measurements are still there and the
  window says there is no audio to extract from.
- Extraction ignores detections within the first 0.1 s of the window. A window
  that starts mid-stream begins with no noise-floor history, so its first
  frames look like an onset against an unestablished floor; that is an artefact
  of the cut, not a sound.
- Extraction takes a moment (it runs the analysis) and the window stays usable.

---

## 9. Event Review and Annotation

Every event can be judged. The decision is stored in an append-only history:
nothing you record is ever overwritten.

| Control | Meaning |
|---|---|
| **Save** | Worth keeping. The audio is retained and the event is protected from automatic audio eviction |
| **Confirm** | You are confident about the event |
| **Uncertain** | It is real but the classification is doubtful |
| **Reject** | Not a real event. A **negative example**, and as valuable as a positive one — it is protected from eviction too |
| **Label (optional, yours)** | Your own name for the sound. The project's own classification is separate and is not a claim about what the sound is |
| **Your confidence** | Optional, and yours. The system publishes no calibrated confidence and will not invent one |
| **Notes** | Free text |

Saving keeps the event. It is not a claim about what the sound is; a label is
separate and optional.

Human-judged events — saved, confirmed, uncertain **and rejected** — are never
evicted automatically by the audio retention policy (§10.4).

---

## 10. Recordings, Events and Storage

### 10.1 The event tree

```
events/
  2026-09-27/
    event_000042/
      original.wav        float32 WAV, unmodified
      metadata.json       every measurement, and where the audio came from
      separation_001/     §7, if separation has been run
```

Event ids continue past whatever is already on disk, so a new session never
overwrites an earlier one.

### 10.2 Continuous recordings

`recordings/` holds the continuous stream in 15-minute chunks
(`record.chunk_seconds`), kept separately from events. This directory also
holds `capture.log` and the live status file.

### 10.3 The database

`events.db` (SQLite) holds events, fingerprints, measurements, and the
annotation history. It is **derived**: everything in it can be rebuilt from the
event directories. Deleting it loses nothing, and the launcher recreates it.

### 10.4 Audio retention — byte caps

Fingerprints are the application's long-term memory and are kept indefinitely.
**Stored audio is bounded by bytes.**

| Limit | Default |
|---|---|
| Per classification | 500 MB |
| Total | 5 GB |

Both are configurable, in human-readable or explicit form
(`"500MB"`, `524288000`).

When a new recording would exceed a limit, audio is evicted **to make room
before the new bytes are written**, so the total is never transiently exceeded.
Eviction prefers **redundancy over age**: the most redundant recordings go
first — those with the closest near-duplicates already kept — so the store
converges on a representative collection rather than becoming a FIFO archive
of the most recent sounds.

The policy never reduces a classification to zero audio files, and never
evicts anything you have judged. An event whose audio has been evicted stays
fully listable, searchable, comparable and reviewable; the interface shows its
audio as evicted rather than as never recorded.

See `docs/DATA_AND_STORAGE.md` §14 for the full policy.

---

## 11. Command-Line Reference

### 11.1 `python -m app` — the launcher

Starts capture and opens the review window.

| Option | Default | Meaning |
|---|---|---|
| `--source {network,device}` | `network` | Where audio comes from |
| `--device` | — | Input device index, name or substring |
| `--port` | `8190` | TCP stream port |
| `--db` | `events.db` | Event database |
| `--events` | `events` | Event directory root |
| `--recordings` | `recordings` | Continuous recording directory, and where live status is published |
| `--theme {dark,light}` | `dark` | Theme flavour |
| `--font-scale` | `1.0` | Multiply every text size |
| `--seconds` | — | Stop capture after this long |
| `--no-capture` | off | Open the window without listening |
| `--no-splash` | off | Skip the start-the-stream splash |
| `--capture-only` | off | Listen without opening the window |

`./launch.sh` passes its own arguments through to this command and honours
`AUDIOMICROSCOPE_SOURCE`, `AUDIOMICROSCOPE_DB` and
`AUDIOMICROSCOPE_FONT_SCALE`.

### 11.2 `python -m app.capture` — capture only

| Option | Default | Meaning |
|---|---|---|
| `-s, --source {device,network}` | `network` | Where audio comes from |
| `-d, --device` | — | Device index, name or substring |
| `--port` | `8190` | Stream port |
| `--reserve` | `8060-8064` | Ports to hold for the session |
| `--no-reserve` | off | Do not hold the extra ports |
| `-o, --output` | `recordings` | Recording directory |
| `--events` | `events` | Event directory root |
| `--db` | — | Event database |
| `-c, --config` | — | JSON configuration file |
| `--seconds` | — | Stop after this long |
| `--no-record` | off | Detect without writing continuous chunks |
| `--no-analysis` | off | Capture without analysis |
| `--no-detection` | off | Capture without event detection |
| `--wait-client` | — | Report client state after a pause; never gates startup |
| `--status` | off | Print per-client detail on each status line |
| `--list-devices` | — | List available inputs and exit |

### 11.3 `python -m app.gui` — review window only

| Option | Default | Meaning |
|---|---|---|
| `--db` | `events.db` (or `AUDIOMICROSCOPE_DB`) | Database to review |
| `--recordings` | `recordings` (or `AUDIOMICROSCOPE_RECORDINGS`) | Where live status is published |
| `--events` | `events` (or `AUDIOMICROSCOPE_EVENTS`) | Event tree |
| `--theme {dark,light}` | `dark` | Theme flavour |
| `--font-scale` | `1.0` | Multiply every text size |
| `--theme-dir` | — | Override the theme pack location |
| `--no-splash` | off | Skip the splash |
| `--port` | `8190` | Port shown on the splash |

### 11.4 `python -m app.test` — pipeline over a WAV

The same pipeline, driven from a file, so development is repeatable without a
microphone.

```bash
./venv/bin/python -m app.test input.wav
```

| Option | Default | Meaning |
|---|---|---|
| `-o, --output` | `test_output` | Recording directory |
| `--seconds` | — | Stop after this much audio |
| `--block-size` | — | Override the block size |
| `--chunk-seconds` | — | Override the recording chunk length |
| `--realtime` | off | Pace playback to real time |
| `--no-record` | off | Do not write recordings |
| `--json` | off | Machine-readable summary |

### 11.5 `python -m app.analyse` — analysis only

```bash
./venv/bin/python -m app.analyse input.wav [--frames N] [--seconds N] [--tail]
                                    [--bands] [--spectrogram FILE] [--json]
```

Shows features and the noise-floor estimate for a file.

### 11.6 `python -m app.events` — the event database

```bash
./venv/bin/python -m app.events              # list everything
./venv/bin/python -m app.events --stats      # counts by decision and label
./venv/bin/python -m app.events --similar event_000042
```

| Option | Meaning |
|---|---|
| `--root`, `--day`, `--event` | Narrow the listing |
| `--decision {unreviewed,saved,rejected,uncertain,confirmed}` | Filter by decision |
| `--label` | Filter by label |
| `--stats` | Counts by decision and label |
| `-v, --verbose` | Verbose: per-band energy, noise-floor detail and all measurements, not just identity |
| `--json` | Machine-readable output |
| `--similar EVENT_ID` | Closest stored events by fingerprint |
| `--save` / `--reject` / `--uncertain` / `--confirm` EVENT_ID | Record a decision |
| `--label`, `--user-label`, `--user-confidence`, `--notes` | Annotate |
| `--limit` | Maximum rows |

### 11.7 `python -m app.separate` — source separation

| Option | Default | Meaning |
|---|---|---|
| `input` | — | Input WAV file |
| `-q, --query` | — | **Required.** What to isolate, in your own words |
| `-o, --output` | `isolated.wav` | Where to write the result |
| `--start` | whole file | Region start, in seconds |
| `--duration` | to end | Region length, in seconds |
| `--event EVENT_ID` | — | Separate this event, into a fresh `separation_NNN` directory beside it |
| `--events-root` | `events` | Event tree to search for `--event` |
| `--no-enhance` | off | Write only the raw model output, with no `enhanced.wav` |
| `--json` | off | Print the result metadata as JSON on stdout |
| `--preflight` | off | Report whether separation can run, and why not |
| `-c, --config` | — | JSON configuration file |

`--event` and `--output` are mutually exclusive: `--event` chooses the output
directory itself. Examples are in §7.5.

---

## 12. Configuration Reference

Configuration is a JSON file passed with `-c/--config`, or a file named by the
tooling. Every key below is shown with its default. A file may contain any
subset; omitted keys keep their defaults.

```json
{
  "audio":         { "sample_rate": 48000, "channels": 1, "block_size": 1024, "queue_max_blocks": 256, "device": null },
  "buffer":        { "seconds": 30.0 },
  "event":         { "pre_roll_seconds": 5.0, "post_roll_seconds": 5.0, "max_event_seconds": 30.0 },
  "record":        { "enabled": true, "directory": "recordings", "chunk_seconds": 900.0, "retention_days": null },
  "playback":      { "device": null, "sample_rate": 48000, "volume": 1.0, "loop": false },
  "analysis":      { "frame_ms": 40.0, "hop_ms": 10.0, "window": "hann" },
  "detection":     { "onset_threshold": 0.3, "continuation_threshold": 0.16, "release_timeout": 0.4 },
  "separation":    { "enabled": true },
  "audio_retention": { "enabled": true, "per_class_cap_bytes": 524288000, "total_cap_bytes": 5368709120 }
}
```

### 12.1 The keys that matter most

| Key | Default | What it does |
|---|---|---|
| `audio.sample_rate` | 48000 | Internal rate. Conversion happens once, at input |
| `buffer.seconds` | 30 | Ring buffer length. Must exceed the pre-roll you configure |
| `event.pre_roll_seconds` | 5 | Audio kept before an event starts |
| `event.post_roll_seconds` | 5 | Audio kept after it ends |
| `event.max_event_seconds` | 30 | An event longer than this is truncated |
| `analysis.frame_ms` / `hop_ms` | 40 / 10 | 100 frames per second |
| `detection.onset_threshold` | 0.3 | Higher detects less |
| `detection.release_timeout` | 0.4 | How long a gap ends an event |
| `record.chunk_seconds` | 900 | Continuous recording chunk length |
| `separation.max_input_seconds` | 30 | Longest separable input |
| `separation.workers` | 1 | Deliberately 1; the model is the bottleneck |
| `separation.torch_threads` | 3 | Leaves a core for capture |
| `separation.model_idle_unload_seconds` | 900 | Frees the model's 4.5 GB when idle |
| `audio_retention.per_class_cap_bytes` | 524288000 | 500 MB per classification |
| `audio_retention.total_cap_bytes` | 5368709120 | 5 GB in total |
| `audio_retention.min_files_per_class` | 1 | Never empty a class of audio |

### 12.2 Size values

Byte sizes accept a plain integer or a human-readable string: `524288000`,
`"500MB"`, `"5GB"`, `"1.5GiB"`, `"2g"`. `MB` means 1024², so `"500MB"` is
exactly 524288000 and there is no decimal surprise. Internally everything is
explicit bytes, and a saved configuration is always written in bytes.

### 12.3 Changing the settings that are easiest to regret

- **Lowering `buffer.seconds` below `event.pre_roll_seconds`** will leave
  events without their pre-roll. The metadata records it as missing rather than
  hiding it.
- **Raising the retention caps** is safe but needs the disk.
- **Raising `separation.torch_threads` to 4** takes a core from live capture.
  Detection competes with the audio callback for CPU, and the callback is the
  one thing that must never be late.

---

## 13. Files and Directories

| Path | What it is |
|---|---|
| `venv/` | The application's Python environment |
| `venv-sep/` | The separation model's environment (torch, 1.7 GB) |
| `third_party/AudioSep/` | The model and its weights (3.4 GB) |
| `app/` | The application |
| `app/audio/` | Capture, playback, resampling, ring buffer, the network source |
| `app/analysis/` | Framing, features, noise floor |
| `app/detection/` | Tracking, classification |
| `app/events/` | Events, store, database, fingerprints, retention, segmentation |
| `app/separation/` | The separation model process, worker, validation |
| `app/gui/` | The review window |
| `app/gui/srp-css-theme-pack/` | The design system the window is styled from |
| `docs/` | Design documents, one per subsystem |
| `tasks/` | The phase-by-phase build plan |
| `tests/` | 627 tests |
| `tools/` | Verification and snapshot tools (§14) |
| `events/` | Stored events — your evidence, never committed |
| `recordings/` | Continuous chunks and the capture log |
| `events.db` | The derived event index |
| `launch.sh` | The desktop entry point |
| `run.sh` | Command-line entry point |
| `STATE.md` | What exists and what does not, honestly |
| `PASSDOWN.md` | Operational handover notes |
| `AGENTS.md` | The engineering contract this project is held to |

Nothing under `events/`, `recordings/` or `venv*/` is in version control, and
nothing is ever auto-deleted except by the audio retention policy.

---

## 14. Verification and Testing

```bash
# the whole suite.  Needs the ports 8060-8064 free, so stop the app first.
QT_QPA_PLATFORM=offscreen ./venv/bin/python -m pytest tests/ -q
```

627 tests pass, 2 are skipped (they need the separation model and an opt-in
environment variable). The one test that binds 8060–8064 fails while the app is
running; that is the test being honest about a real port conflict, not a defect.

### Verification tools

These check the running system rather than the code, and are the right way to
answer "does it actually work on this machine".

| Tool | What it proves | Cost |
|---|---|---|
| `tools/separation_check.py` | The whole separation chain with the real model: select, query, run, create, play, save. Also multiple queries and a region | Minutes |
| `tools/retention_check.py` | Byte accounting equals the filesystem; fingerprints survive eviction; a restart recovers its accounting | Seconds |
| `tools/gui_snapshot.py` | Renders a tab of the window to a PNG without a display | Seconds |
| `tools/android_stream_client.py` | A fake sender, for testing the receiver without a phone | — |

---

## 15. Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| **The icon does nothing** | The launcher used to die on startup with an error that `Terminal=false` hid. Fixed, but if the window never appears, run `./launch.sh` in a terminal to see why | Read the traceback |
| **No PyQt5 error, capture works, no window** | PyQt5 is not installed | `./venv/bin/pip install PyQt5` (§2.2) |
| **The phone never connects** | The phone was started before the app | Restart the phone app. It gives up after ~30 attempts and never reconnects |
| **Live tab: "Capture is not publishing status"** | Capture is not running, or the window is reading a different `--recordings` | Check the path in the footer |
| **Live tab: "No update from capture"** | The capture process died or is wedged | The reason is in `recordings/capture.log` |
| **Live tab: "Stalled"** | Connected, but no audio arriving. Nothing is being recorded | Check the phone; the sender may have frozen |
| **The launcher says "capture started" but there is no audio** | Now warns explicitly when capture does not come up. Most often port 8190 is held by another instance | Stop the other instance; only one can hold the port |
| **Separation: "model not available"** | `venv-sep` or the model weights are missing | `./venv/bin/python -m app.separate --preflight` |
| **Separation appears to hang** | It is not. First run loads the model (35 s), then runs at 3–9× realtime | Watch the Live/panel estimate; the Isolate button is disabled while running |
| **Separation: "exceeds the separation limit"** | Input longer than `separation.max_input_seconds` | Select a region instead |
| **Extracted selection has no classification** | The detector did not fire inside it | Expected for short transients. The audio is saved; the record says no classification was measured |
| **An event's audio is gone but the event is not** | Audio retention evicted it | Expected and recorded. The fingerprint, measurements and decisions remain |
| **Events are 20–40 s long** | Continuous ambient sound is segmented as one event | Known behaviour (§17). Extract the part you want |
| **Every duration is slightly wrong** | The sender's sample rate is assumed, not declared | Check the Live tab's rate line. If the phone used Bluetooth, the assumption is wrong |
| **Dots appear in the waveform of a quiet room** | Input overflow on this host's audio path | Counted and reported in the metadata; the audio really is missing |

---

## 16. Performance and Measured Costs

Measured on the development host: 4 cores, 15 GB RAM, no GPU.

| Stage | Cost |
|---|---|
| Analysis | 0.63 ms per frame at 100 frames/s |
| Detection | 0.65 ms per frame |
| Analysis + detection | ~1.3 ms per frame, about one core |
| Live capture | 1.00× realtime, no overruns over a 500 s run |
| Separation, first run | 35 s model load + 3–9× realtime |
| Separation, resident | 4.5 GB |
| Review window | idle-polling only; a short file read per tick for live status |

The audio callback is the priority. Analysis and detection run on their own
threads and drop work rather than delay it; the separation model is a separate
process, capped at 3 threads and `nice`d, so it cannot starve capture.

---

## 17. Known Limitations

Stated plainly, because a system that hides these is worse than one that has
them.

1. **Continuous ambient sound becomes one long event.** The detector is
   deliberately broad; a 0.4 s release does not segment steady ambient audio.
   On real phone audio this yields 10–40 s overlapping spans. Use extraction
   (§8) to keep the part you want.
2. **Short bursts read as "sustained".** A burst's presence is measured against
   its own event's span, so a sequence of footfalls scores low because its gaps
   fall inside one event.
3. **The wire sample rate is assumed.** The sender does not declare it. Every
   duration, spectrum and fingerprint is derived from that assumption.
4. **Input overflow on this host.** The local audio path drops a handful of
   samples per 10 s. Counted and reported; the audio really is missing. It does
   not affect the network source.
5. **A signal present 100% of the time is invisible** to the adaptive floor.
6. **Similarity is a distance, not a classifier**, validated on synthetic audio.
7. **The whisper classifier has never heard a real whisper.** Whether a real
   whisper is detected, and then separated well, is untested. A listener has to
   judge it.
8. **Separation is expensive and single-threaded by design.** One worker, 3–9×
   realtime, 4.5 GB resident, and a 30-second ceiling per job.
9. **The Compare/Save row is below the fold** of the Separate panel on a short
   window.
10. **Audio retention evicts whole files only.** A long event is never split
    across a cap. A single file larger than its class cap is kept and the
    overage reported, rather than destroying the last copy of a class.
11. **Retention is a state, not a panel.** An evicted event reads as "audio
    evicted"; there is no per-class usage display.
12. **The GUI must be told the events and recordings trees** (`--events`,
    `--recordings`), or the window looks in the wrong place. The launcher
    passes both through.
13. **Two virtual environments are required for separation**, and the weights
    are 3.4 GB that are not in the repository.
14. **The separation model has not been judged on this project's real audio.**
    It is proven to produce genuine separations, not to produce *good* ones.
15. **The live level trace is a level, not a waveform** — one value per frame.
16. **An extracted selection is not always classified**, so it may not appear in
    similarity search.
17. **Column widths in the event list are fixed**, and scale with the font size
    but do not reflow.

---

## 18. Glossary

**Adaptive noise floor** — the running estimate of the background level, from
which louder sounds are measured. It rises slowly and falls quickly, so a
sudden sound does not immediately become part of the background.

**Analysis frame** — 40 ms of audio, the unit features are computed over. The
system produces 100 per second.

**Band** — a frequency range, used for spectral balance. Bands used here:
20–80, 80–250, 250–500, 500–1k, 1–2k, 2–4k, 4–8k, 8–16k, 16–24k Hz.

**Change point** — a moment where the character of the sound changes
distinctly. Used to end an event at a boundary that is a real change rather
than an arbitrary timeout.

**Classification** — the detector's own generic label for a sound. It is
**not** a claim about what the sound is, and it is not calibrated; the
interface publishes no confidence number for it.

**Continuous recording** — the unbroken stream written in 15-minute chunks,
separate from detected events.

**CREST factor** — peak divided by RMS. High for transients, low for steady
sound.

**dBFS** — decibels relative to full scale. 0 dBFS is the maximum; quieter
values are negative.

**Decision** — your judgement of an event: unreviewed, saved, confirmed,
uncertain or rejected. Recorded in an append-only history.

**Derivation** — the record of where an extracted event came from: its parent
event and the region within it.

**Detection** — deciding that something is an event, and following it from
onset to end.

**Event** — a span of audio the detector decided was worth recording, with 5
seconds of context either side by default.

**Fingerprint** — a vector of ~23 measured features describing an event's
acoustic character. Used for similarity search, independent of its label. Kept
indefinitely.

**Hop** — 10 ms, how often a new analysis frame is started. Frames overlap:
each is 40 ms long, starting every 10 ms.

**Live status** — the connection and level state the capture process publishes
to `recordings/live_status.json` for the window to read.

**Negative example** — an event marked *rejected*. Valuable: it is a labelled
example of something that is not worth keeping, and is protected from eviction.

**Onset** — the start of a sound, where energy rises above the floor.

**Pre-roll / post-roll** — audio kept before and after an event, so it is
heard in context.

**Presence** — how continuously a sound occupies its own event, 0 to 1.

**Query** — the phrase describing what to isolate, in your own words.

**Realtime ratio** — processing seconds per second of audio. Above 1.0 means
slower than real time.

**Retention** — the policy deciding which stored audio to keep. Fingerprints
are never affected; only audio is.

**Ring buffer** — the 30-second rolling buffer between capture and everything
downstream, so consumers can read recent audio without touching the input
path.

**SNR** — signal to noise ratio, in dB: how far a sound sits above the floor.

**Source separation** — isolating one sound from a mixture using a model that
understands the query. Not filtering.

**Spectral centroid** — the centre of mass of the spectrum, in Hz. A rough
"brightness" of the sound.

**Spectral flatness** — how noise-like a spectrum is, 0 (tonal) to 1 (flat).

**Spectral flux** — how much the spectrum changes between frames. Onsets show
as a spike.

**Stall** — a connection that is open while no audio arrives. The connection
looks healthy and nothing is being recorded.

**Track** — one sender's connection, with its own resampler.

**Wire format** — what travels over the network: raw, unframed s16le PCM.

**Wire rate** — the sample rate on that wire. Assumed, not declared (§4.4).

---

## 19. Index

**A**
Adaptive noise floor — §4.1, §18
Analysis frame — §4.1, §12.1, §18
`app.capture` — §11.2
Audio retention — §10.4, §12.1, §17.10
`AUDIOMICROSCOPE_FONT_SCALE` — §5.5, §11.1
`AUDIOMICROSCOPE_SOURCE` — §11.1

**B**
Band — §18
Byte caps — §10.4, §12.2

**C**
Change point — §18
Classification — §9, §18
Click-to-seek — §5.3
Closing the window stops capture — §4.3, §3
Continuous recording — §10.2, §18
CREST factor — §18

**D**
dBFS — §18
Decision — §9, §18
Derivation — §8.2, §18
Detection — §4.1, §18
Desktop icon — §2.4, §15
`duration_seconds` — §7.2

**E**
Event — §4.1, §10.1, §18
Event id — §10.1
Event list — §5.2
`--events` — §11.1, §17.12
Event tree — §10.1
Extract selection — §8
Extraction — §8

**F**
Fingerprint — §9, §10.4, §18
Font size — §5.5, §12.1
`--font-scale` — §5.5, §11.1

**G**
Glossary — §18
GPU — §2.1

**H**
Hop — §4.1, §18

**I**
Index — §19
Input overflow — §17.4
Installation — §2

**J**
Jump to Live tab — §6

**K**
Known limitations — §17

**L**
Label — §9
Live capture monitor — §6
Live status file — §6, §13
`launch.sh` — §3, §11.1

**M**
Manual — this document
Model load — §7.4
`--max_input_seconds` — §7.4, §12.1

**N**
Negative example — §9, §18
Network source — §4.3, §11.2
No update from capture — §6.1, §15
Noise floor — §4.1, §18

**O**
Onset — §4.1, §18

**P**
Phone ordering — §3
Play — §5.3
Port 8190 — §3, §4.1, §15
Post-roll — §12.1, §18
Pre-roll — §12.1, §18
Presence — §18
Precedence of the invariant — §1

**Q**
Query — §7, §18

**R**
Realtime ratio — §7.4, §16, §18
Recording chunks — §10.2
Region selection — §8
`recordings/` — §10.2, §13
`--recordings` — §11.1, §17.12
Reject — §9
Retention — §10.4, §18
`run.sh` — §13

**S**
Save — §9
Similarity — §17.6, §18
Source separation — §7, §18
Spectrogram — §5.4, §8.1
Spectral centroid — §18
Spectral flatness — §18
Spectral flux — §18
SQLite — §10.3
Stall — §6.1, §15, §18
`STATE.md` — §13
`separation_NNN` — §7.2
Separation — §7
Separation environment — §2.3

**T**
Table of contents — top of document
Text size — §5.5
Troubleshooting — §15
Trust and confidence — §1, §9

**U**
Uncertain — §9

**V**
`venv` — §2.2, §13
`venv-sep` — §2.3, §13
Verification tools — §14

**W**
Wave rate assumed — §4.3, §6.2, §15, §17.3
Wire format — §4.3, §18
`--wait-client` — §11.2

---

*For design intent see `docs/`. For the current state of the work — including
what is unfinished — see `STATE.md`. For handover notes see `PASSDOWN.md`.*
