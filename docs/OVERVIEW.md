# AUDIO MICROSCOPE

## Complete Top-to-Bottom Application Build Directive

---

# 1. PURPOSE

Build a new Linux desktop application for **continuous live audio monitoring, acoustic-event detection, sound identification, source isolation, playback, and evidence recording**.

The application must be able to listen to a continuous incoming audio stream and identify events such as:

* whispers
* speech
* footsteps
* people moving
* clothing/rubbing
* knocks
* taps
* impacts
* scraping
* object movement
* mechanical sounds
* environmental sounds
* electrical interference
* hum
* buzz
* clipping
* digital/dropout artifacts
* unknown/unclassified sounds

The critical requirement is:

> **Detection alone is not sufficient. The application must be able to isolate the detected sound from the mixed recording and let the user hear the isolated result.**

The application must therefore preserve the original audio while providing a second processing path for event detection and on-demand source separation.

The application is intended to operate continuously.

It must not require the user to manually record a file before detection can occur.

---

# 2. CORE DESIGN PRINCIPLE

DO NOT build this as:

```text
microphone
→ VAD
→ speech recognition
→ alert
```

DO NOT build this as:

```text
microphone
→ noise gate
→ FFT
→ threshold
```

DO NOT build this as a system that merely says:

```text
"footstep detected"
```

without producing something the user can actually listen to.

The correct architecture is:

```text
LIVE AUDIO
    │
    ├──────────────────────────────► RAW AUDIO
    │                                  │
    │                                  └──► rolling buffer
    │
    ▼
LOW-COST REAL-TIME ANALYSIS
    │
    ├── noise-floor tracking
    ├── transient detection
    ├── spectral analysis
    ├── speech/whisper analysis
    ├── interference analysis
    └── candidate event generation
                 │
                 ▼
          EVENT TRACKER
                 │
                 ▼
        EVENT + AUDIO WINDOW
                 │
                 ▼
       SOURCE SEPARATION
                 │
                 ▼
        ISOLATED AUDIO
                 │
           ┌─────┴─────┐
           ▼           ▼
        PLAYBACK      SAVE
```

---

# 3. HARD REQUIREMENTS

The following are requirements, not suggestions.

## 3.1 Continuous operation

The application must process a live stream continuously.

It must not require:

* manually starting a recording
* stopping the stream
* waiting for an entire recording
* manually selecting a file for every event

## 3.2 Preserve the original

NEVER destroy or permanently modify the raw stream.

The raw stream must remain available.

Processing must operate on copies/buffers.

## 3.3 Event isolation

Every detected event must have an associated time window.

For example:

```text
event onset:       12.431 sec
event peak:        12.614 sec
event end:         12.891 sec
```

The system should retain additional audio before and after the event.

Default:

```text
pre-roll: 5 seconds
post-roll: 5 seconds
```

Make this configurable.

## 3.4 Play isolated audio

The UI must provide:

```text
PLAY ORIGINAL
PLAY ISOLATED
```

for every separable event.

## 3.5 Save isolated audio

The user must be able to save:

```text
original event
isolated event
```

as WAV.

Use lossless WAV as the primary evidence format.

## 3.6 Unknown sounds

UNKNOWN must be a valid classification.

Do not force an event into:

```text
footstep
whisper
movement
```

just because those are the available labels.

## 3.7 Never discard low-level audio simply because it is quiet

A whisper or distant footstep may be only slightly above the environmental noise floor.

Detection must therefore be relative to the local noise floor, not based solely on absolute dBFS.

---

# 4. PLATFORM

Primary target:

```text
Linux x86_64
Ubuntu 24.04+
```

The application must run without a GPU.

GPU support may be added later, but GPU availability must never be required.

All ML inference must support:

```text
CPU
```

as a first-class execution mode.

Do not assume CUDA exists.

Do not make CUDA a dependency.

---

# 5. TECHNOLOGY STACK

Use Python for the initial implementation.

Recommended stack:

```text
Python 3.12
NumPy
SciPy
sounddevice
soundfile
PyTorch
torchaudio where appropriate
librosa where useful
FastAPI or local application service layer if needed
Qt/PySide6 for desktop GUI
```

Keep the architecture modular enough that components can later be moved to Rust/C++ if CPU performance requires it.

Do not prematurely rewrite the entire application in C++.

---

# 6. AUDIO INPUT

Implement an audio-device manager.

The application must enumerate available:

```text
ALSA
PulseAudio
PipeWire
sounddevice
```

input devices as appropriate for the host.

The UI must show:

```text
Input device
Sample rate
Channels
Format
Input level
Connection state
```

Allow the user to select the input device.

Default to the system-selected input if possible.

---

# 7. AUDIO FORMAT

Internally normalize live processing to:

```text
mono
float32
48 kHz
```

if the source permits it.

If the input is another sample rate:

```text
resample → 48 kHz
```

Do not repeatedly resample the same audio.

Preserve the original input metadata for recordings.

---

# 8. AUDIO PIPELINE

The audio pipeline must be non-blocking.

Never run ML inference directly inside the audio callback.

Use:

```text
audio callback
      ↓
lock-free/ring buffer
      ↓
analysis worker
      ↓
event queue
      ↓
separation worker
```

The audio callback must do as little work as possible.

It must never:

* run PyTorch inference
* write large files
* block on disk
* wait for a network request
* wait for another thread
* run a large FFT unnecessarily
* perform UI operations

---

# 9. ROLLING BUFFER

Implement a circular/ring buffer.

Default:

```text
30 seconds
```

of continuous audio.

Make it configurable:

```text
10–120 seconds
```

The buffer must permit extraction of:

```text
event_start - pre_roll
```

through:

```text
event_end + post_roll
```

without interrupting live capture.

---

# 10. REAL-TIME ANALYSIS

Implement a lightweight DSP analysis stage.

The analysis stage should calculate at least:

```text
RMS
peak
crest factor
zero crossing rate
spectral centroid
spectral bandwidth
spectral rolloff
spectral flatness
spectral flux
band energy
noise floor
dynamic range
```

Maintain multiple frequency bands.

At minimum:

```text
20–80 Hz
80–250 Hz
250–500 Hz
500–1 kHz
1–2 kHz
2–4 kHz
4–8 kHz
8–16 kHz
16–24 kHz
```

Use the available Nyquist range if the input does not reach 48 kHz.

---

# 11. NOISE FLOOR

Implement an adaptive local noise-floor estimator.

Do not use one fixed threshold.

Maintain:

```text
noise_floor_rms
noise_floor_db
noise_floor_spectrum
```

with slow adaptation.

The detector should be able to determine:

```text
signal is 1.2 dB above baseline
signal is 5 dB above baseline
signal is 18 dB above baseline
```

rather than merely:

```text
signal > -40 dB
```

The noise floor must not immediately rise to absorb a real event.

Use asymmetric attack/release behavior.

---

# 12. EVENT CANDIDATE DETECTOR

Implement a cheap first-stage detector.

It should identify candidate events from:

* spectral flux
* transient energy
* sudden band changes
* speech-like modulation
* low-level persistent changes
* harmonic changes
* broadband changes
* unusual spectral patterns
* interference signatures

The detector should produce candidate events, not final classifications.

Example:

```json
{
  "start": 123.421,
  "end": 124.103,
  "peak": 123.774,
  "energy_delta_db": 7.4,
  "candidate_type": "unknown",
  "confidence": 0.62
}
```

---

# 13. WHISPER DETECTION

Whispers must receive special treatment.

Do not treat whispers as ordinary speech.

A whisper can have:

* very low RMS
* weak/no fundamental
* broadband energy
* strong high-frequency speech components
* speech-like temporal modulation

The system should detect speech-like acoustic structure even when the absolute level is low.

Provide classifications:

```text
whisper
quiet speech
speech
possible speech
```

Do not claim certainty where the signal is ambiguous.

---

# 14. FOOTSTEP DETECTION

Footstep detection must combine:

```text
transient characteristics
low-frequency energy
spectral shape
duration
attack/release
temporal spacing
```

Do not classify every transient as a footstep.

The system should track repeated events.

For example:

```text
impact
   ↓
candidate
   ↓
another similar impact
   ↓
similar temporal spacing
   ↓
footstep-sequence candidate
```

Track:

```text
single footstep
footstep sequence
possible walking
```

---

# 15. MOVEMENT DETECTION

Movement is a broad category.

Detect:

```text
person moving
clothing movement
rubbing
scraping
object movement
surface contact
handling
unknown movement
```

The classifier must allow multiple possible labels.

Example:

```text
primary: movement
secondary: rubbing
confidence: 0.71
```

---

# 16. AUDIO INTERFERENCE DETECTOR

Interference must be treated as its own subsystem.

Detect:

```text
DC offset
50/60 Hz hum
harmonics
electrical buzz
periodic interference
clipping
digital clipping
dropouts
sample discontinuity
packet loss
buffer underrun/overrun
broadband noise bursts
RF-like repetitive patterns
mechanical microphone noise
```

Each interference event must have:

```text
type
start
end
severity
confidence
frequency characteristics
```

Example:

```text
INTERFERENCE

Type: 60 Hz electrical hum
Severity: moderate
Confidence: 0.96

Fundamental: 60.1 Hz
Harmonics: 120.2 / 180.3 / 240.4 Hz
```

---

# 17. EVENT CLASSIFICATION

Use ML classification only after cheap DSP has identified a candidate.

The classifier should support open-ended acoustic events.

Do not hard-code the system to only recognize three or four sounds.

The architecture must allow:

```text
whisper
footstep
movement
door
knock
tap
impact
scrape
machine
fan
motor
vehicle
animal
speech
music
electrical interference
unknown
```

and future classes.

---

# 18. SOURCE SEPARATION

This is the most important subsystem.

The application must provide **query-based source isolation**.

Use a model capable of separating a target sound from a mixture based on a semantic target/query.

AudioSep is the preferred initial architecture because its stated purpose is open-domain sound separation using natural-language queries and it is intended for audio-event separation and speech enhancement.

The separation API should conceptually look like:

```python
isolated = separator.separate(
    audio=event_audio,
    query="a whisper"
)
```

or:

```python
isolated = separator.separate(
    audio=event_audio,
    query="footsteps"
)
```

or:

```python
isolated = separator.separate(
    audio=event_audio,
    query="electrical interference"
)
```

The actual implementation must use the selected model's real API.

Do not invent a fake separator API.

---

# 19. SEPARATION QUERIES

Provide predefined queries.

At minimum:

```text
whisper
quiet human speech
human footsteps
a person walking
a person moving
clothing rustling
rubbing sounds
scraping sounds
object movement
knocking
tapping
impact
mechanical noise
electrical interference
electrical hum
background noise
```

Also allow the user to enter a custom query:

```text
Separate:
[________________________________]
```

Examples:

```text
someone whispering
a distant footstep
a person moving around
cloth rubbing against clothing
a chair moving
electrical buzzing
a metallic scraping sound
```

---

# 20. DO NOT ASSUME MUSIC SOURCE SEPARATION IS ENOUGH

Generic vocal/drum/bass separation is not sufficient for this application.

Demucs may be useful as a secondary tool for certain material, but it should not be treated as a universal environmental-sound separator.

The primary system must be designed around environmental/event separation.

---

# 21. SEPARATION WORKER

Source separation must run asynchronously.

Architecture:

```text
Live audio
   │
   ▼
Event detected
   │
   ▼
Extract event window
   │
   ▼
Separation queue
   │
   ├── Worker 1
   ├── Worker 2
   └── ...
```

Initially use:

```text
1 CPU separation worker
```

because multiple simultaneous ML jobs could overwhelm the host.

Make worker count configurable.

---

# 22. CPU PROTECTION

The application must never destroy live-stream performance because an expensive separation operation is running.

Implement:

```text
max concurrent separations = 1
```

by default.

If CPU load becomes excessive:

```text
continue capture
continue detection
queue separation
```

Do not drop the raw stream.

Display:

```text
SEPARATION QUEUE: 3
```

---

# 23. ISOLATED AUDIO POST-PROCESSING

After separation:

1. Remove DC offset.
2. Prevent clipping.
3. Normalize conservatively.
4. Preserve dynamics.
5. Do not aggressively noise-gate the result.
6. Do not apply destructive denoising automatically.

The user must be able to hear what the separator actually produced.

Provide optional:

```text
gain
high-pass
low-pass
noise reduction
compression
```

but keep the raw isolated result available.

---

# 24. THREE AUDIO VIEWS

Every event should provide:

```text
ORIGINAL
ISOLATED
ENHANCED
```

### ORIGINAL

The exact event window from the raw stream.

### ISOLATED

Direct output of the source separator.

### ENHANCED

Optional listening processing applied to the isolated signal.

This distinction is extremely important.

Do not silently modify the evidence.

---

# 25. PLAYBACK

Every event must have:

```text
▶ Original
▶ Isolated
▶ Enhanced
```

Playback controls:

```text
play
pause
stop
seek
loop
volume
```

Add:

```text
A/B
```

so the user can quickly compare:

```text
original ↔ isolated
```

---

# 26. WAVEFORM

Display the event waveform.

Show:

```text
event start
event peak
event end
classification
confidence
```

Overlay detected regions.

Example:

```text
──────────────────────────────────────
        │──── FOOTSTEP ────│
───────────────╱╲─────────────────────
              ╱  ╲
─────────────╱────╲───────────────────
```

---

# 27. SPECTROGRAM

Display a spectrogram for the event.

Allow:

```text
linear frequency
log frequency
dB scale
zoom
frequency cursor
time cursor
```

The spectrogram must remain synchronized with playback.

---

# 28. EVENT TIMELINE

Create a continuously scrolling event timeline.

Example:

```text
21:41:02  ambient
21:41:07  movement
21:41:08  footstep
21:41:09  footstep
21:41:10  footstep
21:41:17  whisper
21:41:23  electrical interference
21:41:28  unknown
```

Clicking an event opens its detail view.

---

# 29. EVENT DETAIL PANEL

Display:

```text
EVENT

Time:
21:41:17.842

Duration:
1.37 sec

Classification:
Whisper

Confidence:
0.73

Signal:
+3.4 dB above local noise floor

Frequency range:
420 Hz – 9.2 kHz

Status:
Separated
```

Buttons:

```text
▶ Original
▶ Isolated
▶ Enhanced

[Re-separate]
[Save Original]
[Save Isolated]
[Save Enhanced]
```

---

# 30. MANUAL ISOLATION

The user must be able to select an arbitrary time region manually.

Example:

```text
[ waveform ]

        ┌───────────────────────┐
────────┤ selected region       ├────────
        └───────────────────────┘
```

Then:

```text
Separate this region
```

with a query:

```text
Separate:
[ a whisper                         ]
```

This is mandatory.

The system must not require the detector to have found the sound first.

---

# 31. MANUAL QUERY

Allow:

```text
Separate Anything:
[____________________________]
```

Examples:

```text
whispering
footsteps
a person moving
a distant mechanical sound
electrical interference
```

The user can run separation on any selected region.

---

# 32. MULTIPLE SEPARATION PASSES

Allow multiple separation attempts on the same event.

For example:

```text
EVENT #183

Original

Separation #1
Query: "whisper"
[Play]

Separation #2
Query: "human speech"
[Play]

Separation #3
Query: "background noise"
[Play]
```

Do not overwrite previous results.

---

# 33. RESULT COMPARISON

Allow side-by-side or A/B comparison:

```text
Original
vs.
Isolated
```

and:

```text
Isolation A
vs.
Isolation B
```

This is especially important when the model produces ambiguous results.

---

# 34. SAVED EVENT FORMAT

Each event should have a directory:

```text
events/
    2026-09-25/
        event_000183/
            metadata.json
            original.wav
            isolated.wav
            enhanced.wav
            spectrogram.png
```

If multiple separations exist:

```text
event_000183/
    metadata.json
    original.wav
    separation_001/
        metadata.json
        isolated.wav
    separation_002/
        metadata.json
        isolated.wav
```

---

# 35. METADATA

Store JSON metadata.

Example:

```json
{
  "event_id": "000183",
  "timestamp": "2026-09-25T21:41:17.842",
  "duration": 1.37,
  "classification": "whisper",
  "confidence": 0.73,
  "noise_floor_db": -61.4,
  "peak_db": -57.9,
  "query": "whisper",
  "separator": "AudioSep",
  "sample_rate": 48000,
  "channels": 1,
  "device": "input device name"
}
```

---

# 36. RAW RECORDING

Provide an optional continuous recording mode.

When enabled:

```text
recordings/
    2026-09-25/
        21-00.wav
        22-00.wav
```

Use chunked files rather than one enormous file.

Default chunk duration:

```text
15 minutes
```

Make configurable.

---

# 37. EVIDENCE INTEGRITY

The raw recording must never be altered by:

* normalization
* denoising
* compression
* separation
* filtering

All processing produces derivative files.

---

# 38. INTERFERENCE VISUALIZATION

Display interference separately from acoustic events.

Example:

```text
AUDIO HEALTH

Noise floor       -61.4 dB
Peak              -34.2 dB
Clipping          0.00%
Dropouts          0
DC offset         0.001
60 Hz             -48 dB
120 Hz            -57 dB

STATUS             CLEAN
```

When interference is detected:

```text
STATUS             INTERFERENCE

60 Hz HUM
Confidence: 96%
Severity: Moderate
```

---

# 39. EVENT CONFIDENCE

Confidence must never be represented as certainty.

Use:

```text
0–100%
```

but label it:

```text
model confidence
```

not:

```text
probability that this definitely happened
```

---

# 40. UNKNOWN EVENT HANDLING

If the system detects something unusual but cannot identify it:

```text
UNKNOWN SOUND
```

must appear in the timeline.

The user must still be able to:

```text
▶ Play original
▶ Isolate
▶ Save
```

The user should be able to enter:

```text
"What is this?"
```

as a query for source separation.

---

# 41. LEARNING / MEMORY

Do not initially train the model online.

Instead, implement an event library.

Every saved event can optionally be tagged:

```text
footstep
whisper
movement
false positive
electrical
unknown
```

This creates a dataset for future model adaptation.

Also store feature vectors where practical.

The architecture should leave room for a future similarity system:

```text
new sound
    ↓
embedding
    ↓
nearest previous sounds
    ↓
"similar to these previous events"
```

Do not require a closed-world classifier.

---

# 42. SIMILARITY SEARCH

Plan the data model so events can eventually support:

```text
Find sounds similar to this
```

This should work independently of classification.

Example:

```text
EVENT #291

Classification:
unknown

Similar previous events:

#102  movement       81%
#188  rubbing        77%
#204  unknown        73%
```

Do not implement this at the expense of the core isolation functionality.

---

# 43. GUI LAYOUT

Main window:

```text
┌─────────────────────────────────────────────────────────────┐
│ AUDIO MICROSCOPE                                            │
├─────────────────────────────────────────────────────────────┤
│ INPUT: [device ▼]  48kHz  ● LIVE     CPU: 42%              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                     LIVE WAVEFORM                           │
│                                                             │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│                     LIVE SPECTROGRAM                        │
│                                                             │
├───────────────────────────────┬─────────────────────────────┤
│ EVENT TIMELINE                │ EVENT                       │
│                               │                             │
│ 21:41:07 FOOTSTEP             │ WHISPER                     │
│ 21:41:08 FOOTSTEP             │ confidence 73%              │
│ 21:41:09 FOOTSTEP             │                             │
│ 21:41:17 WHISPER              │ [▶ ORIGINAL]                │
│ 21:41:23 INTERFERENCE         │ [▶ ISOLATED]                │
│                               │ [▶ ENHANCED]                │
│                               │                             │
│                               │ Query: [whisper       ]     │
│                               │ [SEPARATE]                  │
├───────────────────────────────┴─────────────────────────────┤
│ AUDIO HEALTH: CLEAN    EVENTS: 183    QUEUE: 0             │
└─────────────────────────────────────────────────────────────┘
```

---

# 44. EVENT COLORS

Use visual distinctions for:

```text
speech/whisper
footstep
movement
interference
unknown
```

Do not rely solely on color.

Include text labels/icons so the UI remains understandable without color.

---

# 45. CONFIGURATION

Create:

```text
config.json
```

Example:

```json
{
  "audio": {
    "device": null,
    "sample_rate": 48000,
    "channels": 1,
    "buffer_seconds": 30
  },

  "detection": {
    "enabled": true,
    "analysis_hz": 20,
    "pre_roll_seconds": 5,
    "post_roll_seconds": 5,
    "minimum_event_seconds": 0.05,
    "maximum_event_seconds": 15
  },

  "separation": {
    "enabled": true,
    "device": "cpu",
    "workers": 1,
    "model": "audiosep"
  },

  "recording": {
    "enabled": false,
    "chunk_minutes": 15,
    "format": "wav"
  }
}
```

Do not hard-code these values throughout the application.

---

# 46. APPLICATION STATES

The application must expose clear states:

```text
STARTING
NO INPUT
LISTENING
ANALYZING
EVENT DETECTED
SEPARATING
READY
ERROR
```

Do not allow silent failure.

---

# 47. ERROR HANDLING

If the separator fails:

```text
EVENT DETECTED
ISOLATION FAILED
```

must be visible.

The original audio must remain playable.

If the model is unavailable:

```text
LIVE DETECTION: AVAILABLE
ISOLATION: UNAVAILABLE
```

Do not crash the application.

---

# 48. MODEL MANAGEMENT

Create a model abstraction:

```python
class SoundSeparator:
    def separate(
        self,
        audio,
        sample_rate,
        query
    ):
        ...
```

Then implement:

```text
AudioSepSeparator
```

The application must not directly scatter model-specific calls throughout the GUI.

Later models can implement the same interface.

---

# 49. SEPARATOR MODEL LIFECYCLE

Do not reload the model for every event.

Load it once.

Keep it resident.

Provide:

```text
MODEL: LOADED
```

in the UI.

If memory is insufficient, fail gracefully.

---

# 50. DETECTION MODEL LIFECYCLE

Likewise, load classification models once.

Never repeatedly initialize a neural network for every event.

---

# 51. CPU RESOURCE MANAGEMENT

Implement resource monitoring:

```text
CPU %
RAM
queue depth
audio callback latency
separation duration
```

Show:

```text
Separation:
2.7 sec processing
for
5.0 sec audio
```

This lets the user determine whether the system can keep up.

---

# 52. REAL-TIME GUARANTEE

The live audio path has priority over separation.

If separation is too expensive:

```text
capture continues
detection continues
separation queues
```

Never:

```text
separation blocks capture
→ dropped audio
```

---

# 53. TEST MODE

Implement a file-based test mode.

The application must be able to replace the live microphone with:

```text
test WAV
```

while running exactly the same detection/separation pipeline.

This is essential for debugging.

---

# 54. SYNTHETIC TEST SUITE

Create test audio containing mixtures such as:

```text
background noise
+
quiet whisper
```

```text
background noise
+
footsteps
```

```text
background noise
+
movement
```

```text
background noise
+
electrical hum
```

```text
background noise
+
multiple simultaneous events
```

Use these tests to verify:

```text
detection
event timing
buffer extraction
separation
playback
saving
```

---

# 55. MANUAL TEST REQUIREMENT

Before declaring the application complete, verify manually that:

### Test 1

Play a mixed recording containing a whisper.

The application must:

```text
detect candidate
display event
allow original playback
run isolation
produce isolated WAV
play isolated WAV
```

### Test 2

Play footsteps over background noise.

Verify:

```text
footstep event
isolated result
saved WAV
```

### Test 3

Introduce electrical interference.

Verify:

```text
interference event
frequency identification
isolated/interference result
```

### Test 4

Select an arbitrary waveform region manually.

Enter:

```text
whisper
```

Run separation.

Verify the result plays.

---

# 56. NO FAKE FEATURES

Do not create UI controls that do not work.

In particular, do not implement:

```text
[ISOLATE]
```

unless pressing it actually runs a separator.

Do not implement:

```text
[DETECT WHISPER]
```

as a simple volume threshold and call it AI.

Do not display invented confidence values.

Do not claim that a sound was isolated if the separator failed.

---

# 57. LOGGING

Provide structured logging.

Log:

```text
audio device
audio start/stop
buffer overruns
buffer underruns
candidate events
classification
separation start
separation completion
separation failure
file creation
model load
model errors
```

Example:

```text
21:41:17.842 EVENT candidate
21:41:17.901 CLASS whisper confidence=0.73
21:41:18.004 SEPARATION queued query="whisper"
21:41:18.015 SEPARATION started
21:41:21.883 SEPARATION complete
21:41:21.901 SAVED event_000183/separation_001/isolated.wav
```

---

# 58. PROJECT STRUCTURE

Use a structure similar to:

```text
audio_microscope/
│
├── app/
│   ├── main.py
│   │
│   ├── audio/
│   │   ├── capture.py
│   │   ├── ring_buffer.py
│   │   ├── playback.py
│   │   └── devices.py
│   │
│   ├── analysis/
│   │   ├── features.py
│   │   ├── noise_floor.py
│   │   ├── transient_detector.py
│   │   ├── event_tracker.py
│   │   └── interference.py
│   │
│   ├── detection/
│   │   ├── classifier.py
│   │   ├── whisper.py
│   │   ├── footsteps.py
│   │   └── movement.py
│   │
│   ├── separation/
│   │   ├── base.py
│   │   ├── audiosep.py
│   │   └── manager.py
│   │
│   ├── events/
│   │   ├── event.py
│   │   ├── store.py
│   │   └── metadata.py
│   │
│   ├── gui/
│   │   ├── main_window.py
│   │   ├── waveform.py
│   │   ├── spectrogram.py
│   │   ├── event_list.py
│   │   ├── event_detail.py
│   │   └── audio_player.py
│   │
│   ├── config.py
│   └── logging_config.py
│
├── models/
│
├── events/
│
├── recordings/
│
├── tests/
│
├── config.json
├── requirements.txt
├── README.md
└── run.sh
```

---

# 59. TESTING

Write automated tests for:

```text
ring buffer
noise floor
event segmentation
FFT/features
interference detection
metadata
event persistence
audio extraction
separator interface
configuration
```

At least one integration test must execute:

```text
test WAV
→ candidate detection
→ event creation
→ separation
→ output WAV
```

---

# 60. PERFORMANCE TEST

Create a benchmark command:

```bash
python -m app.benchmark
```

It must report:

```text
audio duration
processing time
real-time ratio
CPU utilization
RAM usage
```

Example:

```text
Audio:            10.0 sec
Processing:        4.8 sec
Real-time ratio:   0.48x
CPU:              82%
RAM:             2.1 GB
```

---

# 61. COMMAND-LINE MODES

Implement:

```bash
./run.sh
```

for GUI/live mode.

Also:

```bash
python -m app.test input.wav
```

for detection testing.

And:

```bash
python -m app.separate input.wav --query "a whisper"
```

for direct separation testing.

This makes debugging much easier.

---

# 62. DIRECT SEPARATION CLI

The direct separation command must:

1. Load WAV.
2. Load separator.
3. Run query.
4. Save isolated WAV.
5. Print processing time.
6. Exit with nonzero status on failure.

Example:

```bash
python -m app.separate \
    input.wav \
    --query "a distant whisper" \
    --output isolated.wav
```

---

# 63. GUI INSTALLATION

Provide a setup script.

It must:

1. Verify Python.
2. Create virtual environment.
3. Install dependencies.
4. Verify PyTorch CPU operation.
5. Verify audio device access.
6. Verify model availability.
7. Run a small test inference.
8. Report failures clearly.

Do not silently install CUDA packages.

---

# 64. FIRST-RUN DIAGNOSTICS

On startup perform:

```text
Python check
PyTorch check
CPU inference check
audio-device check
sample-rate check
model check
write-permission check
```

Display a diagnostics screen if something fails.

---

# 65. DOCUMENTATION

README must explain:

```text
installation
audio device setup
model setup
CPU requirements
starting application
testing
recording
event detection
isolation
saving
troubleshooting
```

Include exact commands.

---

# 66. IMPORTANT MODEL CAVEAT

The separator must be evaluated against actual environmental audio.

Do not assume that a model designed for generic source separation will perfectly isolate every whisper or footstep.

The application therefore needs to preserve:

```text
original
isolated
enhanced
```

so the user can compare the result.

---

# 67. DEVELOPMENT ORDER

Implement in this exact order.

## PHASE 1 — AUDIO

Build:

```text
device enumeration
live capture
ring buffer
waveform
audio playback
```

Verify that live audio works.

---

## PHASE 2 — DSP

Implement:

```text
RMS
FFT
spectrogram
noise floor
spectral features
```

Verify the GUI displays live analysis.

---

## PHASE 3 — EVENT ENGINE

Implement:

```text
candidate detection
event segmentation
pre/post roll
event timeline
```

Verify events are correctly extracted.

---

## PHASE 4 — CLASSIFICATION

Implement:

```text
whisper
footstep
movement
interference
unknown
```

Do not proceed until classifications are producing real model outputs rather than placeholders.

---

## PHASE 5 — SOURCE SEPARATION

Integrate:

```text
AudioSep
```

behind the separator interface.

Verify:

```text
WAV
→ query
→ isolated WAV
```

before integrating it into the GUI.

---

## PHASE 6 — EVENT ISOLATION

Connect:

```text
detector
→ event buffer
→ separator
→ isolated result
```

---

## PHASE 7 — GUI

Implement:

```text
event list
event detail
original playback
isolated playback
enhanced playback
manual region selection
manual query
save
```

---

## PHASE 8 — RECORDING

Add:

```text
continuous raw recording
event recording
metadata
```

---

## PHASE 9 — PERFORMANCE

Measure:

```text
CPU
RAM
capture latency
analysis latency
separation latency
queue depth
```

Optimize only after measuring.

---

# 68. DEFINITION OF DONE

The application is NOT complete when:

```text
"footstep detected"
```

appears on screen.

It is complete when this workflow works:

```text
LIVE AUDIO
     ↓
quiet sound occurs
     ↓
application detects candidate
     ↓
event appears
     ↓
user selects event
     ↓
user presses ISOLATE
     ↓
separator runs
     ↓
isolated WAV is produced
     ↓
user presses PLAY ISOLATED
     ↓
user can actually hear the separated result
     ↓
user can save it
```

The same workflow must work for:

```text
whispers
footsteps
movement
interference
unknown sounds
```

and for manually selected regions.

---

# 69. FINAL ACCEPTANCE TEST

Do not report completion until the following can be demonstrated from a clean installation:

```text
[PASS] live microphone capture
[PASS] continuous waveform
[PASS] continuous spectrogram
[PASS] rolling buffer
[PASS] adaptive noise floor
[PASS] candidate event detection
[PASS] event timeline
[PASS] whisper candidate
[PASS] footstep candidate
[PASS] movement candidate
[PASS] interference detection
[PASS] unknown event
[PASS] automatic event extraction
[PASS] manual region selection
[PASS] query-based separation
[PASS] isolated WAV generation
[PASS] isolated playback
[PASS] original playback
[PASS] enhanced playback
[PASS] WAV saving
[PASS] metadata saving
[PASS] asynchronous separation
[PASS] CPU-only operation
[PASS] capture continues while separating
[PASS] separator failure does not destroy original audio
```

If one of these does not work, identify it as incomplete rather than masking the failure with a simulated UI result.

---

# 70. MOST IMPORTANT REQUIREMENT

This application is an **audio investigation and isolation tool**, not merely an audio event classifier.

The central user action is:

```text
I HEARD SOMETHING.
FIND IT.
ISOLATE IT.
LET ME HEAR IT.
LET ME SAVE IT.
```

Everything in the architecture must support that workflow.

The original audio must always remain available, and every processed result must be traceable back to the original time range.

Build the system so that a future model can be swapped in without rewriting the application.

Start with the working end-to-end path:

```text
audio input
→ buffer
→ event
→ isolate
→ play
→ save
```

Then improve classification and detection accuracy.

Do not spend the first implementation cycle building an elaborate GUI around a separator that has not yet been proven to produce usable isolated audio.
