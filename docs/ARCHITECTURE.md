# Audio Microscope — Architecture

## 1. System Overview

The application consists of independent processing stages.

    ┌────────────────────┐
    │   Audio Capture    │
    └─────────┬──────────┘
              │
              ▼
    ┌────────────────────┐
    │   Ring Buffer      │
    └──────┬─────────────┘
           │
     ┌─────┴───────────────┐
     │                     │
     ▼                     ▼
┌──────────────┐     ┌───────────────┐
│ Real-time DSP│     │ Raw Recording │
└──────┬───────┘     └───────────────┘
       │
       ▼
┌──────────────────┐
│ Candidate Events │
└────────┬─────────┘
         │
         ▼
┌────────────────────┐
│ Event Classification│
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ Event Store         │  original.wav + metadata.json
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ Fingerprint + Index │  versioned summary -> SQLite, reviewable
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ Human Review (GUI)  │  saved / rejected / uncertain / confirmed + labels
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ Separation Manager │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ Isolated Audio     │
└─────────┬──────────┘
          │
          ▼
┌────────────────────┐
│ Playback / Save    │
└────────────────────┘

---

# 2. Process Responsibilities

## Audio Capture

Responsibilities:

- open input device
- receive samples
- convert to internal format
- write to ring buffer
- publish lightweight analysis blocks

Must not perform expensive work.

---

## Ring Buffer

Responsibilities:

- maintain recent audio history
- provide pre-roll
- provide post-roll
- provide arbitrary region extraction
- remain thread-safe

Default capacity:

    30 seconds

---

## Analysis Worker

Responsibilities:

- calculate DSP features
- update noise floor
- detect candidate events
- publish event candidates

It must not perform source separation.

---

## Event Tracker

Responsibilities:

- merge nearby candidate frames
- determine event start
- determine event end
- avoid excessive duplicate events
- create event objects

---

## Classifier

Responsibilities:

- classify candidates
- attach evidence
- return unknown when appropriate

Classification must never block capture.

---

## Separation Manager

Responsibilities:

- queue separation jobs
- manage model lifetime
- invoke the actual source separation implementation
- create output files
- report errors
- measure processing time

---

# 3. Threading Model

Recommended structure:

### Audio thread

Highest priority.

Responsibilities:

- capture
- minimal conversion
- ring-buffer write

### Analysis worker

Responsibilities:

- DSP
- candidate detection
- event tracking

### Event writer

One background thread, added in Phase 3.

Responsibilities:

- fingerprinting completed events
- writing the event index row
- never runs on the detection or analysis threads

Fingerprinting and the database write are deliberately off the detection
thread. Measured cost of the writer competing for the GIL is +0.018 ms per
analysis frame, and a write takes 1.7 ms; both are paid here rather than in
event extraction. If its queue fills, events are dropped *from the index* and
counted - the audio and full metadata are already on disk, so only the index
entry is late, and that is reported rather than hidden.

### Separation worker

Responsibilities:

- neural source separation
- post-processing
- output writing

### GUI thread

Responsibilities:

- rendering
- user interaction
- playback control

No heavy processing should execute on the GUI thread.

Added in Phase 5, and the same rule applies in reverse: the GUI performs no
analysis and writes no detector data. Waveform envelopes, spectrograms and audio
decoding are all dispatched to worker threads; the only persistence it performs
is appending a human annotation. See `docs/GUI.md` section 8.

---

# 4. Queue Model

Recommended queues:

    audio_queue
    analysis_queue
    event_queue
    writer_queue
    separation_queue
    result_queue

The queues must have bounded sizes.

Do not allow unbounded queues to consume memory during CPU overload.

`writer_queue` carries completed events from the detection thread to the event
writer. It is bounded, and overflow is counted rather than silently absorbed.

---

# 5. Priority

When the machine is overloaded:

    capture > analysis > UI > writer > separation

Source separation may be delayed.

Live audio must not be sacrificed to finish a separation job.

Persistence sits below the UI: a slow disk may delay indexing, but it may never
delay capture, and the event's audio and metadata are already safely written by
the time indexing is attempted.

---

# 6. Event Object

An event should contain at least:

    id
    start_time
    end_time
    duration
    classification
    confidence
    noise_floor
    peak
    source_device
    sample_rate
    channels

Confidence must be optional if the classifier cannot produce a meaningful calibrated value.

Phase 3 adds, alongside the above:

    fingerprint            compact, versioned summary (23 normalised fields
                           plus a 16-point energy envelope)
    fingerprint_version    the field set
    normalisation_version  the normalisation scheme
    detector_version       which build produced the record
    segmentation_reason    why the tracker ended the event
    annotations            the human's decisions, appended and kept separate

The fingerprint and the annotations are **additional** to the event, never
substituted for it. A human annotation records what the user decided; it never
overwrites what the detector measured. See `docs/EVENT_DATABASE.md`.

---

# 7. Separation Job

A separation job should contain:

    event_id
    input_audio
    query
    model
    output_directory
    requested_at

The job must produce:

    status
    output_path
    duration
    processing_time
    error

---

# 8. Separation Result

A result must reference the actual generated audio file.

The UI must not assume success merely because the job was submitted.

Only completed output should be marked playable.

---

# 9. Error Isolation

Failures should be isolated by subsystem.

A source-separation exception must not terminate:

- audio capture
- event detection
- GUI

An audio-device failure should produce a visible UI error and attempt recovery where possible.

---

# 10. Configuration

Configuration should control:

- input device
- sample rate
- channels
- ring-buffer duration
- event pre-roll
- event post-roll
- noise-floor behavior
- detector thresholds
- separation worker count
- recording settings
- output directory
- model configuration
- playback settings

Avoid scattering constants throughout the codebase.
