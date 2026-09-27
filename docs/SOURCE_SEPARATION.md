# Source Separation

## 1. Importance

Source separation is one of the primary features of Audio Microscope.

The purpose is to isolate a requested sound from a mixed recording.

This is NOT equivalent to:

- volume adjustment
- equalization
- bandpass filtering
- spectral gating
- ordinary noise reduction
- VAD

Those techniques may be useful as supporting processing, but they do not replace source separation.

---

# 2. Query-Based Separation

The system should support natural-language queries such as:

    "a whisper"
    "a quiet voice"
    "footsteps"
    "a person moving"
    "a knock"
    "scraping"
    "mechanical noise"
    "electrical interference"

The query must be passed to a real source-separation model.

---

# 3. Model Integration

Implement the separator behind a stable interface.

Example conceptual interface:

    separator.separate(
        audio_path,
        query,
        output_directory
    )

The concrete implementation may use AudioSep or another appropriate supported model.

The model integration must use its real documented API.

Do not invent an API.

---

# 4. CPU Operation

The separator must have a CPU-compatible execution path.

GPU acceleration may be optional.

The application must remain functional without CUDA.

---

# 5. Separation Lifecycle

    user selects event
          ↓
    user enters query
          ↓
    create separation job
          ↓
    queue job
          ↓
    worker loads/reuses model
          ↓
    run separation
          ↓
    validate output
          ↓
    post-process conservatively
          ↓
    save isolated WAV
          ↓
    publish result
          ↓
    UI enables playback

---

# 6. Model Loading

Do not reload the model for every tiny operation if the model can safely remain resident.

However, model memory usage must be monitored.

If model memory is too large, implement controlled lifecycle management.

---

# 7. Job Queue

Separation jobs must be asynchronous.

A separation job should contain:

    event_id
    source_path
    query
    requested_time
    model_name
    output_directory

---

# 8. CPU Protection

Only a limited number of separation jobs should run simultaneously.

Initial default:

    1 worker

This is intentional.

The live audio pipeline has higher priority.

---

# 9. Output Validation

Do not mark a job successful merely because the model returned.

Validate:

- file exists
- file can be opened
- duration is nonzero
- sample rate is valid
- samples exist
- output is not NaN/Inf
- output is not catastrophically clipped

---

# 10. Raw Output

The direct model output should be preserved.

Example:

    separation_001/
        isolated.wav

Do not overwrite it with enhancement.

---

# 11. Enhanced Output

If enhancement is enabled:

    separation_001/
        isolated.wav
        enhanced.wav

The enhanced version is optional.

The raw isolated result remains the authoritative separation output.

---

# 12. Multiple Queries

The same event may be processed repeatedly.

Example:

    event_000042

    separation_001:
        "a whisper"

    separation_002:
        "a person speaking"

    separation_003:
        "background mechanical noise"

Each output is independent.

---

# 13. Manual Separation

Manual region selection must create a normal separation job.

Do not implement manual isolation as a separate fake path.

The selected region should enter the same separation infrastructure as automatically detected events.

---

# 14. Failure

If separation fails:

- retain original.wav
- retain metadata
- record the error
- return failure status
- keep the GUI responsive

A failed separation must never destroy the original event.

---

# 15. Processing Metrics

Record:

    input_duration
    processing_duration
    realtime_ratio

For example:

    input = 10 seconds
    processing = 40 seconds

    realtime_ratio = 4.0

This is useful for CPU performance evaluation.

---

# 16. No Fake Separation

The following is NOT acceptable:

    low-pass(original)
    high-pass(original)
    bandpass(original)
    noise_gate(original)

followed by labeling the result:

    "isolated.wav"

Those operations can be offered as enhancement tools, but they are not source separation.

---

# 17. As Built

Sections 1-16 are the requirements. This is what exists, and where.

## 17.1 The model

**AudioSep** (`ncsoft-ailab`/`Audio-AGI`), vendored in `third_party/AudioSep`
with its two checkpoints, running in its own `venv-sep` (Python 3.10, torch
2.5.1+cpu, transformers 4.28.1).

A CLAP text encoder turns the query into an embedding; a ResUNet30 estimates
the matching waveform from the mixture. The application calls the project's own
`build_audiosep()` from its `pipeline.py` and then its own inference path —
nothing about the separation is reimplemented here (section 3).

Measured on this host (4 cores, no GPU), from `tools/separation_check.py`:

| | |
|---|---|
| Model load | 35 s (resident afterwards) |
| Resident memory | 4.5 GB |
| Cold separation of 2 s | 109 s (8.8x realtime, includes the load) |
| Warm separation | ~3-9x realtime, depending on duration |
| Correlation of output with the input mixture | **-0.013** |

That last row is the acceptance evidence. A bandpass or a spectral gate of the
input correlates with the input at ≈1.0; this does not correlate at all.

## 17.2 Two pinned versions, and why

The vendored model is from 2023 and two of its dependencies have since moved:

* **torch 2.5.1**, not current. torch 2.6 changed `torch.load` to
  `weights_only=True`, which refuses this checkpoint. Pinning the last release
  with the old default was preferable to patching vendored code.
* **transformers 4.28.1 / huggingface_hub 0.16.4**, as the model's own
  `environment.yml` specifies. Later versions dropped the
  `text_branch.embeddings.position_ids` buffer that the CLAP checkpoint
  contains.

Documented here because the next person to see an old torch pin will assume it
is neglect.

## 17.3 Process isolation

The model does not run in the application process. It runs in a child process
(`app/separation/runner_main.py`, launched by `app/separation/model.py`)
reached over a newline-delimited JSON protocol.

This is the CPU-budget decision the spec asks for, made structurally rather
than by throttling. A 4.5 GB model with a four-thread torch pool inside the
capture process would compete with the audio callback for cores and memory
bandwidth, and the callback is the one thing that must never be late. A worker
thread could promise nothing. The child is capped at
`SeparationConfig.torch_threads` (default 3, one core left for capture) and
`nice`d, and it runs with `CUDA_VISIBLE_DEVICES=""`.

Two operational details that are load-bearing, both found by running it:

* **The child's working directory must be the model repository.** The CLAP
  encoder resolves its own weights through the *relative* path
  `checkpoint/music_speech_audioset_epoch_15_esc_89.98.pt`, so it finds them
  only from that root - which is how the project's own inference example runs
  it. Otherwise it reports "pretrained weights not found" while the 2.3 GB file
  sits right there.
* **The protocol lives on its own file descriptor.** AudioSep, transformers and
  timm all print to stdout, and one extra line makes the parent's JSON parse
  fail. `os.dup2(2, 1)` moves stdout to stderr and the protocol keeps a
  duplicated descriptor, so no dependency's printing can corrupt a response.
  The parent additionally skips non-JSON lines, because the alternative is
  losing a 35-second load to a stray `print`.

## 17.4 Interface

    separator.available()            # can it run, and if not, why
    separator.unavailable_reason()   # the reason, in full
    result = separator.separate(job) # SeparationResult

`app/separation/separator.py`. `UnavailableSeparator` is a real implementation
that reports a reason and writes nothing, so a missing model degrades to a
message instead of an exception at the GUI - and never to a filtered copy.

## 17.5 Output layout

    event_000042/
      original.wav
      separation_001/
        isolated.wav      <- authoritative, never enhanced over
        enhanced.wav      <- optional
        region.wav        <- only for a manually selected region
        metadata.json
      separation_002/
        ...

Attempt numbers are allocated by scanning, so a number is never reused
(sections 10-12). With the CLI's `--output` there is no attempt directory: the
file goes exactly where the caller asked, and the metadata and enhanced copy
sit beside it under their own names.

`region.wav` is **kept** on purpose. It is the exact audio the model saw, and
without it a reviewer cannot tell which part of a long recording a separation
came from, or reproduce the run.

## 17.6 Validation

`app/separation/validate.py` implements section 9. A rejected output is
**deleted rather than published**, so a clipped or non-finite separation is
never offered for playback.

The distinction that matters: a very quiet result is a *valid* result, not a
failure. If the query names something genuinely absent, a low-level file is the
correct answer, and it is recorded with a note rather than an error. What is
rejected is empty, unreadable, non-finite, wrong-rate, truncated, or clipped -
the failures that mean a broken run.

## 17.7 Enhancement

`app/separation/enhance.py`, section 11. DC removal, normalisation to a
`-3 dBFS` peak, and a clip guard that is a no-op in normal operation. Silence
is left alone, because normalising digital silence manufactures a signal and
every later measurement would describe the gain rather than the audio. The
applied gain is recorded in the metadata, so a listener comparing the two files
can see exactly what changed.

## 17.8 Manual separation

Section 13. Dragging a region in the waveform creates an ordinary
`SeparationJob` with `origin = manual_region`; there is no second code path.
The region is expressed in seconds over the source, and the cut is written into
the attempt directory. A collapsed drag means "no region" and the whole event
is separated instead.

## 17.9 Metrics

Section 15. `input_seconds`, `processing_seconds`, `realtime_ratio`,
`model_load_seconds`, `model_rss_mb` and the applied enhancement gain all reach
`metadata.json`, and the panel shows the ratio on every attempt row. A user
asking for a fourth separation deserves to know what the first three cost.

A first-job-after-cold-start carries an explicit warning that the load is
included in the reported time, rather than quietly inflating the ratio.

## 17.10 The events root

The controller must be told which event tree the capture pipeline wrote. It
defaults to the project's `events` directory, which is right for a default run
and wrong for every run that passed `--events`.

Getting this wrong is worse than a misplaced file: the attempt directory is
found by searching for a day directory containing the event id, and an
unrelated event with the same id in the project tree will match. An attempt
would then be written into that event's evidence directory. The separator now
prefers the source file's own parent - which *is* the event directory - and
`tools/separation_check.py` plus three unit tests cover it.

The model and the events are two independent roots, and conflating them makes
separation report itself unavailable whenever events live outside the project.

## 17.11 Running it

    # one file, one query
    ./venv/bin/python -m app.separate input.wav -q "a whisper" -o isolated.wav

    # is it installed?
    ./venv/bin/python -m app.separate --preflight

    # an event, into its own separation_NNN directory
    ./venv/bin/python -m app.separate input.wav -q footsteps --event event_000042

    # a region, in seconds
    ./venv/bin/python -m app.separate input.wav -q scraping --start 1.2 --duration 2.0

Exit codes: 0 written, 2 usage/config, 3 model unavailable, 4 no usable
result. In the window: the **Separate** tab.

## 17.12 Verifying

    ./venv/bin/python tools/separation_check.py

Proves the whole AGENTS.md section 26 chain with the real model - select,
query, run, create, play, save - plus multiple queries without overwriting and
a manually selected region. Minutes of CPU, so it is a tool rather than a test.

`tests/test_separation.py` covers the application layer against a fake model
(78 tests, always run), because every bug found while building this was in that
layer and none was in the model. Two further tests need the real model and are
gated behind `AUDIOMICROSCOPE_SEPARATION_TESTS=1`.

## 17.13 Known limits

* **A single file cannot be made to fit** `max_input_seconds` (default 30 s,
  about 2.5 minutes of CPU). Longer events must be separated by region; the
  refusal says so.
* **Separation runs at the model rate, 32 kHz**, resampled once at the model
  boundary. Recorded in metadata so a result is never mistaken for 48 kHz
  measurement.
* **The redundancy and the model are independent.** Separation does not feed
  back into detection or the noise floor; a whisper still has to be detected
  before it can be separated.
* **The classifier has still never heard a real whisper.** The model will
  happily separate one from a recording; whether the detector ever hands it
  such a recording is a separate, unproven question (see STATE.md).
* **`enhanced.wav` is not listened to as part of acceptance.** The raw
  `isolated.wav` is authoritative and is what the A/B comparison and the save
  handler use.
