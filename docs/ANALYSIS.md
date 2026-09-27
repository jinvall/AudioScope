# Analysis (Phase 2)

Continuous acoustic analysis: per-frame features and an adaptive noise floor.
No detection, no classification, no neural inference — those are Phases 3, 4
and 5 (tasks/PHASE_03_EVENTS.md, tasks/PHASE_04_SEPARATION.md).

Run it:

    python -m app.analyse input.wav
    python -m app.analyse input.wav --json
    python -m app.analyse input.wav --frames 20 --bands
    python -m app.analyse input.wav --spectrogram out.pgm

Analysis is **on by default** during live capture and can be turned off with
`--no-analysis`.

---

## 1. Structure

    app/analysis/
        frames.py         block-size-independent framing
        features.py       per-frame measurements
        noise_floor.py    adaptive floor and signal-to-floor ratio
        worker.py         the analysis thread
        spectrogram.py    rolling spectrogram and waveform buffers

    AudioPipeline.enable_analysis()  attaches the worker
    AudioPipeline.on_block           is the hook it consumes

The worker is started and stopped by the pipeline, so a caller that only wants
analysis never manages its lifetime.

---

## 2. Framing

`app/analysis/frames.py` cuts the stream into **40 ms frames on a 10 ms hop** by
default (1920 and 480 samples at 48 kHz), giving 100 analyses per second with
75% overlap. All values are configurable in `AnalysisConfig`.

Frames are cut on a fixed hop grid measured from the start of the stream, not
from wherever a block happens to end. That is what makes detector timing
independent of frame size, as docs/AUDIO_PIPELINE.md section 4 requires:
feeding 1024-sample blocks and 777-sample blocks produces byte-identical frames.
`tests/test_analysis_frames.py` asserts this for seven block sizes.

An incomplete tail at end of stream is **discarded and counted**, not zero-padded.
Padding would be actively harmful: a padded frame is mostly digital silence, and
feeding that to an adaptive noise floor teaches it that the room is silent.
A monitor must not invent audio it did not capture.

---

## 3. Features

Computed per frame (`FrameFeatures`):

| Feature | Notes |
| --- | --- |
| `rms`, `peak`, `rms_db`, `peak_db` | measured on the **raw** samples |
| `crest_factor` | peak / RMS; high for impulsive sounds |
| `zero_crossing_rate` | on the mean-removed signal, with a deadband |
| `dc_offset` | AGENTS.md section 12 needs it for interference analysis |
| `spectral_centroid_hz` | brightness centre of mass |
| `spectral_bandwidth_hz` | spread around the centroid |
| `spectral_rolloff_hz` | 85% of energy (configurable) |
| `spectral_flatness` | ~1 broadband noise, ~0 a pure tone |
| `spectral_flux` | half-rectified change; `None` on the first frame |
| `bands` | the nine bands below |

### Windowing is applied inside, for the spectrum only

Time-domain features are measured on the raw frame and the window is applied
only to the copy used for the FFT. A Hann window scales a full-scale sine by
about 0.5, so measuring level on the windowed frame halves the reported RMS and
doubles the crest factor. Doing the windowing inside `FeatureExtractor` removes
that whole class of mistake rather than relying on the caller.

### Zero-crossing rate is not fooled by DC or silence

Counted on the mean-removed signal, with values inside a relative deadband
treated as zero. Without the deadband, a constant DC input reports a stream of
crossings from rounding dust, and so does a silent frame.

### Bands

    20-80  80-250  250-500  500-1000  1000-2000
    2000-4000  4000-8000  8000-16000  16000-24000

Each band reports `linear`/`db` (total energy) **and** `density_linear`/
`density_db` (power per hertz). Both, not one substituted for the other: a wide
band sums more energy than a narrow one, so totals are not comparable across
bands, and the per-hertz form is what answers "is there more high-frequency
energy than usual?". A band above Nyquist, or narrower than one FFT bin, reports
zero rather than borrowing energy from a neighbour.

### A known limitation: band-edge leakage

A Hann window's main lobe is four bins wide, so a tone within one bin of a band
edge puts real energy into the neighbouring band — its window response is only
about 6 dB down one bin from centre. At 40 ms and 48 kHz the bins are 25 Hz, so a
100 Hz tone reads about 7 dB into the 20-80 Hz band.

This is inherent to the 20-50 ms frame the specification allows; fixing it needs
a longer frame or a window with a narrower main lobe and lower sidelobes. It is
asserted as a bounded property in `tests/test_analysis_features.py` so a future
change that made it worse would fail loudly rather than quietly distort a
detector's input.

---

## 4. Adaptive noise floor

`app/analysis/noise_floor.py`. Requirements are AGENTS.md section 8 and
docs/AUDIO_PIPELINE.md section 6: learn the environment, never assume a fixed
dBFS, and never let an event become the floor.

### Method

1. **Sub-window averaging.** Frame levels are grouped into 0.5 s sub-windows; a
   ring of the last 3 s supplies the raw estimate.
2. **Asymmetric smoothing.** The floor falls at 12 dB/s and rises at 1.5 dB/s, so
   a quieting room is tracked promptly and a sudden loud one is not absorbed.
3. **Transient gate.** A sub-window whose typical level is more than 9 dB above
   the floor is rejected and does not move the estimate.
4. **Sustain.** If a level stays above the gate for 3 s it is reclassified as the
   environment and the floor does follow it.

Measured behaviour:

| Situation | Result |
| --- | --- |
| Steady noise at −40 dBFS | floor −40.01 dBFS, SNR 0.12 dB, 0 false gates |
| 40 dB event, 2 s | floor moves < 0.6 dB, SNR reports ~40 dB |
| 40 dB event, sustained | floor learns the new level after the sustain window |
| Room goes quiet | floor returns to the true level within a second or two |
| Silent room | floor sits at the silence floor and does not drift |

### Two design decisions worth knowing about

**The estimate is a mean, not a minimum.** The classical minimum-statistics
approach takes the minimum over a window. That is biased here, and the bias
depends on band width: the 20-80 Hz band is three FFT bins at 48 kHz, and the
minimum of three bins sits far below the minimum of a wide band. Measured, a
minimum-based floor reported about 10 dB of phantom signal in ordinary noise.
Since the transient gate already excludes events, the mean of the accepted
sub-windows is unbiased at any band width and is what is used.

**Averaging happens in the linear power domain, not in decibels.** Levels arrive
as dB because that is how thresholds are expressed, but the mean of a log is not
the log of a mean, and the bias grows with a band's log-variance. For the
three-bin 20-80 Hz band that bias alone accounted for several dB of phantom SNR.
Energy is summed linearly and converted to dB once, at the end.

The gate compares the sub-window's *typical* level rather than its minimum, for
the same reason: the minimum of a short window of noise wanders by several dB
between sub-windows, which made the gate fire on a quiet room.

### There is no floor ceiling

An earlier design clamped the floor to a ceiling to stop a very quiet room
reporting a drifting floor. That was wrong: a floor is a measurement, and
clamping it pinned every band louder than the ceiling to that ceiling and
reported a permanent phantom SNR. The floor now follows the measurement, and a
silent room is handled by the fall rate and the silence floor.

### Reported state

`NoiseFloorState` carries the overall floor, the per-frame SNR, a per-band floor
and SNR for each of the nine bands, and two flags:

* `transient_detected` — an event is happening **now**. Computed per frame
  (`snr > transient_gate_db`), so it is meaningful on every frame rather than
  only on the rare frames where a sub-window happens to close.
* `gate_rejected` — this frame is the one where a sub-window was rejected.
  Diagnostics for the estimator itself.

A band carrying no energy does not participate in the gate and reports SNR 0,
because a band sitting at the −200 dBFS floor fluctuates numerically and would
otherwise report phantom events forever.

---

## 5. The analysis thread

docs/ARCHITECTURE.md section 3 and AGENTS.md section 2.3: analysis runs on its
own thread, never in the capture callback, and must never be able to block the
stages above it.

* `AnalysisWorker.submit` is called from the capture pump thread and does nothing
  but a non-blocking put on a bounded queue. A test drives 2000 submits and
  asserts the caller is not delayed.
* Under overload, blocks are dropped, never buffered without limit, and the loss
  is **counted and reported** — `blocks_dropped`, and a `gap` flag on affected
  results so a downstream detector can tell that evidence is missing rather than
  concluding nothing happened.
* An exception in analysis or in a consumer callback is recorded and the thread
  continues. It never takes the capture path down.
* The result queue is deliberately small so a slow GUI cannot make the worker
  back up. Offline runs opt into `lossless=True` and a large `history_size`,
  because "a third of the file was skipped" is not an acceptable analysis
  result.

### Cost

Measured on this machine, 100% analysis coverage at 48 kHz:

| | |
| --- | --- |
| per frame | ~0.9 ms off-line, ~1.4-1.7 ms while capturing |
| realtime ratio | 0.09-0.17 of one core |
| headroom | 6-11x below the 1.0 budget |

docs/PERFORMANCE.md section 6 defines the ratio as processing time over audio
duration; below 1.0 leaves the machine room for the GUI and, later, separation.

---

## 6. Spectrogram and waveform

`SpectrogramBuffer` holds a fixed ring of dB-scaled magnitude columns, and
`WaveformBuffer` holds a min/max envelope. Columns share the analysis hop, so a
column index means the same instant in the waveform, the spectrogram and the
event timeline (docs/AUDIO_PIPELINE.md section 11).

Each spectrogram column is normalised to its own peak. That is right for a
scrolling live view — the spectrum shape is visible however quiet the moment was
— but it makes a *static* image flat, because every column reaches full
brightness. The PGM export therefore rescales against the global maximum so the
time structure is visible. Pass `renormalise=False` to keep per-column scaling.

`write_pgm` needs nothing but numpy, so the spectrogram can be inspected on a
machine with no plotting library. A `.png` path exists and uses matplotlib when
it is installed, falling back to PGM when it is not.

The waveform buffer stores peaks and troughs rather than means: a mean makes an
AC-coupled signal look flat and hides transients, which is the opposite of what
a monitor needs to show.

---

## 7. What is deliberately absent

* **No candidate detection, event tracking or classification.** Phase 3. The
  worker exposes features and floors only; nothing here decides that an event
  happened.
* **No confidence values.** These are measurements, not opinions
  (docs/DETECTION_AND_CLASSIFICATION.md section 8).
* **No source separation.** Phase 4.
* **No GUI.** Phase 5. The buffers are populated and tested; drawing them is
  Phase 5's job.
