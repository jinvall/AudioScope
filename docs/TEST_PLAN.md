# Test Plan

## 1. Audio Capture

Verify:

- input device opens
- audio arrives
- correct sample rate
- correct channel count
- ring buffer receives data
- capture survives normal runtime

---

# 2. Ring Buffer

Test:

- write
- read
- wraparound
- pre-roll extraction
- arbitrary range extraction
- concurrent access

---

# 3. DSP

Test:

- RMS
- peak
- spectral flux
- band energy
- spectral centroid
- noise floor

Use known synthetic signals.

---

# 4. Event Detection

Test:

- quiet environment
- sudden impulse
- sustained sound
- repeated transients
- overlapping events
- low-level event

---

# 5. Whisper

Create test mixtures containing:

    background + whisper

Verify:

- candidate event occurs
- classification is possible
- event window contains the whisper
- separation job can be created

---

# 6. Footsteps

Create:

    background + footsteps

Verify:

- events are detected
- multiple steps can be grouped
- event window includes the sequence

---

# 7. Movement

Test:

    background + rubbing
    background + scraping
    background + object movement

Verify candidate detection.

---

# 8. Interference

Test:

- 50 Hz
- 60 Hz
- harmonics
- clipping
- discontinuity
- broadband artifact

---

# 9. Unknown

Create a sound that does not match known categories.

Verify:

    unknown

is allowed.

---

# 10. Source Separation

Verify independently that:

1. input WAV loads
2. model loads
3. query is accepted
4. inference executes
5. output WAV is generated
6. output WAV is valid
7. output can be played

---

# 11. Manual Separation

Test:

1. select waveform region
2. enter query
3. submit
4. separation executes
5. output appears
6. output plays
7. output saves

---

# 12. Multiple Separations

Run multiple queries on the same event.

Verify that outputs do not overwrite each other.

---

# 12a. Event Fingerprint and Review Layer

Added with Phase 3. Implemented in `tests/test_fingerprint.py`.

Fingerprint creation:

- a completed event produces a fingerprint
- the fingerprint is deterministic: identical measurements give an identical result
- an event whose optional measurements are missing still fingerprints
- an event whose level series is empty does not crash generation

Versioning:

- `fingerprint_version`, `normalisation_version` and `detector_version` are present
- fingerprints from different schema versions are refused by similarity
- a lookup skips records from other versions rather than scoring them
- a fingerprint survives a JSON round trip unchanged

Compactness:

- the fingerprint is the same size for a 1 s and a 60 s event
- the serialised payload stays under 4 kB
- a 60x longer event does not produce a 60x fingerprint
- the moment accumulators retain a fixed number of numbers regardless of length

Normalisation:

- every normalised component is bounded to 0..1
- very different raw levels and durations both map into range
- the vector length matches the declared field list

Persistence:

- an event is stored and retrieved with its fingerprint intact
- an event whose fingerprinting failed is still stored, without one
- records and fingerprints survive closing and reopening the database
- creating the schema twice is idempotent
- similarity lookup returns the closest stored event first

Annotation:

- annotating an event does not change any detector measurement
- `saved` is not a confirmation and invents no label
- a label is optional
- annotations are append-only and the history is preserved
- filtering by decision uses the *current* decision, not a superseded one
- filtering by user label works, and the distinct labels are listable
- counts by decision include unreviewed events

Negative examples:

- a rejected event is retained with its metadata and fingerprint
- rejecting does not delete the record

Real-time behaviour:

- submitting to the writer does not block and does not raise
- the writer drains its queue on stop
- a saturated queue is counted, and stored + dropped equals submitted
- a fingerprint failure does not lose the event record
- submitting to a stopped writer returns false rather than blocking
- a detected event reaches the index with a fingerprint, end to end

Genericity:

- the fingerprint, database and persistence modules contain no
  application-specific vocabulary

---

# 13. Failure Tests

Force:

- invalid model
- missing input
- invalid query
- disk error
- separator exception

Verify that original audio remains intact.

---

# 14. CPU Test

Run without GPU.

Verify:

- capture works
- analysis works
- GUI works
- separation works through CPU path

---

# 15. Long-Running Test

Run continuous monitoring for several hours.

Watch for:

- memory growth
- queue growth
- dropped audio
- dead threads
- GUI degradation
- corrupted recordings

---

# 16. Acceptance Test

The final end-to-end test is:

    microphone
      ↓
    live audio
      ↓
    event
      ↓
    event captured
      ↓
    separation query
      ↓
    source separation
      ↓
    isolated WAV
      ↓
    playback
      ↓
    save

This must work without manually editing generated files.
