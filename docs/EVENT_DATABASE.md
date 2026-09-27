# Event Fingerprints and the Review Database

Phase 3 completion. Every detected event becomes a compact, versioned
**fingerprint** and a persistent record that a human can review, annotate, and
later match against.

* `app/events/fingerprint.py` — the fingerprint, its versioning, normalisation
  and distance
* `app/events/database.py` — SQLite storage, annotations, similarity lookup
* `app/events/persistence.py` — the background writer
* `app/events/cli.py` — review commands (`python -m app.events`)

    python -m app.capture --seconds 60            # detects, stores, indexes
    python -m app.events --stats
    python -m app.events --list --decision unreviewed
    python -m app.events --save event_000003
    python -m app.events --save event_000003 --user-label "door" --notes "clear"
    python -m app.events --similar event_000003

---

## 1. What a fingerprint is

A fixed-size summary of one event, built **only** from measurements the detector
already makes. There is no second pass over the audio, no new transform, and
nothing that grows with the event's length.

Size is fixed at construction: **16 envelope points and 23 normalised fields**,
about 1.5 kB of JSON, for a 1 s event or a 5 minute one.

### Fields, and where each comes from

| group | field | source |
| --- | --- | --- |
| timing | `duration` | `last_active_sample - onset_sample` |
| energy | `level_mean/std/min/max/dynamic_range` | new `RunningMoments` over `rms_db` |
| energy | `snr_peak`, `snr_mean`, `snr_std` | existing `peak_snr_db`, new moments over candidate SNR |
| energy | `presence` | existing `presence_fraction` |
| spectral | `centroid_mean/std/min/max/spread/trend` | new moments over `spectral_centroid_hz` |
| spectral | `flatness_mean/std/min/max/spread` | new moments over `spectral_flatness` |
| spectral | `bandwidth_mean` | existing `mean_bandwidth_hz` |
| spectral | `flux_max` | existing `max_flux` |
| temporal | `crest_max` | existing `max_crest` |
| temporal | `modulation_mean` | existing `mean_modulation` |
| temporal | `envelope[16]` | new bounded level series, binned |
| temporal | `onset_count`, `onset_density`, `inter_onset_mean/std/cv` | existing onset counters and intervals |
| bands | `high_to_low`, `band_snr` | existing band SNR summaries |

**Nothing is recomputed.** The four `RunningMoments` accumulators and the
bounded level series are the only additions to the detector, and they are
O(1) folds over values already in hand — four float additions and one deque
append per frame. They are filled on the detection thread, which is already
several stages behind the audio callback, and they cost no transform.

### Reuse summary

Reused unchanged: duration, peak and mean SNR, level means, crest, modulation,
mean centroid, mean bandwidth, max flux, presence, band SNR summaries, high/low
ratio, onset count, onset density, onset intervals, dominant frequency, frame
counts, truncation flag.

Added (arithmetic only, no new measurement): level/SNR/centroid/flatness
moments, centroid trend, a bounded level series for the envelope, and
inter-onset mean/std/cv derived from intervals the tracker already records.

---

## 2. Compactness

`RunningMoments` is Welford's algorithm: count, mean, M2, min, max, first, last
in seven numbers regardless of how many frames went in. `level_series` is a
`deque(maxlen=2048)`, so a long event is decimated rather than stored.

A test asserts that a 60× longer event produces a fingerprint of less than
twice the serialised size, and that the payload stays under 4 kB.

---

## 3. Versioning

Three versions are recorded per fingerprint:

| field | meaning |
| --- | --- |
| `fingerprint_version` | the field set. A change is a new version, never a silent reinterpretation |
| `normalisation_version` | the normalisation scheme. Two fingerprints are only comparable if these match |
| `detector_version` | which build produced the record |

`similarity()` raises `IncomparableFingerprints` on a version mismatch rather
than computing a distance between vectors that are not on the same scale. A
lookup skips records from other versions.

---

## 4. Normalisation

Raw values are stored because they are what a human reads. A normalised vector
is also stored, because that is what a distance is computed over — duration in
seconds, level in dBFS, centroid in Hz and flatness in 0..1 have nothing to do
with each other.

The scheme uses **fixed constants, not statistics from the current session**, so
two events recorded weeks apart remain comparable:

* duration and onset count are log-compressed (they span orders of magnitude)
* everything else maps onto 0..1 with a documented range: level over
  −100..0 dBFS, centroid over 0..8 kHz, flatness over 0..1, SNR over 0..60 dB, and
  so on

`VECTOR_FIELDS` fixes the field order, and that order *is* the comparison
contract. `DEFAULT_WEIGHTS` gives duration and level slightly more weight
because they describe coarse event shape.

---

## 5. Storage

SQLite. Local file, no server, no daemon — consistent with the existing event
directory store.

```sql
events(event_id PK, timestamp, start_seconds, end_seconds, duration,
       audio_path, detector_version, fingerprint_version,
       segmentation_reason, classification, created_at, metadata_json)

event_fingerprints(event_id PK -> events, fingerprint_version,
                    normalisation_version, detector_version,
                    vector_json, payload_json, created_at)

annotations(id PK, event_id -> events, decision, label,
            confidence, notes, created_at)      -- append-only
```

WAL is enabled, so a read (a review query, a similarity scan) never blocks the
writer.

**There is deliberately no `event_features` table.** The compact feature
summaries it would hold are already the contents of the fingerprint payload, and
a second copy would drift. The *full* detector measurements — larger, and which
the fingerprint deliberately does not carry — live in `events.metadata_json`.

### Audio retention

Retention of the audio file is separate from retention of the record. A rejected
event keeps its metadata and fingerprint, which is what makes it usable as a
negative example; whether its WAV is still on disk is a file-store policy and is
not decided here.

---

## 6. Human in the loop

The detector records what it measured. The user records what they decided. The
two are stored separately and never overwrite each other — a test asserts that
annotating an event leaves its detector classification and measurements
untouched.

| state | meaning |
| --- | --- |
| `unreviewed` | nobody has looked |
| `saved` | the user wants to keep it. **Not** a claim about what it is |
| `rejected` | not wanted. Kept as a negative example |
| `uncertain` | the user could not decide |
| `confirmed` | the user is confirming it, and may supply a label |

`saved` and `confirmed` are deliberately separate: saving means "keep this", and
a label is a separate assertion. A label is never required, and the detector
never supplies one — the classes are the user's to define.

Annotations are append-only, so the review history of an event is preserved. The
current state is the most recent one, and the current label is the most recent
non-null one.

User-supplied `confidence` is recorded separately from any detector score. The
detector's classification score lives in the metadata and is explicitly not a
confidence (`docs/DETECTION_IMPLEMENTATION.md`).

---

## 7. Similarity

A weighted Euclidean distance over the normalised vector. Deliberately just
that: no index, no clustering, no classifier.

`find_similar()` is a linear scan at about **0.07 ms per comparison**, so a
thousand-event store is a fraction of a millisecond. An index would be
premature and would need rebuilding on every fingerprint version change. When
the store outgrows a linear scan, that is the moment to add one — and the
version check is already the hook for it.

---

## 8. Real-time behaviour

Persistence is never inline. `DetectionWorker` hands a finished event to a
bounded queue and returns; a single writer thread builds the fingerprint and
writes the row.

* `submit()` is a queue put. A test drives 500 submissions and asserts the
  caller is not delayed, and that the accounting balances:
  `stored + dropped == submitted`.
* Fingerprinting happens on the writer thread, so its cost is entirely off the
  detection thread.
* If the queue is full the event is dropped *from the index* and counted. The
  audio and full metadata are already on disk in the event directory, so only
  the index entry is late — and that is recorded rather than hidden.
* A fingerprint failure does not lose the event. The row is stored without one
  and the failure is counted; a missing fingerprint is visible, never invented.
* The writer is stopped last during shutdown, after detection has drained, so
  queued events are not lost at exit.

### Measured cost

| | |
| --- | --- |
| fingerprint build, 1 s event | 176 µs |
| fingerprint build, 300 s event | 617 µs (bounded by the level series cap) |
| fingerprint build at 100 events/s | 1.8 % of a core |
| similarity, one comparison | 0.07 ms |
| SQLite write, measured | 1.7 ms |
| added detector cost per frame | +0.018 ms (0.631 → 0.649 ms) |

The detector figure is the GIL cost of the writer thread competing for time;
the database work itself is not on the detection thread at all.

---

## 9. Genericity

No application-specific terminology anywhere in `fingerprint.py`,
`database.py` or `persistence.py`, enforced by a test. A label is user-supplied
free text; the system has no built-in classes and no opinion about what the
microphone is listening for.

---

## 10. What this does not do

* **No classifier.** Similarity is a distance, nothing more.
* **No automatic labelling.** The detector's own label is a
  `possible_*` guess with an evidence list, and remains a guess.
* **No confidence from the detector.** Still `null`, for the same reason as
  before: a rule score is not a calibrated probability.

### A deliberate divergence from the original directive

`docs/OVERVIEW.md` section 39 asks for confidence as a 0-100% "model
confidence", explicitly labelled so it is never read as certainty. This
implementation reports `confidence: null` instead, and reports
`label_scores` plus a `margin` alongside the evidence.

That is an intentional disagreement, not an oversight, and it is worth stating
plainly. A percentage can only be produced by something calibrated. What exists
today is a set of hand-chosen thresholds over measured features; presenting
`0.73` from that would be exactly the cosmetic number
`AGENTS.md` section 21 and `docs/DETECTION_AND_CLASSIFICATION.md` section 8
forbid, and the directive's own label "model confidence" would be untrue,
because there is no model.

The two requirements are compatible over time rather than in conflict. When a
real calibrated model is integrated, confidence becomes a 0-100% model
confidence as section 39 specifies. Until then `null` with the reason recorded
is the honest value, and `label_scores` is what a later stage can calibrate
into one.
* **No deletion of rejected events.** They are negative examples.
* **No trained model, no GPU, no neural embedding.** CPU only, as required.
