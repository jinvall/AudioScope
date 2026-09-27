# Data and Storage

## 1. Event Storage

Events are stored independently.

Recommended:

    events/
      YYYY-MM-DD/
        event_NNNNNN/

An event directory is the durable record: the original audio and the full
detector metadata. Alongside it, Phase 3 keeps a queryable *index* so events
can be reviewed and matched without parsing the tree. The index is derived
data and can be rebuilt; the directory is the record of truth.

See sections 10 to 13.

---

# 2. Event Directory

Example:

    event_000123/
        metadata.json
        original.wav

        separation_001/
            isolated.wav
            metadata.json

        separation_002/
            isolated.wav
            enhanced.wav
            metadata.json

---

# 3. Original Audio

original.wav must contain the event audio extracted from the original capture.

It must not be enhanced or normalized destructively.

---

# 4. Metadata

Example conceptual structure:

    {
        "event_id": "...",
        "timestamp": "...",
        "duration": 10.0,
        "classification": "...",
        "confidence": null,
        "noise_floor": null,
        "peak": null,
        "sample_rate": 48000,
        "channels": 1,
        "device": "...",
        "separations": []
    }

Only populate fields with real values.

---

# 5. Separation Metadata

Each separation directory should record:

- query
- model
- source event
- creation timestamp
- processing duration
- input duration
- output file
- status
- error if applicable

---

# 6. Continuous Recording

Optional continuous recording uses chunked files.

Default:

    15 minutes per file

The recording should be lossless.

---

# 7. Naming

Avoid filenames based solely on timestamps.

Use stable event IDs.

Example:

    event_000001

This makes references stable.

---

# 8. Retention

Retention policy should be configurable.

Never automatically delete event evidence without an explicit configured policy.

**As built**, retention is implemented as a byte-capped cache over stored
audio, and section 14 describes it. The governing separation is that the
*fingerprint record* is kept indefinitely and only the *audio file* is
disposable, under an explicit configured policy.

---

# 9. Corruption Protection

Write output to a temporary filename first.

After successful close/validation:

    temporary → final

This reduces the chance of leaving corrupt final files after an interrupted write.


# 10. Event Index (SQLite)

Phase 3 adds a local SQLite index so events can be listed, filtered and
compared without walking the event tree. It is a derived store: deleting it
loses nothing that is not recoverable from the event directories.

    events.db

Tables:

    events(event_id, timestamp, start_seconds, end_seconds, duration,
           audio_path, detector_version, fingerprint_version,
           segmentation_reason, classification, created_at, metadata_json)

    event_fingerprints(event_id, fingerprint_version,
                       normalisation_version, detector_version,
                       vector_json, payload_json, created_at)

    annotations(id, event_id, decision, label, confidence,
                notes, created_at)

WAL is enabled so a read never blocks the writer. The writer runs on its own
thread; see `docs/ARCHITECTURE.md` section 3 and `docs/EVENT_DATABASE.md`
section 8.

There is deliberately no separate feature table. The compact feature summaries
are already the contents of the fingerprint payload, and a second copy would
drift. The full detector measurements, which the fingerprint deliberately does
not carry, live in `metadata.json`.

---

# 11. Fingerprints

Every completed event can produce a compact, versioned fingerprint: 23
normalised comparison fields plus a 16-point energy envelope, about 1.5 kB of
JSON, for a 1 s event or a 5 minute one.

Three versions are recorded, and all three are part of the stored contract:

    fingerprint_version     the field set
    normalisation_version   the normalisation scheme
    detector_version        which build produced the record

A version bump means a new schema. Fingerprints from different
`fingerprint_version` or `normalisation_version` are **not** comparable, and
similarity raises rather than returning a meaningless number.

The fingerprint is derived entirely from measurements the detector already
makes. Nothing is recomputed from the audio and no additional transform is
performed. Field-by-field provenance is in `docs/EVENT_DATABASE.md` section 1.

The fingerprint is also written into the event's `metadata.json` and retained
for rejected events, because a negative example is as useful as a positive one.

---

# 12. Annotations

Human decisions about events are stored separately from detector measurements
and never overwrite them.

    unreviewed   nobody has looked
    saved        the user wants to keep it
    rejected     not wanted; the record is retained
    uncertain    the user could not decide
    confirmed    the user is confirming it, and may supply a label

`saved` is not a claim about what the event is. A label is a separate decision
and is never required.

Annotations are append-only, so the review history of an event is preserved.
The current state is the most recent annotation; the current label is the most
recent non-null one.

---

# 13. Rejected Events and Audio Retention

Retention of the *record* and retention of the *audio* are separate decisions.

A rejected event keeps its metadata and fingerprint, because
`detector output -> user decision -> positive and negative examples` is the
point of collecting them. Deleting rejected events destroys the negative half
of a future training set.

Whether the original WAV of a rejected event is still on disk is a file-store
policy, not part of this decision. It is implemented (section 14), and the
decision it records is not one retention is allowed to act on: a rejected
event is protected from automatic eviction exactly as a confirmed one is,
because a negative example is as valuable as a positive one.

Never automatically delete event evidence without an explicit configured
policy; this continues to apply to the audio of saved and confirmed events.

---

# 14. Audio Retention as Built

`app/events/retention.py`, configured by `AudioRetentionConfig` in
`app/config.py`. `tools/retention_check.py` verifies it against the running
pipeline.

## 14.1 The invariant

    FINGERPRINT HISTORY   long-lived, effectively unbounded
    AUDIO HISTORY         bounded by bytes

Every event keeps its fingerprint, its full detector measurements, its
classification and its entire append-only review history. Only the audio file
is disposable, and only when bytes demand it.

Evicting audio never deletes a record. It sets `audio_available = 0` and
`audio_path = NULL`, records `audio_evicted_at`, and leaves everything else
alone. The event stays listable, comparable by fingerprint, and reviewable.
`StoredEvent.is_playable` requires both the flag and a path, so a row that
claims availability without a file cannot offer playback.

## 14.2 Byte caps, not event counts

    per_class_audio_cap = 500 MB (524288000)
    total_audio_cap     =   5 GB (5368709120)

Both are configurable, and both apply to stored audio only — never to
fingerprints, feature vectors, metadata, indexes or classification records.

A count-based limit would cap the wrong quantity: 100 five-second events and
100 thirty-second events are the same "100 events" and very different
amounts of storage.

Sizes may be written as `500MB`, `5GB`, `1.5GiB` or plain byte counts.
`MB` means 1024², matching the value the configuration example states
(`500MB` → `524288000`); the ambiguity is available to the reader via `MiB`
rather than hidden in the parser. Internally everything is explicit bytes,
and the config file is always written with explicit byte values.

## 14.3 Account before writing

The obvious flow — write the file, check the total, then evict if over —
permits a transient overshoot of the global cap by one whole file, every
time. Instead:

1. the store computes the exact encoded size from the sample count
   (`wav_bytes_for`, float32 evidence format, no compression);
2. `reserve()` makes room **before** the bytes exist;
3. the file is written;
4. `record_written()` counts it;
5. the writer thread's `link_row()` attaches the row.

## 14.4 Eviction prefers redundancy to age

Deleting the oldest recording is the standard trick and it is wrong here: it
turns a bounded evidence cache into a FIFO archive, which after a day of
recording holds a near-identical run of the most recent sounds and none of the
variation that made the recording worth keeping.

`select_victims` evicts the most redundant audio first, ranked by
**nearest-neighbour** fingerprint distance: a small distance means the
recording has a near-duplicate among those already retained, so losing it
costs the collection least.

Nearest neighbour rather than mean distance, and the distinction is not
cosmetic. With a mean, a tight cluster of near-identical recordings and a
pair of rare ones score almost identically — every candidate has two very
close neighbours and two very distant ones, so the average is dominated by
the cross-cluster distances. Measured on a synthetic set, a mean ranks a rare
recording as more evictable than a cluster member. The nearest neighbour
separates them.

The survivors are re-scored after each removal, so the selection spreads out
across a class instead of draining one dense cluster to a single survivor.

Tie-breaks, in order: quality penalty, then file size, then age, so the
choice is deterministic. Quality uses only recorded measurements — incomplete
capture (missing pre-roll or post-roll, the strongest signal, because it
indicates the recording is *wrong* rather than merely plain), near silence,
clipping, and a duration under 0.2 s. A candidate that cannot be scored at
all is ranked last but remains reachable; a missing fingerprint is unknown,
not evidence of redundancy.

`preserve_representative_audio: false` is the degraded oldest-first mode. It
is offered because it is predictable and auditable, and it is documented as
producing a FIFO archive rather than a representative one.

## 14.5 What is protected

Events a human has judged — `confirmed`, `saved`, `rejected` and `uncertain` —
are never evicted automatically, in both directions. A confirmed example is
labelled data; a rejected one is a negative example the project deliberately
keeps. Either way, a rule-based storage policy does not get to overrule a
human decision.

`min_files_per_class` (default 1) means retention never reduces a
classification to zero audio files. A single recording larger than its class
cap is kept, with the overage reported — a storage policy is not a reason to
destroy the only copy of a class's evidence. Writes are the exception: there
an incoming file must fit, and making room for it is the entire reason room
is reserved first.

## 14.6 Accounting

Runtime usage is maintained incrementally in `RetentionAccounting`; the event
table is the durable mirror (`audio_bytes`, `audio_available`,
`audio_evicted_at`) and `retention_state` holds a snapshot. The hot path
never walks the tree.

On startup: load the snapshot, validate it against the database (the database
wins, because it is written in the same transaction as the event rows),
reconcile against the filesystem, and clean up any class already over its cap.

`reconcile()` is the safety net, run periodically and on demand. It rebuilds
usage from disk and:

* marks events whose file has vanished as unavailable;
* records files that exist with no event row — under `unclassified`, because
  ignoring an unknown file is how usage grows without bound;
* corrects accounting drift;
* enforces the caps on classes already over them.

The system does not assume its own bookkeeping is correct. A run of
`tools/retention_check.py` asserts the equality that matters: accounted bytes
equal the bytes on disk, and every row claiming available audio has a file.

### Concurrency

The event store (detection thread), the writer thread and the GUI all touch
retention. Two orderings were found to corrupt the accounting during
development, and both are now prevented structurally:

* `link_row` and `_evict` hold the manager lock across the row read, the file
  check, the accounting change and the row update. Without that, `link_row`
  could observe an available row and a present file, have eviction happen in
  between, and count bytes for a file that had just been deleted.
* An event evicted before the writer links its row has `audio_bytes IS NULL`,
  so eviction used the write-time recorded size. Trusting the null row would
  remove zero bytes while deleting megabytes of file.

`store_event` uses `ON CONFLICT DO UPDATE` rather than `INSERT OR REPLACE`,
and does not list the retention columns. `REPLACE` deletes and recreates the
row, resetting those columns to their defaults and resurrecting evicted audio
as "available".

## 14.7 Scope

The caps cover `original.wav` in the event tree. `metadata.json` is a record,
not audio. Separation outputs are derived artefacts of a query the user asked
for and are governed with the audio they belong to. Continuous recordings
(`recordings/`) are a separate series with their own policy
(`RecordConfig.retention_days`); they are not counted here.

## 14.8 Known limits

* Eviction deletes whole files. Splitting a long event across the cap would
  keep partial evidence, and a truncated recording that looks complete is
  worse than none.
* The redundancy ranking is a distance over fingerprints, so it can only
  prefer what it can measure. Validated on synthetic audio, like similarity
  itself.
* A file larger than the whole budget cannot be made to fit. It is stored
  anyway and the overage reported, because the alternative is silently
  discarding the newest event's audio.
