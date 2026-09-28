"""Human-readable presentation of events.

Two rules from the brief shape this module:

* **Human-readable first.** The list and inspector show duration, level, SNR and
  decision in plain language. Technical fields - fingerprint internals, version
  numbers, raw metadata keys - are grouped separately and never dominate.
* **Nothing is invented.** Every value here is read from the stored event or its
  fingerprint. Where a measurement is absent the formatter says so rather than
  substituting a plausible number, and the detector's ``confidence`` is printed
  as unavailable when it is null rather than as 0%.

No sound-class vocabulary appears anywhere in this module. A user label is shown
verbatim; the interface never assumes what a sound is.
"""

from __future__ import annotations

import datetime
import math
from dataclasses import dataclass, field
from typing import Optional

from ..events.database import Decision

#: Shown when the backend reports a measurement as unavailable.  The backend
#: uses -120 dBFS as its floor for level-like quantities, so this is a real
#: sentinel rather than a guess.
UNAVAILABLE = "-"

_LEVEL_FLOOR = -119.0

#: Human wording for each backend decision.  The values stay the backend's
#: Decision enum values; only the presentation is human.
DECISION_LABELS = {
    Decision.UNREVIEWED.value: "Not reviewed",
    Decision.SAVED.value: "Saved",
    Decision.REJECTED.value: "Rejected",
    Decision.UNCERTAIN.value: "Uncertain",
    Decision.CONFIRMED.value: "Confirmed",
}

#: Short column text for the event list.
DECISION_SHORT = {
    Decision.UNREVIEWED.value: "new",
    Decision.SAVED.value: "saved",
    Decision.REJECTED.value: "rejected",
    Decision.UNCERTAIN.value: "?",
    Decision.CONFIRMED.value: "confirmed",
}


def format_decision(value) -> str:
    """Human wording for a decision, tolerating a raw string."""
    text = str(getattr(value, "value", value) or Decision.UNREVIEWED.value)
    return DECISION_LABELS.get(text, text)


def decision_short(value) -> str:
    text = str(getattr(value, "value", value) or Decision.UNREVIEWED.value)
    return DECISION_SHORT.get(text, text)


# ----------------------------------------------------------------------
# Value formatting
# ----------------------------------------------------------------------
def level(value, digits: int = 1) -> str:
    """Format a dBFS value, reporting the backend's floor as unavailable."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return UNAVAILABLE
    if number <= _LEVEL_FLOOR:
        return UNAVAILABLE
    return f"{number:.{digits}f} dBFS"


def level_plain(value, digits: int = 1) -> str:
    """dBFS without its unit, for a column."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return UNAVAILABLE
    if number <= _LEVEL_FLOOR:
        return UNAVAILABLE
    return f"{number:.{digits}f}"


def hertz(value, digits: int = 0) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return UNAVAILABLE
    if number <= 0:
        return UNAVAILABLE
    if number >= 1000:
        return f"{number / 1000:.1f} kHz"
    return f"{number:.{digits}f} Hz"


def rate(value) -> str:
    """A sample rate.  Shown in Hz, because 48000 and 44100 are exact values
    and rounding them to "48.0 kHz" loses the distinction from other rates."""
    try:
        number = int(round(float(value)))
    except (TypeError, ValueError):
        return UNAVAILABLE
    return f"{number:,} Hz".replace(",", " ")


def ratio(value, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return UNAVAILABLE
    if not math.isfinite(number):
        return UNAVAILABLE
    return f"{number:.{digits}f}"


def per_second(value, digits: int = 2) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return UNAVAILABLE
    return f"{number:.{digits}f}/s"


def duration(value) -> str:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return UNAVAILABLE
    if seconds < 1.0:
        return f"{seconds * 1000:.0f} ms"
    if seconds < 60.0:
        return f"{seconds:.2f} s"
    return f"{seconds / 60:.1f} min"


def parse_timestamp(timestamp) -> Optional[datetime.datetime]:
    """Parse a stored timestamp, or return None.

    Timestamps are written in UTC with an offset, which is right: the record
    has to mean the same instant whatever machine reads it.  A timestamp with
    no offset is assumed to be UTC, because that is what this project has
    always written, and silently treating it as local would shift every older
    record by the machine's offset.
    """
    text = str(timestamp or "").strip()
    if not text or text == UNAVAILABLE:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=datetime.timezone.utc)
    return parsed


def local_time(timestamp) -> Optional[datetime.datetime]:
    """A stored timestamp as a datetime in the machine's own timezone.

    Everything the operator reads is converted to system time.  Timestamps are
    *stored* in UTC, which is the only choice that keeps a record meaningful
    across machines and daylight-saving changes, but showing a UTC clock to
    someone whose day runs on their own clock is simply wrong: a recording at
    04:45 UTC was made at 21:45 the previous evening, and displaying the former
    makes the event look like it belongs to a different session - or, near
    midday, to a different day entirely.
    """
    parsed = parse_timestamp(timestamp)
    return parsed.astimezone() if parsed is not None else None


def clock_time(timestamp) -> str:
    """System-local wall-clock time, for a compact list column."""
    local = local_time(timestamp)
    if local is None:
        text = str(timestamp or "")
        if "T" not in text:
            return text or UNAVAILABLE
        return text.split("T", 1)[1][:8]
    return local.strftime("%H:%M:%S")


def local_stamp(timestamp) -> str:
    """Full system-local date and time, for tooltips and detail panels."""
    local = local_time(timestamp)
    if local is None:
        return str(timestamp or "") or UNAVAILABLE
    return local.strftime("%Y-%m-%d %H:%M:%S")


def local_date(timestamp) -> str:
    """The system-local date, so a time near midnight is not ambiguous."""
    local = local_time(timestamp)
    return local.strftime("%Y-%m-%d") if local is not None else UNAVAILABLE


#: Where an event's label came from.  A user label outranks the detector's
#: for anything the reviewer reads, which is the whole point of reviewing.
LABEL_SOURCE_USER = "yours"
LABEL_SOURCE_DETECTOR = "detector"
LABEL_SOURCE_NONE = "none"


def effective_label(stored) -> tuple:
    """The label the event should be called, and who called it that.

    A label you set wins over the detector's, always.  That is what reviewing
    is for, and the alternative - showing the detector's guess above your own
    judgement - would make the review controls pointless.

    The detector's label is not overwritten.  It stays recorded, attributed,
    and visible, because the two answer different questions: yours says what
    this sound was, the detector's says what its rules could see.  A
    difference between them is the most informative thing in the record, and
    it is only informative if both are kept.

    A comparison of "whose confidence is higher" is deliberately not
    attempted: the rule-based classifier reports no confidence at all, so there
    is nothing on its side of the comparison to compare against.
    """
    metadata = getattr(stored, "metadata", {}) or {}
    user = (getattr(stored, "label", None) or "").strip()
    if user:
        return user, LABEL_SOURCE_USER
    detector = detector_label(metadata)
    if detector and detector != UNAVAILABLE:
        return detector, LABEL_SOURCE_DETECTOR
    return "", LABEL_SOURCE_NONE


def user_corrected(stored) -> bool:
    """True when a label you set differs from what the detector called it.

    This is supervision, and it is worth keeping visible: an event where your
    judgement and the rules disagree is exactly the evidence needed to find out
    where the rules are wrong.
    """
    user, source = effective_label(stored)
    if source != LABEL_SOURCE_USER:
        return False
    detector = detector_label(getattr(stored, "metadata", {}) or {})
    return bool(detector and detector != UNAVAILABLE
                and detector.casefold() != user.casefold())


def detector_label(metadata: dict) -> str:
    """The detector's own label, clearly attributed to the detector."""
    classification = (metadata or {}).get("classification") or {}
    label = classification.get("label")
    return str(label) if label else UNAVAILABLE


def detector_confidence(metadata: dict) -> str:
    """The detector's confidence, or an honest statement that it has none.

    The backend reports ``null`` because its classifier is rule-based and not
    calibrated. Printing 0% would be a different and wrong claim, so the
    unavailable case is spelled out.
    """
    classification = (metadata or {}).get("classification") or {}
    value = classification.get("confidence")
    if value is None:
        return "not calibrated"
    try:
        return f"{float(value) * 100:.0f}%"
    except (TypeError, ValueError):
        return "not calibrated"


def user_confidence(value) -> str:
    """The reviewer's own confidence, kept separate from the detector's."""
    if value is None:
        return "not stated"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "not stated"
    if 0.0 <= number <= 1.0:
        return f"{number * 100:.0f}%"
    if 0.0 <= number <= 100.0:
        return f"{number:.0f}%"
    return "not stated"


# ----------------------------------------------------------------------
# Sections
# ----------------------------------------------------------------------
@dataclass
class Field:
    label: str
    value: str

    def as_row(self) -> tuple:
        return self.label, self.value


@dataclass
class EventDescription:
    """Everything the inspector shows, split by audience."""

    event_id: str
    headline: list = field(default_factory=list)      # label/value pairs
    acoustics: list = field(default_factory=list)    # label/value pairs
    source: list = field(default_factory=list)       # label/value pairs
    review: list = field(default_factory=list)       # label/value pairs
    technical: list = field(default_factory=list)    # label/value pairs

    def as_dict(self) -> dict:
        return {
            "event_id": self.event_id,
            "headline": [f.as_row() for f in self.headline],
            "acoustics": [f.as_row() for f in self.acoustics],
            "source": [f.as_row() for f in self.source],
            "review": [f.as_row() for f in self.review],
            "technical": [f.as_row() for f in self.technical],
        }


def describe_event(stored) -> EventDescription:
    """Build the inspector's content from a stored event.

    Reads the persisted record only.  Nothing is recomputed, and the fingerprint
    is summarised rather than dumped: the brief asks for human-readable values
    with the raw vector available separately.
    """
    metadata = getattr(stored, "metadata", {}) or {}
    fingerprint = getattr(stored, "fingerprint", None)
    measurements = metadata.get("measurements") or {}
    audio = metadata.get("audio") or {}
    provenance = metadata.get("provenance") or {}

    description = EventDescription(event_id=stored.event_id)

    # -- headline: what and when -------------------------------------
    description.headline = [
        Field("Event", stored.event_id),
        Field("Time", local_stamp(stored.timestamp)),
        Field("Duration", duration(stored.duration)),
        Field("Audio stored", audio.get("stored_duration_seconds")
              and duration(audio["stored_duration_seconds"])
              or UNAVAILABLE),
        Field(
            "Audio completeness",
            "complete" if provenance.get("span_complete") else
            "truncated" if (
                provenance.get("preroll_missing_frames")
                or provenance.get("postroll_missing_frames")
                or audio.get("stored_duration_seconds") is not None
                and audio.get("span_duration_seconds") is not None
                and audio["stored_duration_seconds"] < audio["span_duration_seconds"]
            ) else UNAVAILABLE,
        ),
        Field("Decision", format_decision(stored.decision)),
    ]

    # -- acoustics: what the detector measured ----------------------
    if fingerprint is not None:
        acoustics = [
            Field("Mean level", level(fingerprint.level_mean)),
            Field("Level variation", level_plain(
                fingerprint.level_dynamic_range, 1) + " dB"),
            Field("Peak SNR", level(fingerprint.snr_peak)),
            Field("SNR variation", level_plain(fingerprint.snr_std, 1) + " dB"),
            Field("Centroid", hertz(fingerprint.centroid_mean)),
            Field("Centroid variation", hertz(fingerprint.centroid_std)),
            Field("Flatness", ratio(fingerprint.flatness_mean, 3)),
            Field("Presence", ratio(fingerprint.presence)),
            Field("Onsets", str(fingerprint.onset_count)),
            Field("Onset rate", per_second(fingerprint.onset_density)),
            Field("Crest (max)", ratio(fingerprint.crest_max, 1)),
        ]
    else:
        acoustics = [
            Field("Mean level",
                  level(measurements.get("mean_rms_dbfs"))),
            Field("Peak SNR", level(measurements.get("peak_snr_db"))),
            Field("Noise floor", level(measurements.get("noise_floor_db"))),
            Field("Presence", ratio(measurements.get("presence_fraction"))),
            Field("Onsets", str(measurements.get("impulsive_onsets", UNAVAILABLE))),
            Field("Fingerprint", "not stored"),
        ]
    description.acoustics = acoustics

    # -- source: where the audio came from ---------------------------
    source_block = metadata.get("source") or {}
    audio_path = getattr(stored, "audio_path", None)
    evicted_at = getattr(stored, "audio_evicted_at", None)
    if evicted_at:
        # Said plainly, with the time, because the event is still fully
        # reviewable - it is the file that is gone, not the evidence.
        audio_location = f"evicted by audio retention ({evicted_at[:19]})"
    else:
        audio_location = audio_path or "not recorded"
    source_fields = [
        Field("Source", source_block.get("name") or UNAVAILABLE),
        Field("Input format", source_block.get("wire_format") or UNAVAILABLE),
        Field("Source rate", rate(source_block.get("stream_rate"))
              if source_block.get("stream_rate") else UNAVAILABLE),
        Field("Internal rate", rate(audio.get("sample_rate"))
              if audio.get("sample_rate") else UNAVAILABLE),
        Field("Channels", str(audio.get("channels"))
              if audio.get("channels") else UNAVAILABLE),
        Field("Stored as", " / ".join(
            str(part) for part in (audio.get("format"), audio.get("subtype"))
            if part
        ) or UNAVAILABLE),
        Field("Audio file", audio_location),
        Field("Audio kept", audio.get("stored_duration_seconds")
              and duration(audio["stored_duration_seconds"]) or UNAVAILABLE),
        Field("Event span", audio.get("span_duration_seconds")
              and duration(audio["span_duration_seconds"]) or UNAVAILABLE),
    ]
    # A standing caveat, not a claim about any particular sender. A network
    # sender typically applies gain and filtering before transmitting, so the
    # "original" audio held here is the signal *as received* - which is the only
    # thing this system can truthfully claim to have preserved.
    source_fields.insert(
        1,
        Field("Signal path", "as received from the source; any processing the "
                             "sender applied before sending is part of it"),
    )
    for index, config in enumerate(source_block.get("client_configs") or []):
        if isinstance(config, dict) and config:
            # Shown verbatim: this is what the sender reported, presented as
            # data.  The GUI does not interpret these keys or attach meaning to
            # them - the user decides what they mean.
            source_fields.append(
                Field(f"Sender setting {index + 1}",
                      ", ".join(f"{k}={v}" for k, v in sorted(config.items())))
            )
    description.source = source_fields

    # -- review: what the human decided ------------------------------
    description.review = [
        Field("Decision", format_decision(stored.decision)),
        Field("Label", _label_field_value(stored)),
        Field("Notes", stored.notes or ""),
        Field("Reviewer confidence", user_confidence(stored.confidence)),
        Field("Annotations", str(len(stored.annotations))),
        Field("Detector's label", detector_label(metadata)),
        Field("Detector confidence", detector_confidence(metadata)),
    ]

    # -- technical: versions and provenance -------------------------
    technical = [
        Field("Segmentation reason", stored.segmentation_reason or UNAVAILABLE),
        Field("Fingerprint version",
              str(fingerprint.fingerprint_version) if fingerprint
              else UNAVAILABLE),
        Field("Normalisation version",
              str(fingerprint.normalisation_version) if fingerprint
              else UNAVAILABLE),
        Field("Detector version", stored.detector_version or UNAVAILABLE),
        Field("Created", clock_time(stored.created_at)),
        Field("Sample rate", f"{audio.get('sample_rate')}"
              if audio.get("sample_rate") else UNAVAILABLE),
        Field("Format", str(audio.get("subtype")) if audio.get("subtype")
              else UNAVAILABLE),
        Field("Merged detections",
              str(provenance.get("merged_detections", UNAVAILABLE))),
        Field("Truncated at max duration",
              "yes" if provenance.get("truncated_at_max_duration") else "no"),
        Field("Source", (metadata.get("source") or {}).get("name")
              or UNAVAILABLE),
    ]
    if fingerprint is not None:
        technical.append(
            Field("Fingerprint fields", str(len(fingerprint.vector)))
        )
    description.technical = technical

    return description


# ----------------------------------------------------------------------
# List rows
# ----------------------------------------------------------------------
@dataclass
class EventRow:
    """One event list row: compact by default, full on request."""

    event_id: str
    time: str
    duration: str
    decision: str
    decision_short: str
    label: str
    status: str
    #: Full system-local date and time, for the tooltip.  The compact ``time``
    #: column cannot carry a date, and an event recorded near midnight is
    #: genuinely ambiguous without one.  Declared last because a defaulted
    #: field may not precede the others.
    when: str = ""
    #: "yours", "detector" or "none" - which of the two labels is shown.
    label_source: str = LABEL_SOURCE_NONE
    #: True when your label differs from the detector's.
    corrected: bool = False
    detail: dict = field(default_factory=dict)

    def compact(self) -> list:
        """The default columns: event number, time, duration, decision, label.

        The event number is shown because it is the identifier everything else
        uses: the separation CLI takes it as ``--event``, the metadata file
        names it, the similarity results quote it, and a reviewer comparing an
        event against a note or a chat message has to be able to read it off
        the list rather than counting rows.
        """
        return [self.event_id, self.time, self.duration,
                self.decision_short, self.label]

    def full(self) -> list:
        """Optional extra columns, behind a toggle rather than shown always."""
        return self.compact() + [
            self.detail.get("onsets", UNAVAILABLE),
            self.detail.get("presence", UNAVAILABLE),
            self.detail.get("level", UNAVAILABLE),
            self.detail.get("snr", UNAVAILABLE),
            self.detail.get("reason", UNAVAILABLE),
        ]

    def tooltip(self) -> str:
        bits = [
            f"{self.event_id}",
            # Full date and time, not just the column's clock: an event at
            # 00:10 belongs to the previous day in local time even though its
            # stored UTC timestamp says otherwise.
            f"time: {self.when}",
            f"label: {self.label or 'none'}"
            + (f" (yours, overriding the detector)"
               if self.label_source == "yours" and self.corrected
               else (f" (yours)" if self.label_source == "yours"
                     else (f" ({self.label_source})"
                           if self.label_source != "none" else ""))),
            f"decision: {self.decision}",
            f"status: {self.status}",
        ]
        return "\n".join(bits)


def _label_field_value(stored) -> str:
    """The Label field, saying which label wins and why.

    When you have labelled an event that disagrees with the detector, the
    precedence is stated rather than left to be inferred from two adjacent
    rows: yours is the event's name, and the detector's is recorded
    underneath as what its rules saw.
    """
    label, source = effective_label(stored)
    if source == LABEL_SOURCE_USER:
        confidence = getattr(stored, "confidence", None)
        stated = f", your confidence {confidence:g}" if confidence is not None else ""
        if user_corrected(stored):
            detector = detector_label(getattr(stored, "metadata", {}) or {})
            return (
                f"{label}  (yours{stated}, overriding the detector's "
                f"{detector})"
            )
        return f"{label}  (yours{stated})"
    if source == LABEL_SOURCE_DETECTOR:
        return f"{label}  (detector's; add a label of your own to override)"
    return "none"


def build_row(stored) -> EventRow:
    """Compact list row for one stored event."""
    label, source = effective_label(stored)
    corrected = user_corrected(stored)
    metadata = getattr(stored, "metadata", {}) or {}
    measurements = metadata.get("measurements") or {}
    fingerprint = getattr(stored, "fingerprint", None)
    audio = metadata.get("audio") or {}
    provenance = metadata.get("provenance") or {}

    stored_seconds = audio.get("stored_duration_seconds")
    span_seconds = audio.get("span_duration_seconds")
    # Keyed off the recorded path, not the metadata block: the metadata records
    # the requested span even when nothing was written, so trusting it would
    # report "partial audio" for an event that has no audio at all.
    if not getattr(stored, "audio_path", None):
        # "no audio" and "the audio was evicted by retention" are different
        # facts about different events, and conflating them would make a
        # bounded evidence cache look like a broken recorder.  The fingerprint
        # is still here either way; only the file went.
        status = (
            "audio evicted" if getattr(stored, "audio_evicted_at", None)
            else "no audio"
        )
    elif stored_seconds is None:
        status = "unknown"
    elif span_seconds is not None and stored_seconds < span_seconds:
        status = "partial audio"
    else:
        status = "complete"

    onsets = (
        fingerprint.onset_count if fingerprint
        else measurements.get("impulsive_onsets", UNAVAILABLE)
    )
    presence = (
        fingerprint.presence if fingerprint
        else measurements.get("presence_fraction", UNAVAILABLE)
    )
    mean_level = (
        fingerprint.level_mean if fingerprint
        else measurements.get("mean_rms_dbfs")
    )
    snr = (
        fingerprint.snr_peak if fingerprint
        else measurements.get("peak_snr_db")
    )

    return EventRow(
        event_id=stored.event_id,
        time=clock_time(stored.timestamp),
        when=local_stamp(stored.timestamp),
        duration=duration(stored.duration),
        decision=format_decision(stored.decision),
        decision_short=decision_short(stored.decision),
        label=label,
        label_source=source,
        corrected=corrected,
        status=status,
        detail={
            "onsets": str(onsets) if onsets != UNAVAILABLE else UNAVAILABLE,
            "presence": ratio(presence) if presence != UNAVAILABLE
            else UNAVAILABLE,
            "level": level_plain(mean_level) if mean_level is not None
            else UNAVAILABLE,
            "snr": level_plain(snr) if snr is not None else UNAVAILABLE,
            "reason": stored.segmentation_reason or UNAVAILABLE,
        },
    )


def build_rows(stored_events) -> list:
    return [build_row(event) for event in stored_events]
