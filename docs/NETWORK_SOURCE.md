# Network Audio Source

The primary audio source is a live TCP stream from an Android device.

- **Stream port:** `8190` (TCP)
- **Reserved ports:** `806[0-4]`, `806[0-4]`, `806[0-4]`, `806[0-4]`, `806[0-4]`

While streaming, the five ports 806[0-4]-806[0-4] are **bound and held** by the
application. The reservation is real, not documentary: the sockets stay open for
the life of the session, so a conflict is reported at startup rather than
discovered later at first use. Binding deliberately omits `SO_REUSEADDR`, so a
port already owned by another process is a hard error.

The reserved ports are taken *before* the stream port is offered, so by the time
a client can connect, 806[0-4]-806[0-4] already belong to this session.

Implementation: `app/audio/network.py`.
Reference client (Python, portable to Kotlin): `tools/android_stream_client.py`.

---

## 1. Why not port 8090

The AMP receiver (`/home/jason/amp/server/audio_receiver.py`) binds its PCM,
control and visualization listeners inside **8090..8099**, and the host's 808x
range is congested. Two applications cannot hold the same port, so this project
takes a disjoint range and keeps the same shape AMP uses: one stream port, five
reserved ports 30 below it.

`tests/test_network.py` asserts the ranges do not overlap, so a future change
cannot quietly reintroduce the collision.

---

## 2. Wire format

This is the format the Android device already sends, so one app can talk to
either receiver.

1. The client connects to port 8190.
2. The client *optionally* sends one newline-terminated JSON configuration
   object.
3. The client streams **raw, unframed** signed 16-bit little-endian PCM at
   **44 100 Hz, mono**.

There is no length prefix, no handshake reply, and no end marker. The server
sends nothing at all. A client that disconnects has ended its stream.

88 200 bytes per second of audio.

### Detecting the config line

A leading `{` plus a newline within the first 4 KiB means JSON. Anything else
is audio from the first byte. Binary PCM is not reliably distinguishable from
text, so this mirrors the existing receiver's heuristic rather than pretending
to a certainty it does not have.

Two details that matter and are easy to get wrong:

- If a newline appears but the text before it is **not** a JSON object, that
  newline was just a byte of audio. Everything read so far must be treated as
  audio, including the bytes *before* the newline.
- A short pause while probing is not an answer. A client that connects and then
  takes a moment to send must not be permanently downgraded to raw-PCM-only.

### Configuration keys observed

Sent by the device, recorded in metadata, and **not** applied to stored audio:

    {"amplification": 2.185,
     "breathing_sensitivity": 48,
     "breathing_cooldown": 5,
     "segment_duration_min": 5}

`amplification` is a gain. Applying it at capture time would bake it into the
evidence permanently, so it is recorded and left for playback instead
(AGENTS.md section 2.1).

### Kotlin reference

```kotlin
val socket = Socket(host, 8190)
val out = BufferedOutputStream(socket.getOutputStream())

// Optional config line, must be a single line.
out.write("""{"amplification":1.0}""".toByteArray())
out.write('\n'.code)
out.flush()

// Raw stream: no framing, no length prefix.
val input = socket.getInputStream()
val buffer = ByteArray(64 * 1024)
while (true) {
    val n = input.read(buffer)
    if (n <= 0) break                 // end of stream
    // buffer[0 until n] is s16le mono 44 100 Hz
    // send straight to the capture pipeline; do not add headers.
}
```

---

## 3. Conversion

The wire format is s16le at 44 100 Hz. The internal format is float32 at
48 000 Hz (`app/config.py`). Both conversions happen once, here at the input
boundary, so nothing downstream ever handles 16-bit or 44.1 kHz:

    s16le bytes -> float32 (/ 32768) -> mono -> 48 kHz

Int16 is divided by 32768 rather than 32767, so the *scaling* step maps the wire
range into (-1.0, +1.0) and never produces exactly full scale.

That does not bound the final samples: the anti-alias filter has finite
stopband rejection, so a discontinuous waveform produces Gibbs overshoot above
1.0 after resampling. That is correct filter behaviour, not a bug. It is
harmless because evidence is stored as float32 WAV, which holds values beyond
+/-1.0 without clipping, so an overshoot is recorded faithfully instead of being
silently flattened.

### Per-client resampler

Each client owns its own resampler. A single shared one would interleave two
senders into one filter stream, and would make a disconnect ambiguous: flushing
the shared filter for a sender that just left would corrupt a sender still
connected. On disconnect, that client's resampler is flushed so the ~3 ms it
held for rate alignment is not silently lost, and the remainder is reported.

The streaming resampler is verified to produce output that is **bit-identical
regardless of how the stream is chunked**, so block boundaries cannot introduce
a click. See `tests/test_resample.py`.

---

## 4. One client at a time

A second concurrent connection is **refused**, with `ERR server is already
streaming` sent to the client and a message logged.

Summing two senders sounds obvious and is wrong here. The chunks arrive on
independent TCP connections with independent jitter, so "take one chunk from
each and add" combines samples captured at different wall-clock instants. The
result is a signal that is neither device's audio, and it would quietly corrupt
the evidence. Refusing is honest; a plausible-looking bad mix is not.

Multi-device capture needs real time alignment, and is separate work.

Once the first client disconnects, the next one is accepted normally.

---

## 5. Starting without a client

The server does **not** need a client in order to start. It binds, reserves its
ports, listens, and runs; the phone may connect at any time. A missing client is
a normal idle state, not a startup failure, and an idle run exits cleanly with
no chunks written.

    ./run.sh --source network

`--wait-client N` pauses for N seconds and reports on the client state. It is a
diagnostic only and never gates startup.

---

## 6. Operating

    python -m app.capture --source network                 # port 8190
    python -m app.capture --source network --port 9090     # different stream port
    python -m app.capture --source network --reserve 9060,9061,9062,9063,9064
    python -m app.capture --source network --no-reserve
    python -m app.capture --source network --status        # per-client detail

Send audio from the reference client:

    python tools/android_stream_client.py --server <host> --file input.wav
    python tools/android_stream_client.py --server <host> --tone 440 --realtime
    python tools/android_stream_client.py --server <host> --file in.wav \
        --config '{"amplification":1.0}'

---

## 7. Error handling

| Condition                    | Behaviour                                        |
| ---------------------------- | ------------------------------------------------ |
| Stream port already bound    | Startup fails clearly; nothing is bound          |
| A reserved port is in use    | Reported per port; the rest are still reserved   |
| All reserved ports in use    | Startup fails                                    |
| Second concurrent client     | Refused with `ERR`; first client unaffected      |
| Client disconnects           | Clean end of stream; stream reusable afterwards   |
| Invalid JSON config line     | Reported; the stream is still treated as audio   |
| Config line absent           | Treated as raw PCM from the first byte           |
| Rate needs converting        | Converted once, at this boundary                 |

A stalled sender is not treated as end of stream: `read()` keeps waiting, because
a live source has no end until it is stopped. Use `read(timeout=...)` to poll.
