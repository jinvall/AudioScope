# Performance Requirements

## 1. Priority

System priority:

    1. Audio capture
    2. Audio buffering
    3. Real-time analysis
    4. GUI responsiveness
    5. Source separation

---

# 2. Audio Callback

The callback must remain lightweight.

No:

- model inference
- disk writes
- GUI operations
- blocking locks
- expensive allocations

---

# 3. CPU

The application must operate on CPU-only hardware.

Source separation is expected to be the expensive operation.

---

# 4. Separation Workers

Initial configuration:

    workers = 1

Increasing workers is optional and should be configurable.

---

# 5. Queue Monitoring

Track:

    analysis_queue_depth
    separation_queue_depth

If queues grow continuously, report overload.

---

# 6. Processing Ratio

Calculate:

    realtime_ratio =
        processing_time / audio_duration

Example:

    2 seconds processing
    10 seconds audio

    ratio = 0.2

Lower is faster than real time.

---

# 7. Memory

Monitor:

- process RSS
- model memory
- ring buffer size
- queue sizes

Prevent unbounded memory growth.

---

# 8. GUI

The GUI must not freeze while separation runs.

Large audio files should not be copied repeatedly in memory.

---

# 9. Performance Logging

Useful measurements:

- capture callback duration
- analysis duration
- event detection latency
- separation duration
- output writing duration
- queue wait time

---

# 10. Overload Policy

When CPU pressure increases:

1. Continue capture.
2. Continue buffering.
3. Continue basic analysis.
4. Delay expensive separation.
5. Inform the user.

Never sacrifice the original live stream to complete a separation job.
