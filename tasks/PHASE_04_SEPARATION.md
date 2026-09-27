# Phase 04 — Source Separation

## Objective

Implement real query-based source separation.

## Tasks

### 1. Prove the model independently

Before integrating into the GUI:

- load model
- load WAV
- execute query
- generate output WAV
- verify output
- play output

---

### 2. Implement Separator Interface

Create a stable application-level interface around the actual model.

---

### 3. Implement Worker

Use an asynchronous worker.

Default:

    one worker

---

### 4. Implement Queries

Support arbitrary text queries.

Examples:

    a whisper
    footsteps
    a person moving
    electrical interference

---

### 5. Output

Create:

    isolated.wav

Preserve the direct model output.

---

### 6. Validation

Verify:

- file exists
- readable
- nonzero duration
- valid samples
- valid sample rate

---

### 7. Multiple Runs

Never overwrite previous separation attempts.

---

## Acceptance

Given a valid event WAV:

    query = "footsteps"

the system must produce a real playable WAV containing the model's separation output.

Filtering the original recording does not satisfy this task.
