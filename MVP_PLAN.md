# Remembrain Hybrid MVP Plan (4 Weeks)

## Goal

Ship a reliable, local-first desktop MVP that provides:

- Real-time face recognition from webcam
- Contextual recall from local memory data
- Simple voice-assisted logging flow (unknown-person enrollment by voice)
- Situational reminders based on last-seen context (time/place)

while keeping internals modular so a mobile client can be added in Phase 2 without rewriting core logic.

## Baseline Behavior (Source of Truth)

This release gate is derived from the existing behavior in:

- main.py
- README.md

Current expected baseline:

- Webcam feed opens and continuously scans frames.
- Known faces are matched against local face data.
- Unknown faces are flagged and can be saved.
- Person memory card shows name, relationship, last seen, notes.
- Voice reminders are optional and cooldown-limited.
- Data is stored locally in data/people.json and faces/.
- Diagnostics script validates environment and device readiness.

## MVP Release Gate (Must Pass)

All criteria below must pass before MVP is marked releasable.

### A. Recognition Reliability

1. App starts and opens camera without crash on supported Python versions.
2. Known person appears in camera and is recognized as the correct entry.
3. Unknown person appears and is labeled as unknown.
4. Fallback recognition mode works when face_recognition is unavailable.

### B. Contextual Recall

1. Recognized person card displays name, relationship, last seen, and notes.
2. last_seen data updates locally with write cooldown protections.
3. Previous encounter context is preserved and used in reminder text.

### C. Voice and Logging Flows

1. Voice reminder playback works when pyttsx3 is installed.
2. Speak button can replay current reminder.
3. Save-by-voice enrollment flow captures name + relationship and persists record.

### D. Local-First Data Integrity

1. New enrollment writes face image to faces/ and person record to data/people.json.
2. Newly enrolled person is recognized in the same run without app restart.
3. App remains functional if optional packages are missing (graceful degrade).

### E. Operational Readiness

1. diagnostics.py runs and reports clear pass/fail status.
2. Core checks pass: imports, database parse, camera access, recognition pipeline.
3. No blocking exceptions in normal app startup and shutdown path.

## 4-Week Hybrid Execution Plan

## Week 1 - Release Gate Definition and Baseline Lock

Deliverables:

- Freeze MVP gate criteria (this document).
- Capture baseline runtime behavior and known limitations.
- Add smoke test checklist for manual run-through.

Done when:

- Team agrees to release gate and test sequence.

## Week 2 - Internal Service Boundary Refactor

Deliverables:

- Split desktop internals into service modules:
  - services/recognition.py
  - services/memory.py
  - services/voice.py
- Keep main.py focused on GUI orchestration.
- Preserve existing user-facing behavior.

Done when:

- App behavior matches baseline while logic is modularized.

## Week 3 - MVP Hardening for Recall + Voice Flow

Deliverables:

- Stabilize unknown enrollment flow (typed + voice path).
- Improve reminder consistency for situational context.
- Add basic event-level logging hooks for enrollment/reminder actions.

Done when:

- Recognition, memory updates, and voice-assisted enrollment remain stable across repeated runs.

## Week 4 - Stabilization and Phase 2 Readiness

Deliverables:

- Regression pass using release gate checklist.
- Code cleanup and docs update.
- Define mobile handoff interface contract around service boundaries.

Done when:

- MVP release gate is fully green.
- Core service layer can be reused by a future mobile client adapter.

## Phase 2 Readiness Contract

To avoid rework when mobile is introduced, keep these boundaries stable:

- Recognition boundary:
  - Input: frame/image
  - Output: recognized identities + confidence context
- Memory boundary:
  - Input: identity events and updates
  - Output: structured memory records and reminder context
- Voice boundary:
  - Input: reminder text or voice-capture request
  - Output: speech playback / captured text

Desktop UI should call service APIs, not embed recognition/memory rules directly.
