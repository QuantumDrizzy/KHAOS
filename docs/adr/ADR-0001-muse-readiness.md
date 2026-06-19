# ADR-0001: KHAOS Muse-readiness & validation path

**Status:** Accepted
**Date:** 2026-06-19
**Deciders:** Antonio (QuantumDrizzy)

## Context

KHAOS is real, well-architected systems/CUDA software — ~14k LoC, 98 tests green, a
sophisticated real-time CUDA DSP hot-path (CUDA Graphs, `__constant__` coeffs, lock-free
pinned ring, prioritized streams), and a genuine compile-time-enforced safety layer
(`#error`/`static_assert`, SHA-256 chained tamper-evident audit ledger, FPGA watchdog
killswitch). But its honest status is **validated in simulation only**: the decoder/feature
path has been exercised on *synthetic* EEG and a Python simulation of the C++ core, never on
real human EEG. As the README states honestly, the `<100µs` hot-path is a **design target,
unbenchmarked**.

The real hardware (a **Muse 2**, 4-ch dry EEG @256 Hz) arrives **this summer (~August)**.
KHAOS-the-software only *measures/decodes* — stimulation is the host hardware's job, not
KHAOS's. Until the Muse arrives, KHAOS is being **parked** (and removed from the dev Desktop,
re-cloned in August). The forces: (1) when it comes back, the Muse + the validation must plug
in with **zero rework**; (2) the engineering-honesty brand requires the small accuracy
inconsistencies in the codebase be fixed now, not left for a reviewer to catch.

## Decision

**Park KHAOS in a Muse-ready, honesty-clean state, with the validation path defined and a
runnable harness in place.** Three parts:

1. **Validation path (the order that matters):**
   - **Step 0 — public dataset (now, free, no hardware):** run KHAOS's feature path on a
     *labelled* public EEG set (PhysioNet EEG Motor Movement/Imagery, or BCI Competition IV).
     Real recorded human EEG with ground truth → check the features carry discriminative
     information above chance. This converts "validated in simulation" → "validated on real
     human EEG", reproducibly, **before spending a euro**, and de-risks the Muse (if it can't
     separate clean lab EEG, the noisier Muse won't).
   - **Step 1 — Muse 2 live (August):** swap `SyntheticMuse2Adapter` → the real Muse over LSL
     (Petal/BlueMuse). Validate real-time, live decode, neurofeedback. The integration is
     already written (`src/io/muse2_adapter.py`) — a drop-in.
   - **Step 2 — `<100µs` measurement (CUDA build):** build + a CUDA-event end-to-end timing
     harness (`process_frame → sync_theta`). Kill the "unverified" flag with a real number.
2. **Device abstraction = config, not code** (so "just swap hardware" is *literally* true):
   sample rate, channel count, montage, LSL stream name should be a config object, not edits +
   coefficient regeneration per device. Captured here as a target; not built now (premature
   without ≥2 real devices to abstract over) — see Options.
3. **Accuracy/honesty fixes (applied in this change):** the codebase said `sm_89` for an
   `sm_120`/Blackwell card; a resampler docstring said 1.25 ms where the real group delay is
   ~39 ms; a dead test slice produced a NaN warning. All corrected — the brand is honesty.

## Options Considered

### Option A: Park as-is (do nothing)
| Dimension | Assessment |
|-----------|------------|
| Complexity | Low |
| Cost | Zero now |
| Risk | Returns in August with stale inaccuracies + an undefined validation path; rework + lost context |

**Rejected** — leaves known inaccuracies for a reviewer to find and no runway for August.

### Option B: Park Muse-ready — fixes + validation harness + ADR (CHOSEN)
| Dimension | Assessment |
|-----------|------------|
| Complexity | Low–Med |
| Cost | One focused session now |
| Risk | Low — small, tested changes; August is plug-and-play |

**Pros:** honesty-clean; the dataset validation can start *this weekend* (no hardware); a
runnable harness means August is a data swap, not a rebuild.
**Cons:** the harness's real-data loader is a scaffold (PhysioNet path stubbed) until run.

### Option C: Full device-config refactor now
**Deferred.** Building the generic device-config layer before having ≥2 real devices to
abstract over is premature — you'd abstract the wrong axes. Revisit after the Muse (and a
second device, e.g. OpenBCI) are real. The contract design (density-invariant 4/16/32/64) is
already the right foundation.

## Trade-off Analysis

The real tension is **how much to build now vs at August**. Building the full device layer (C)
risks abstracting the wrong things without real hardware. Doing nothing (A) wastes the
re-entry. B threads it: lock in the *cheap, certain* wins now (honesty fixes + the
validation protocol + a harness that runs today in synthetic mode and swaps to real data
later), and defer the *expensive, uncertain* work (device-config, latency bench) to when the
hardware makes it concrete and verifiable.

## Consequences

- **Easier:** August is a data/hardware swap, not a rebuild; the dataset validation is
  unblocked now; the public face stays honest.
- **Harder:** nothing materially — the deferred items (device-config, `<100µs` bench) need the
  hardware/toolchain anyway.
- **Revisit:** the device-config layer and the latency benchmark, both gated on real hardware
  (Muse, and the CUDA toolchain build).

## Action Items
1. [x] Fix `sm_89`→`sm_120` (signal_processor.cu, CMakeLists), the resampler group-delay
   docstring (1.25 ms → ~39 ms), and the dead NaN-producing test slice. Tests green (98).
2. [x] Validation harness `scripts/validate_on_dataset.py` — runs today in `--synthetic`
   mode (proves the wiring), with the PhysioNet/BCI-IV loader scaffolded for real data.
3. [ ] **(this weekend / August, no hardware):** run Step 0 — feed a public dataset through
   the harness, report accuracy vs chance → real-EEG validation.
4. [ ] **(August):** swap in the real Muse 2 over LSL; validate live.
5. [ ] **(needs CUDA toolchain):** build + measure the `<100µs` hot-path end-to-end.
6. [ ] **(later, ≥2 devices):** the device-config layer (sample rate / channels / montage as
   config).

## References
- KHAOS deep analysis (2026-06-19): real CUDA/safety core, sim-only, `<100µs` unverified.
- `src/io/muse2_adapter.py` (Muse integration, already written + `SyntheticMuse2Adapter`).
- PhysioNet EEG Motor Movement/Imagery (via MNE `mne.datasets.eegbci`); BCI Competition IV.
