# Computational Neuroscience → KHAOS — study map (Thu 2026-06-18 → Sun 2026-06-21)

> Built for the no-coding study week (weekly quota spent; Claude back Monday).
> The rule: **study so it lands in KHAOS**, not as detached theory. KHAOS is a
> *non-invasive EEG closed-loop BCI* — a CUDA DSP hot-path → compact feature contract
> → safe feedback. So this map deliberately **down-weights invasive spike-train
> material** (Hodgkin–Huxley, single-neuron tuning, Poisson spiking) to "foundation,
> light touch", and **up-weights** EEG signals, decoding, and closed-loop control —
> the parts that are actually load-bearing for shipping KHAOS.
>
> Each block: **what to learn · the canonical source · → the real KHAOS file it feeds ·
> what it unlocks on Monday.** Everything here is reading/paper-and-pencil — no GPU, no
> quota. Compute-heavy follow-ups are parked in the "Monday bridge" at the end.

---

## Day 1 (Thu) — EEG & neural oscillations: *the signal KHAOS eats*

**Learn:** neural oscillations and the bands (δ θ α β γ) and what each *means*
physiologically; power spectral density; event-related potentials; time–frequency
analysis (wavelets, Hilbert/analytic signal, why STFT trades time vs frequency);
referencing/montages; artifacts (EOG/EMG/line noise) and why they dominate real EEG.

**Source (the practical bible):** Mike X Cohen, *Analyzing Neural Time Series Data* —
the time–frequency and filtering chapters. Free companion lectures on his YouTube.
Intuition primer: Buzsáki, *Rhythms of the Brain*, ch. 1–4.

**→ KHAOS:** `include/dsp_pipeline.h`, `src/dsp` (the sub-100 µs filtering hot-path),
`src/neuro` (feature extraction). The bands you filter and the biomarkers you pull out
*are exactly these*. Goal: be able to justify, physiologically, every feature that ends
up in the 240-element vector — why this band, why this biomarker, for which montage.

**Unlocks Monday:** a principled review of which features the 4-ch (per-electrode
biomarker) vs >4-ch (PCA spatial) paths extract — keep/cut by physiological merit.

---

## Day 2 (Fri) — Neural encoding & decoding: *the BCI core*

**Learn:** the encoding ↔ decoding duality (Dayan–Abbott framing); population decoding,
MAP/MLE linear decoders; then the EEG-specific stack — **Common Spatial Patterns (CSP)**,
feature selection, classifiers (LDA, SVM), and the modern **Riemannian-geometry** approach
(EEG trial = covariance matrix living on the SPD manifold; classify *on the manifold* —
Barachant/Congedo). This is current SOTA for EEG BCIs and conceptually beautiful (it's
geometry on positive-definite matrices — close to your math taste).

**Source:** Dayan & Abbott, *Theoretical Neuroscience*, ch. 3 (neural decoding) for the
rigorous base; Lotte et al. 2018, *"A review of classification algorithms for EEG-based
BCIs"* (the practical map of the whole field); Barachant's Riemannian-BCI tutorial / the
`pyRiemann` docs for the geometry.

**→ KHAOS:** `src/bci` (decoder), `src/models`. Your >4-ch path already does PCA spatial
components — **CSP / Riemannian is the natural upgrade**, and the encoding/decoding duality
is *why* a fixed feature contract is defensible. This is the day that most directly raises
KHAOS's ceiling.

**Unlocks Monday:** a decoder-validation harness + a concrete decision on a CSP or
Riemannian front-end before the quantum-mirror stage.

---

## Day 3 (Sat) — Closed-loop & neurofeedback: *the "closed-loop" in closed-loop BCI*

**Learn:** closed-loop neurofeedback principles and the perception→action loop; **why
latency is causal, not cosmetic** (plasticity windows, the need to act inside the same
neural event you measured); control-theory basics (feedback, stability, the cost of loop
delay); stimulation paradigms (tACS/tDCS/closed-loop DBS *conceptually*) and the hard
**safety** case — dose limits, why amplitude must be bounded at every layer.

**Source:** Sitaram et al. 2017, *Nat Rev Neurosci*, *"Closed-loop brain training: the
science of neurofeedback"* (the field's anchor review); Åström & Murray, *Feedback
Systems* (free PDF), ch. 1–2, for the control-theory vocabulary (feedback, stability,
delay).

**→ KHAOS:** `include/feedback_engine.h`, `include/safety_constants.h`, the 3-layer safety
(Python runtime / C++17 `static_assert(STIM_ABSOLUTE_MAX_AMP <= 50)` / FPGA 5 ms watchdog),
stim gating. **This day justifies your whole latency obsession physiologically** — the
sub-100 µs target exists *because* closed-loop causality demands it. It also grounds why
safety is a hard architectural constraint, not a checkbox.

**Unlocks Monday:** a principled latency *target* (derived from a real plasticity/timing
argument) to design the `tests/bench/` closed-loop benchmark against.

---

## Day 4 (Sun) — Real-time systems + the feature contract: *synthesis*

**Learn:** real-time signal-processing constraints and the **latency-budget decomposition**
(host→kernel, kernel, D2H — you already estimate ~3 µs / ~5 µs in code); streaming via
**LSL (Lab Streaming Layer)** — the de-facto BCI transport; dimensionality reduction for
compact representations (PCA → your 240-element vector; ties back to Day 2's manifold
view); and a re-read of your own `MANIFESTO.md` / `INTEGRATION_REPORT.md` with the week's
new eyes.

**Source:** LSL documentation (labstreaminglayer.org); a PCA/manifold refresher (links to
Day 2 Riemannian); your own KHAOS docs.

**→ KHAOS:** `include/lsl_connector.h`, `include/khaos_bridge.h`, the density-invariant
12-qubit / 240-element output contract (`tests/unit/test_density_invariance.py`). Goal:
fully own the path from electrodes → invariant contract, end to end.

**Unlocks Monday (highest-value task):** the **end-to-end latency benchmark** that finally
*measures* the < 100 µs claim — which the README honestly marks **unverified** today.
Turning that into a real number is the single most valuable thing you can do for KHAOS's
credibility (engineering honesty: distrust the number until measured).

---

## Monday bridge — what each study day unlocks, prioritized

| Priority | KHAOS task (Monday, fresh weekly quota) | Fed by | Why it matters |
|---|---|---|---|
| **1** | End-to-end CUDA latency benchmark (`tests/bench/`) → measure the < 100 µs hot-path | Day 3 + Day 4 | Converts the one honest-unverified headline into a real number. Highest credibility ROI. |
| **2** | Decoder-validation harness; evaluate a CSP / Riemannian front-end | Day 2 | Raises the actual decoding ceiling; the BCI's whole point. |
| 3 | Audit the feature vector by physiological merit (4-ch vs PCA paths) | Day 1 | Trims/justifies what maps to the 240-element contract. |
| 4 | Derive a principled latency target from a timing/plasticity argument | Day 3 | Replaces "< 100 µs because fast" with "because the loop must close inside X". |

## If you only do one thing
**Day 1 (Cohen, EEG time-series).** It is the most directly load-bearing — everything
downstream in KHAOS is operations on that signal. Geometry (Day 2) is the highest-upside
*new* idea if you want one stretch goal.

## Career angle (Zürich north)
This is the AI×Neuroscience pillar. The CSP/Riemannian + closed-loop-safety + real-time
CUDA combination is exactly the profile that reads as serious to a neurotech/safety-critical
employer — and the latency-honesty discipline is the tell that you measure instead of
claim. Keep the unverified flag until Monday's benchmark kills it.
