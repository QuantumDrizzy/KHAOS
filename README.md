# KHAOS

**Closed-loop BCI kernel: a CUDA DSP hot-path built for sub-millisecond EEG filtering, with compiler-enforced safety and post-quantum cryptography.**

High-performance CUDA DSP pipeline with compiler-enforced safety and post-quantum cryptography.

### Key Results
- **CUDA DSP hot-path** designed for **< 100 µs/frame** at 1000 Hz, 64-channel (per-stage estimates in code: ~3 µs host→kernel, ~5 µs D2H). **Design target — not yet benchmarked end-to-end** (`tests/bench/` harness pending the GPU build). Distrust this number until it is measured.
- The **feature-extraction + quantum-mirror stage runs asynchronously**, off the hot path (~9 ms at 4-ch, ~16 ms at 64-ch in Python — measured, `tests/bench/bench_feature_extraction.py`). It is *not* part of the sub-100 µs path.
- Electrode-density-invariant **output contract**: 4/16/32/64-channel EEG all map to the same 12-qubit / 240-element interface (shape `(240,)`, values in `[0, 2π]`), so the quantum layer is channel-count-agnostic. The feature *values* differ by montage (4-ch uses per-electrode biomarkers; >4-ch uses PCA spatial components) — it's the contract that's invariant, not the vector. (`tests/unit/test_density_invariance.py`)
- Three independent safety layers: Python runtime, C++17 `static_assert`, and FPGA hardware watchdog (5 ms)
- SHA-256 chained forensic audit ledger on every state transition
- Post-quantum cryptography (CRYSTALS-Kyber-1024)

### What it does
KHAOS is a real-time closed-loop BCI kernel designed with sovereignty and safety as hard architectural constraints. It processes EEG signals through a CUDA-accelerated DSP pipeline and extracts a compact quantum-inspired feature representation while enforcing strict safety limits at multiple layers.

### Stack
- **Languages**: C++17, CUDA, Python
- **Key Technologies**: CUDA-Q, CMake, OpenSSL
- **Target**: RTX 40/50 series (sm_89+)

### Safety Architecture
Safety is enforced at three independent layers:
1. **Python runtime** — bounds checking and stimulation gating
2. **C++ compile-time** — `static_assert(STIM_ABSOLUTE_MAX_AMP <= 50.0f)`
3. **FPGA hardware** — 5 ms watchdog timer (independent of software)

If any layer detects a violation, the system enters a safe state.

### Build
```bash
git clone https://github.com/QuantumDrizzy/KHAOS.git
cd KHAOS
python3 scripts/gen_coefficients.py
cmake -B build -DCMAKE_BUILD_TYPE=Release -DETHICS_COMPLIANT=ON
cmake --build build --parallel
```

### Run
```bash
# Synthetic mode (no hardware required)
./build/khaos_mirror --dry-run

# Live with a Muse 2 headset
./build/khaos_mirror --stream "EEG"
```

### Documentation
- [ETHICS.md](ETHICS.md) — the safety and ethics contract this kernel is built to honour.
- [INTEGRATION_REPORT.md](INTEGRATION_REPORT.md) — end-to-end integration status and measured timings.

## License

MIT © QuantumDrizzy — see [LICENSE](LICENSE).
