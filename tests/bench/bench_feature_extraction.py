"""bench_feature_extraction.py — honest latency of the Python feature stage.

`tests/bench/` was empty: KHAOS advertised "sub-100µs end-to-end latency at
1000 Hz" with no reproducible benchmark behind it. This measures the one stage
that runs without a GPU — the Python EEG → 12-qubit feature extraction (Welch
PSD, coherence, FAA, engagement) — so there is at least one real, repeatable
number on record.

Reality (RTX-class dev box, numbers vary): this stage is ~9 ms (4-ch) to ~16 ms
(64-ch) per frame, i.e. NOT in the <100µs hot path. It belongs to the async /
lower-rate bridge stage (see src/io/muse2_adapter.py: "NOT in the hard real-time
path"). The <100µs target is for the CUDA DSP filtering hot-path only, and that
end-to-end figure is still UNBENCHMARKED here (needs nvcc + GPU; see README).

Run:  python tests/bench/bench_feature_extraction.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.bci.feature_extractor import Muse2FeatureExtractor  # noqa: E402

ITERS = 200
N_SAMPLES = 512


def bench(n_channels: int, iters: int = ITERS) -> float:
    """Return mean microseconds per extract() call for n_channels."""
    ext = Muse2FeatureExtractor(n_channels=n_channels)
    if n_channels > 4:
        cal = np.random.default_rng(1).standard_normal((n_channels, 8192)) * 20e-6
        ext._spatial.fit(cal)
    alpha = np.random.default_rng(2).standard_normal((n_channels, N_SAMPLES)) * 20e-6
    theta = np.random.default_rng(3).standard_normal((n_channels, N_SAMPLES)) * 20e-6

    ext.extract(alpha, theta)  # warm up (JIT-free, but caches/page-ins)
    t0 = time.perf_counter()
    for _ in range(iters):
        ext.extract(alpha, theta)
    return (time.perf_counter() - t0) / iters * 1e6


def main() -> int:
    print("KHAOS feature-extraction latency (Python stage, async bridge path)\n")
    print(f"  {'channels':>9}  {'us/frame':>10}  {'frames/s':>9}")
    for n in (4, 16, 32, 64):
        us = bench(n)
        print(f"  {n:>9}  {us:>10.1f}  {1e6 / us:>9.0f}")
    print(
        "\n  Note: this stage is NOT the <100µs hot path — that target is the\n"
        "  CUDA DSP filtering path and is not yet benchmarked end-to-end here."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
