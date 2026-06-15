"""test_density_invariance.py — what "electrode-density-invariant" actually means.

KHAOS's headline claim is that 4-, 16-, 32-, or 64-channel EEG all map into the
same 12-qubit / 240-element representation. This test pins down precisely which
part of that is true:

  TRUE  — the OUTPUT CONTRACT is density-invariant: regardless of channel count,
          extract() returns shape (240,), dtype float64, values in [0, 2pi].
          Downstream code (the quantum circuit layer) never has to know N.
  FALSE — the VALUES are NOT identical across channel counts. The 4-ch path uses
          channel-specific biomarkers (per-electrode FAA, fronto-temporal
          coherence); the >4-ch path uses PCA spatial components. Same brain
          state, different numbers. "Identical feature vector" would be wrong;
          "identical contract / interface" is the honest statement.

It also checks bit-for-bit determinism (a stated KHAOS requirement).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.bci.feature_extractor import Muse2FeatureExtractor, THETA_LEN  # noqa: E402

CHANNEL_COUNTS = [4, 16, 32, 64]
N_SAMPLES = 512
TWO_PI = 2.0 * math.pi


def _synth_windows(n_channels: int, seed: int = 0):
    """Deterministic synthetic alpha/theta band windows, shape (n_channels, 512)."""
    rng = np.random.default_rng(seed)
    alpha = rng.standard_normal((n_channels, N_SAMPLES)) * 20e-6
    theta = rng.standard_normal((n_channels, N_SAMPLES)) * 20e-6
    return alpha, theta


def _make_extractor(n_channels: int) -> Muse2FeatureExtractor:
    ext = Muse2FeatureExtractor(n_channels=n_channels)
    if n_channels > 4:
        # Fit the PCA spatial filter so we exercise the real path, not the
        # identity fallback. Calibration data uses a different seed than the
        # evaluation windows.
        cal = np.random.default_rng(123).standard_normal((n_channels, 8192)) * 20e-6
        ext._spatial.fit(cal)
    return ext


# ── TRUE: the output contract is invariant across channel counts ──────────────

@pytest.mark.parametrize("n_channels", CHANNEL_COUNTS)
def test_contract_shape_invariant(n_channels):
    ext = _make_extractor(n_channels)
    alpha, theta = _synth_windows(n_channels)
    out = ext.extract(alpha, theta)
    assert out.shape == (THETA_LEN,) == (240,)
    assert out.dtype == np.float64


@pytest.mark.parametrize("n_channels", CHANNEL_COUNTS)
def test_contract_range_invariant(n_channels):
    ext = _make_extractor(n_channels)
    alpha, theta = _synth_windows(n_channels)
    out = ext.extract(alpha, theta)
    assert out.min() >= 0.0
    assert out.max() <= TWO_PI + 1e-9


# ── Determinism: same input -> bit-identical output ───────────────────────────

@pytest.mark.parametrize("n_channels", CHANNEL_COUNTS)
def test_deterministic(n_channels):
    ext = _make_extractor(n_channels)
    alpha, theta = _synth_windows(n_channels, seed=7)
    a = ext.extract(alpha.copy(), theta.copy())
    b = ext.extract(alpha.copy(), theta.copy())
    assert np.array_equal(a, b), "extraction must be bit-for-bit reproducible"


# ── HONESTY: values are NOT identical across channel counts ───────────────────

def test_values_differ_across_density():
    """4-ch and 64-ch paths produce different vectors (different biomarkers).

    This is the assertion that keeps the README honest: density-invariance is a
    property of the CONTRACT, not of the numbers. If someone ever makes the two
    paths actually coincide, this test will flag that the claim changed.
    """
    ext4 = _make_extractor(4)
    ext64 = _make_extractor(64)
    a4, t4 = _synth_windows(4, seed=1)
    a64, t64 = _synth_windows(64, seed=1)
    out4 = ext4.extract(a4, t4)
    out64 = ext64.extract(a64, t64)
    assert out4.shape == out64.shape == (240,)        # same contract
    assert not np.allclose(out4, out64)               # different values


# ── The 240 vector is 12 qubits tiled across 20 layers (structure check) ──────

@pytest.mark.parametrize("n_channels", CHANNEL_COUNTS)
def test_layer_tiling_structure(n_channels):
    """theta = tile(12 qubits, 20 layers); every 20 layers repeat the 12 base
    angles. Verifies the (240,) is genuinely 12x20, not 240 free parameters."""
    ext = _make_extractor(n_channels)
    alpha, theta = _synth_windows(n_channels, seed=3)
    out = ext.extract(alpha, theta)
    base = out[:12]
    for layer in range(20):
        chunk = out[layer * 12:(layer + 1) * 12]
        assert np.array_equal(chunk, base), f"layer {layer} differs from base"
