"""
validate_on_dataset.py — KHAOS validation harness.

Validation is the moment KHAOS stops being "well-architected code" and becomes
"software that demonstrably extracts real brain-state information". The protocol:

    labelled EEG  ->  KHAOS feature extractor (240-vec)  ->  simple classifier  ->  accuracy vs chance

If the classifier beats chance on held-out trials, the KHAOS feature path carries real
discriminative information about the labelled mental states. (ADR-0001, Step 0.)

Two data sources
----------------
--synthetic   Runs NOW, no data/hardware. Two classes (low vs high alpha power); the
              extractor's alpha qubits should separate them well above chance. This proves
              the harness + methodology wiring is correct end-to-end.
--physionet   REAL human EEG: PhysioNet Motor Movement/Imagery via MNE (left- vs right-hand
              motor imagery). Needs `pip install mne scikit-learn` + a first-run download.
              This is the step that converts "validated in simulation" -> "validated on
              real human EEG".

Usage
-----
    python scripts/validate_on_dataset.py --synthetic
    python scripts/validate_on_dataset.py --physionet --subjects 1 2 3

The synthetic path is the contract this harness guarantees today; the PhysioNet path is the
August / this-weekend real-data swap (the loader is implemented but unverified until run
against the download).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.bci.feature_extractor import Muse2FeatureExtractor, THETA_LEN  # noqa: E402

FS = 256          # Hz — Muse 2 native (the synthetic path mirrors it)
WIN = 512         # samples per window (2 s @ 256 Hz)


# ── Data source 1: synthetic (runs now, proves the wiring) ────────────────────────────
def make_synthetic_trials(n_per_class=60, n_ch=4, fs=FS, n=WIN, seed=0):
    """Two classes that differ only in alpha power -> the extractor should separate them.

    Returns (alpha_windows, theta_windows, labels): lists of (n_ch, n) band windows + y.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n) / fs
    alpha_w, theta_w, y = [], [], []
    for label in (0, 1):
        alpha_amp = 20e-6 if label == 0 else 40e-6   # class 1 = 2x alpha
        for _ in range(n_per_class):
            a = (alpha_amp * np.sin(2 * np.pi * 10 * t + rng.uniform(0, 2 * np.pi))
                 + 4e-6 * rng.standard_normal((n_ch, n)))
            th = (8e-6 * np.sin(2 * np.pi * 6 * t + rng.uniform(0, 2 * np.pi))
                  + 4e-6 * rng.standard_normal((n_ch, n)))
            # broadcast the alpha sinusoid across channels with small per-channel jitter
            a = a + alpha_amp * 0.1 * rng.standard_normal((n_ch, 1))
            alpha_w.append(np.atleast_2d(a)[:n_ch])
            theta_w.append(np.atleast_2d(th)[:n_ch])
            y.append(label)
    return alpha_w, theta_w, np.array(y)


# ── Data source 2: PhysioNet real EEG (the August / weekend real swap) ────────────────
def load_physionet_trials(subjects=(1, 2, 3)):
    """Left- vs right-hand motor imagery from PhysioNet (real human EEG) via MNE.

    Implemented but UNVERIFIED until run against the download (needs `pip install mne`).
    Returns the same (alpha_windows, theta_windows, labels) contract as the synthetic path,
    so the rest of the harness is identical.
    """
    try:
        import mne
        from mne.datasets import eegbci
        from mne.io import concatenate_raws, read_raw_edf
    except ImportError as e:
        raise SystemExit(
            "PhysioNet path needs MNE: pip install mne scikit-learn\n" + str(e))

    alpha_w, theta_w, y = [], [], []
    runs = [4, 8, 12]   # motor imagery: left vs right fist
    for subj in subjects:
        files = eegbci.load_data(subj, runs)
        raw = concatenate_raws([read_raw_edf(f, preload=True) for f in files])
        eegbci.standardize(raw)
        events, event_id = mne.events_from_annotations(raw)
        # T1 = left fist, T2 = right fist (the two classes)
        picks = mne.pick_types(raw.info, eeg=True)
        for band, store in (("alpha", alpha_w), ("theta", theta_w)):
            lo, hi = (8.0, 13.0) if band == "alpha" else (4.0, 8.0)
            rb = raw.copy().filter(lo, hi, picks=picks, verbose="ERROR")
            ep = mne.Epochs(rb, events, event_id={"T1": event_id.get("T1", 2),
                                                  "T2": event_id.get("T2", 3)},
                            tmin=0.0, tmax=WIN / FS, picks=picks, baseline=None,
                            preload=True, verbose="ERROR")
            data = ep.get_data()                 # (n_epochs, n_ch, n_times)
            for trial in data:
                store.append(trial[:, :WIN])
        y.extend([0 if ev == event_id.get("T1", 2) else 1 for ev in ep.events[:, -1]])
    return alpha_w, theta_w, np.array(y[: len(alpha_w)])


# ── The validation core (data-source-agnostic) ───────────────────────────────────────
def extract_features(alpha_w, theta_w):
    """Run every trial through the KHAOS feature extractor -> X of shape (n_trials, 240)."""
    n_ch = alpha_w[0].shape[0]
    ext = Muse2FeatureExtractor(n_channels=n_ch)
    X = np.stack([ext.extract(a, t) for a, t in zip(alpha_w, theta_w)])
    assert X.shape[1] == THETA_LEN, f"feature dim {X.shape[1]} != {THETA_LEN}"
    return X


def classify(X, y, folds=5):
    """Cross-validated LDA accuracy vs chance. Returns (mean, std, chance)."""
    try:
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.model_selection import cross_val_score
    except ImportError as e:
        raise SystemExit("Needs scikit-learn: pip install scikit-learn\n" + str(e))
    scores = cross_val_score(LinearDiscriminantAnalysis(), X, y, cv=folds)
    chance = float(np.max(np.bincount(y)) / len(y))   # majority-class baseline
    return float(scores.mean()), float(scores.std()), chance


def main():
    ap = argparse.ArgumentParser(description="KHAOS validation harness (ADR-0001 Step 0)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--synthetic", action="store_true", help="runs now, no data/hardware")
    src.add_argument("--physionet", action="store_true", help="real EEG (needs mne + download)")
    ap.add_argument("--subjects", type=int, nargs="+", default=[1, 2, 3])
    args = ap.parse_args()

    if args.synthetic:
        print("== KHAOS validation — SYNTHETIC (wiring + methodology check) ==")
        alpha_w, theta_w, y = make_synthetic_trials()
    else:
        print(f"== KHAOS validation — PhysioNet (REAL EEG), subjects {args.subjects} ==")
        alpha_w, theta_w, y = load_physionet_trials(args.subjects)

    print(f"  trials: {len(y)}  ({np.bincount(y)} per class)  channels: {alpha_w[0].shape[0]}")
    X = extract_features(alpha_w, theta_w)
    print(f"  features: {X.shape}  (KHAOS 240-vec per trial)")
    mean, std, chance = classify(X, y)
    print(f"  LDA accuracy: {mean:.3f} ± {std:.3f}   |   chance: {chance:.3f}")
    verdict = "PASS — features carry discriminative info" if mean > chance + 0.10 \
        else "INCONCLUSIVE — features near chance (investigate)"
    print(f"  -> {verdict}")
    return 0 if mean > chance + 0.10 else 1


if __name__ == "__main__":
    sys.exit(main())
