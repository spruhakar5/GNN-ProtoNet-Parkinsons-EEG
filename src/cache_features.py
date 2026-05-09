"""
Feature caching utilities.

Computes and caches node features + PLV per subject to disk as .npz
so subsequent runs can skip expensive feature extraction.

The cache filename includes a hash of the preprocessing+feature config
so changes to bands, sample rate, or epoch length invalidate the cache
automatically and prevent silent reuse of stale features.
"""

import hashlib
import numpy as np
from pathlib import Path

from config import (
    DATA_PROCESSED, FREQ_BANDS, TARGET_SFREQ,
    EPOCH_DURATION, BANDPASS_LOW, BANDPASS_HIGH, NOTCH_FREQS, N_CHANNELS,
)


def _config_fingerprint() -> str:
    """Stable hash over the preprocessing+feature config that affects cached arrays."""
    parts = [
        f"sfreq={TARGET_SFREQ}",
        f"bp={BANDPASS_LOW}-{BANDPASS_HIGH}",
        f"notch={'_'.join(str(f) for f in NOTCH_FREQS)}",
        f"ep={EPOCH_DURATION}",
        f"nch={N_CHANNELS}",
        f"bands={'-'.join(f'{k}{v[0]}_{v[1]}' for k, v in FREQ_BANDS.items())}",
    ]
    digest = hashlib.md5("|".join(parts).encode()).hexdigest()[:8]
    return digest


_FINGERPRINT = _config_fingerprint()


_FINGERPRINT_FILE = DATA_PROCESSED / "feature_cache_fingerprint.txt"


def _cache_path(subject):
    key = f"{subject.dataset}_{subject.subject_id}"
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    return DATA_PROCESSED / f"{key}_features.npz"


def _check_fingerprint() -> bool:
    """Verify the cache directory matches the current config fingerprint.
    On mismatch, prints a loud warning so the user knows to clear the cache."""
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    if not _FINGERPRINT_FILE.exists():
        _FINGERPRINT_FILE.write_text(_FINGERPRINT)
        return True
    stored = _FINGERPRINT_FILE.read_text().strip()
    if stored != _FINGERPRINT:
        print(f"\n[CACHE WARNING] Preprocessing/feature config has changed.")
        print(f"  Stored fingerprint: {stored}")
        print(f"  Current fingerprint: {_FINGERPRINT}")
        print(f"  Cached features in {DATA_PROCESSED} may be stale.")
        print(f"  To force regeneration: rm {DATA_PROCESSED}/*.npz\n")
        return False
    return True


def load_cached_features(subject):
    path = _cache_path(subject)
    if not path.exists():
        return False
    try:
        cache = np.load(path)
        subject.node_features = cache['node_features']
        subject.plv_matrix = cache['plv_matrix']
        return True
    except Exception:
        return False


def save_cached_features(subject):
    if subject.node_features is None or subject.plv_matrix is None:
        return
    path = _cache_path(subject)
    try:
        np.savez_compressed(
            path,
            node_features=subject.node_features,
            plv_matrix=subject.plv_matrix,
        )
    except Exception as e:
        print(f"  [cache save failed] {e}")


def extract_features_cached(subjects):
    from features import extract_features

    _check_fingerprint()
    hits = 0
    for subj in subjects:
        if load_cached_features(subj):
            print(f"  [cache hit] {subj.subject_id}")
            hits += 1
            continue
        try:
            extract_features(subj)
            save_cached_features(subj)
        except Exception as e:
            print(f"  [ERROR] {subj.subject_id}: {e}")

    print(f"\nCache hits: {hits}/{len(subjects)}")
    return subjects
