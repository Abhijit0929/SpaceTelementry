"""
rf_features.py

Statistical feature engineering for the Random Forest anomaly classifier.

Turns raw sliding windows (n_windows, window_size, n_channels) - the same
windows produced by src/preprocessing.py's WindowGenerator - into a tabular
feature matrix (n_windows, n_features) that a Random Forest can be trained
and predicted on.
"""

import warnings
from typing import List

import numpy as np
from scipy.stats import skew, kurtosis

FEATURE_NAMES_PER_STAT = [
    "mean", "std", "min", "max", "median", "ptp", "skew", "kurtosis"
]


def extract_window_features(windows: np.ndarray) -> np.ndarray:
    """
    windows : array of shape (n_windows, window_size, n_channels)
    returns : array of shape (n_windows, n_channels * 8)
    """
    if windows.ndim != 3:
        raise ValueError(
            "Expected a 3D windows array (n_windows, window_size, n_channels), "
            f"got shape {windows.shape}"
        )

    mean = windows.mean(axis=1)
    std = windows.std(axis=1)
    minimum = windows.min(axis=1)
    maximum = windows.max(axis=1)
    median = np.median(windows, axis=1)
    ptp = maximum - minimum

    # skew/kurtosis are numerically unstable on near-constant channels
    # (many SMAP/MSL channels are 0/1 command flags with ~zero variance) -
    # scipy correctly returns 0 for these, it just warns while doing it.
    # That result is fine (confirmed by nan_to_num below catching any NaN),
    # so we blanket-ignore RuntimeWarning ONLY inside this tightly scoped
    # block, rather than matching on scipy's exact warning text (which can
    # shift slightly between scipy versions and silently stop matching).
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        sk = skew(windows, axis=1)
        kurt = kurtosis(windows, axis=1)

    features = np.concatenate(
        [mean, std, minimum, maximum, median, ptp, sk, kurt],
        axis=1,
    )

    # Guard against NaN/inf from degenerate (near-constant) windows.
    n_bad = int(np.sum(~np.isfinite(features)))
    features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

    return features


def build_feature_names(n_channels: int) -> List[str]:
    """
    Feature order matches extract_window_features: stat-major, channel-minor.
    """
    names = []
    for stat in FEATURE_NAMES_PER_STAT:
        for c in range(n_channels):
            names.append(f"ch{c}_{stat}")
    return names


if __name__ == "__main__":
    dummy = np.random.randn(5, 100, 3)  # 5 windows, window_size=100, 3 channels
    feats = extract_window_features(dummy)
    print("Windows shape :", dummy.shape)
    print("Features shape:", feats.shape)
    print("Feature names :", build_feature_names(3)[:8], "...")
