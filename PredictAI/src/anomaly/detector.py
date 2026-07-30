"""
detector.py

Detect anomalies using reconstruction errors.
"""

import numpy as np


class AnomalyDetector:

    def __init__(self, threshold):
        self.threshold = threshold

    def detect(self, reconstruction_errors):

        predictions = reconstruction_errors > self.threshold

        return predictions.astype(int)


if __name__ == "__main__":

    reconstruction_errors = np.load(
        "models/reconstruction_errors.npy"
    )

    threshold = np.load(
        "outputs/threshold.npy"
    )[0]

    detector = AnomalyDetector(threshold)

    predictions = detector.detect(reconstruction_errors)

    np.save(
        "outputs/predictions.npy",
        predictions,
    )

    print("=" * 50)
    print("Anomaly Detection")
    print("=" * 50)

    print(f"Threshold : {threshold:.6f}")
    print(f"Detected anomalies : {predictions.sum()}")

    print("\nPredictions saved.")