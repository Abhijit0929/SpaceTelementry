"""
detector.py

Detect anomalies using reconstruction errors.
"""

import numpy as np
from src.config import Config


class AnomalyDetector:
    def __init__(self, threshold):
        self.threshold = threshold

    def detect(self, reconstruction_errors):
        predictions = reconstruction_errors > self.threshold
        return predictions.astype(int)


def main(channel: str = "A-1"):
    errors_path = Config.PREDICTION_DIR / f"{channel}_test_errors.npy"
    threshold_path = Config.PREDICTION_DIR / f"{channel}_threshold.npy"

    if not errors_path.exists():
        raise FileNotFoundError(f"Reconstruction errors not found at {errors_path}")
    if not threshold_path.exists():
        raise FileNotFoundError(f"Threshold file not found at {threshold_path}")

    reconstruction_errors = np.load(errors_path)
    threshold = np.load(threshold_path)[0]

    detector = AnomalyDetector(threshold)
    predictions = detector.detect(reconstruction_errors)

    predictions_path = Config.PREDICTION_DIR / f"{channel}_predictions.npy"
    np.save(
        predictions_path,
        predictions,
    )

    print("=" * 50)
    print(f"Anomaly Detection for Channel {channel}")
    print("=" * 50)
    print(f"Threshold : {threshold:.6f}")
    print(f"Detected anomalies : {predictions.sum()}")
    print(f"\n[OK] Predictions saved at: {predictions_path}")


if __name__ == "__main__":
    import sys
    target_channel = sys.argv[1] if len(sys.argv) > 1 else "A-1"
    main(target_channel)
