"""
threshold.py

Compute anomaly threshold from reconstruction errors.
"""

import numpy as np
from src.config import Config


class ThresholdCalculator:
    def __init__(
        self,
        method: str = Config.THRESHOLD_METHOD,
        percentile: float = Config.THRESHOLD_PERCENTILE,
        std_factor: float = Config.STD_FACTOR
    ):
        self.method = method
        self.percentile = percentile
        self.std_factor = std_factor

    def calculate(self, reconstruction_errors):
        if self.method == "mean_std":
            threshold = (
                np.mean(reconstruction_errors)
                + self.std_factor * np.std(reconstruction_errors)
            )
        elif self.method == "percentile":
            threshold = np.percentile(reconstruction_errors, self.percentile)
        else:
            raise ValueError("Unknown threshold method.")

        return threshold


def main(channel: str = "A-1"):
    # Load errors (we can use train errors to calculate the threshold)
    errors_path = Config.PREDICTION_DIR / f"{channel}_train_errors.npy"
    if not errors_path.exists():
        # Fall back to test errors if train errors are not found
        errors_path = Config.PREDICTION_DIR / f"{channel}_test_errors.npy"
        
    if not errors_path.exists():
        raise FileNotFoundError(f"Reconstruction errors not found at {errors_path}")

    errors = np.load(errors_path)
    calculator = ThresholdCalculator()
    threshold = calculator.calculate(errors)

    print("=" * 50)
    print(f"Threshold Calculation for Channel {channel}")
    print("=" * 50)
    print(f"Method     : {calculator.method}")
    print(f"Mean Error : {errors.mean():.6f}")
    print(f"Std Error  : {errors.std():.6f}")
    print(f"Threshold  : {threshold:.6f}")

    threshold_path = Config.PREDICTION_DIR / f"{channel}_threshold.npy"
    np.save(threshold_path, np.array([threshold]))

    print(f"\n[OK] Threshold saved at: {threshold_path}")


if __name__ == "__main__":
    import sys
    target_channel = sys.argv[1] if len(sys.argv) > 1 else "A-1"
    main(target_channel)
