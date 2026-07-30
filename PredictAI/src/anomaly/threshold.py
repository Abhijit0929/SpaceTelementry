"""
threshold.py

Compute anomaly threshold from reconstruction errors.
"""

import numpy as np


class ThresholdCalculator:

    def __init__(self, method="mean_std", std_factor=3):
        self.method = method
        self.std_factor = std_factor

    def calculate(self, reconstruction_errors):

        if self.method == "mean_std":

            threshold = (
                np.mean(reconstruction_errors)
                + self.std_factor * np.std(reconstruction_errors)
            )

        elif self.method == "percentile":

            threshold = np.percentile(reconstruction_errors, 99)

        else:
            raise ValueError("Unknown threshold method.")

        return threshold


if __name__ == "__main__":

    errors = np.load("models/reconstruction_errors.npy")

    calculator = ThresholdCalculator()

    threshold = calculator.calculate(errors)

    print("=" * 50)
    print("Threshold Calculation")
    print("=" * 50)

    print(f"Mean Error : {errors.mean():.6f}")
    print(f"Std Error  : {errors.std():.6f}")
    print(f"Threshold  : {threshold:.6f}")

    np.save("outputs/threshold.npy", np.array([threshold]))

    print("\nThreshold saved.")