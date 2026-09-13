"""
preprocessing.py

Telemetry preprocessing pipeline.

Responsibilities
----------------
✔ Data validation
✔ Missing value detection
✔ Infinite value detection
✔ Standardization
✔ Save scaler
✔ Load scaler
✔ Sliding window generation
✔ Convert to PyTorch tensors
"""

from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import torch

from sklearn.preprocessing import StandardScaler

from src.config import Config


class DataValidator:
    """
    Validate telemetry data.
    """

    @staticmethod
    def validate(data: np.ndarray) -> None:

        if not isinstance(data, np.ndarray):
            raise TypeError("Input must be a numpy array.")

        if data.ndim != 2:
            raise ValueError(
                "Telemetry data must be 2-dimensional."
            )

        if np.isnan(data).any():
            raise ValueError("NaN values detected.")

        if np.isinf(data).any():
            raise ValueError("Infinite values detected.")

        if data.shape[0] == 0:
            raise ValueError("Empty telemetry sequence.")

        if data.shape[1] == 0:
            raise ValueError("No telemetry features detected.")


class DataPreprocessor:
    """
    Handles feature scaling.
    """

    def __init__(self):

        self.scaler = StandardScaler()

    def fit_transform(self, data: np.ndarray):

        DataValidator.validate(data)

        return self.scaler.fit_transform(data)

    def transform(self, data: np.ndarray):

        DataValidator.validate(data)

        return self.scaler.transform(data)

    def save_scaler(
        self,
        filename: Optional[str] = None
    ):

        if filename is None:
            filename = Config.SCALER_NAME

        path = Config.SCALER_DIR / filename

        joblib.dump(self.scaler, path)

        print(f"Scaler saved : {path}")

    def load_scaler(
        self,
        filename: Optional[str] = None
    ):

        if filename is None:
            filename = Config.SCALER_NAME

        path = Config.SCALER_DIR / filename

        self.scaler = joblib.load(path)

        print(f"Scaler loaded : {path}")


class WindowGenerator:
    """
    Generate sliding windows.
    """

    def __init__(
        self,
        window_size: int = Config.WINDOW_SIZE
    ):

        self.window_size = window_size

    def create_windows(
        self,
        data: np.ndarray
    ) -> np.ndarray:

        DataValidator.validate(data)

        windows = []

        total_windows = (

            len(data)

            - self.window_size

            + 1

        )

        for i in range(total_windows):

            windows.append(

                data[

                    i:

                    i + self.window_size

                ]

            )

        return np.asarray(windows)

    def to_tensor(
        self,
        windows: np.ndarray
    ) -> torch.Tensor:

        return torch.tensor(

            windows,

            dtype=torch.float32

        )


if __name__ == "__main__":

    from src.dataset import NASADataset

    dataset = NASADataset()

    train, _, _ = dataset.load_channel("A-1")

    preprocessor = DataPreprocessor()

    train = preprocessor.fit_transform(train)

    preprocessor.save_scaler()

    generator = WindowGenerator()

    windows = generator.create_windows(train)

    tensor = generator.to_tensor(windows)

    print("=" * 60)

    print("Preprocessing Test")

    print("=" * 60)

    print("Original Shape :", train.shape)

    print("Windows Shape  :", windows.shape)

    print("Tensor Shape   :", tensor.shape)