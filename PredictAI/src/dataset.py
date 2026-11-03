"""
dataset.py

NASA SMAP/MSL Telemetry Dataset Loader

Responsibilities:
-----------------
✔ Load train telemetry
✔ Load test telemetry
✔ Load anomaly labels
✔ Validate dataset structure
✔ Return metadata
✔ Support loading individual channels
✔ Support loading all available channels
"""

from pathlib import Path
from typing import Dict, List, Tuple

import logging
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.config import Config


logger = logging.getLogger(__name__)


class NASADataset:
    """
    NASA SMAP/MSL Dataset Loader
    """

    def __init__(self):

        self.train_dir = Config.TRAIN_DIR
        self.test_dir = Config.TEST_DIR
        self.label_file = Config.LABEL_FILE

        self._validate_dataset()

        self.labels = pd.read_csv(self.label_file)

        logger.info("NASA Dataset Loaded Successfully")

    # =====================================================
    # Dataset Validation
    # =====================================================

    def _validate_dataset(self):

        if not self.train_dir.exists():
            raise FileNotFoundError(
                f"Training directory not found:\n{self.train_dir}"
            )

        if not self.test_dir.exists():
            raise FileNotFoundError(
                f"Testing directory not found:\n{self.test_dir}"
            )

        if not self.label_file.exists():
            raise FileNotFoundError(
                f"Label file not found:\n{self.label_file}"
            )

    # =====================================================
    # Channel Information
    # =====================================================

    def get_channel_names(self) -> List[str]:
        """
        Return all telemetry channel names.
        """

        channels = sorted(

            file.stem

            for file in self.train_dir.glob("*.npy")

        )

        return channels

    # =====================================================
    # Dataset Statistics
    # =====================================================

    def dataset_summary(self):

        channels = self.get_channel_names()

        print("=" * 60)
        print("NASA TELEMETRY DATASET")
        print("=" * 60)

        print(f"Total Channels : {len(channels)}")
        print(f"Spacecraft      : {self.labels['spacecraft'].unique()}")

        print()

        print("Channels")

        print("-" * 60)

        for channel in channels:

            train = np.load(self.train_dir / f"{channel}.npy")

            print(

                f"{channel:<6}"

                f"Shape : {train.shape}"

            )

    # =====================================================
    # Load One Channel
    # =====================================================

    def load_channel(

        self,

        channel: str,

    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:

        train_path = self.train_dir / f"{channel}.npy"

        test_path = self.test_dir / f"{channel}.npy"

        if not train_path.exists():

            raise FileNotFoundError(train_path)

        if not test_path.exists():

            raise FileNotFoundError(test_path)

        train = np.load(train_path)

        test = np.load(test_path)

        label = self.labels[

            self.labels["chan_id"] == channel

        ]

        return train, test, label

    # =====================================================
    # Load Training Data Only
    # =====================================================

    def load_train(

        self,

        channel: str,

    ) -> np.ndarray:

        path = self.train_dir / f"{channel}.npy"

        if not path.exists():

            raise FileNotFoundError(path)

        return np.load(path)

    # =====================================================
    # Load Testing Data Only
    # =====================================================

    def load_test(

        self,

        channel: str,

    ) -> np.ndarray:

        path = self.test_dir / f"{channel}.npy"

        if not path.exists():

            raise FileNotFoundError(path)

        return np.load(path)

    # =====================================================
    # Load Labels Only
    # =====================================================

    def load_labels(

        self,

        channel: str,

    ) -> pd.DataFrame:

        return self.labels[

            self.labels["chan_id"] == channel

        ]

    # =====================================================
    # Metadata
    # =====================================================

    def get_metadata(

        self,

        channel: str,

    ) -> Dict:

        labels = self.load_labels(channel)

        if labels.empty:

            raise ValueError(

                f"No metadata found for channel {channel}"

            )

        row = labels.iloc[0]

        return {

            "channel": row["chan_id"],

            "spacecraft": row["spacecraft"],

            "class": row["class"],

            "num_values": int(row["num_values"]),

            "anomaly_sequences": row["anomaly_sequences"],

        }

    # =====================================================
    # Load Entire Dataset
    # =====================================================

    def load_all(self):

        dataset = {}

        channels = self.get_channel_names()

        for channel in channels:

            dataset[channel] = {

                "train": self.load_train(channel),

                "test": self.load_test(channel),

                "labels": self.load_labels(channel),

            }

        return dataset


# ==========================================================
# Test
# ==========================================================

if __name__ == "__main__":

    dataset = NASADataset()

    dataset.dataset_summary()

    print()

    print("=" * 60)

    print("Example")

    print("=" * 60)

    train, test, labels = dataset.load_channel("A-1")

    print("Train Shape :", train.shape)

    print("Test Shape  :", test.shape)

    print()

    print(labels)

    print()

    print(dataset.get_metadata("A-1"))


class TelemetryDataset(Dataset):
    """
    Converts NumPy windowed telemetry data into a PyTorch Dataset.
    """
    def __init__(self, data):
        # Support both NumPy array and PyTorch tensor inputs
        if isinstance(data, torch.Tensor):
            self.data = data.clone().detach().float()
        else:
            self.data = torch.tensor(data, dtype=torch.float32)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]