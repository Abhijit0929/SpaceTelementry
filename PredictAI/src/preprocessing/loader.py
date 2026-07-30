"""
loader.py

Production-ready data loader for the NASA SMAP/MSL telemetry dataset.
"""

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.config.config import TRAIN_DIR, TEST_DIR, LABEL_FILE


class DataLoader:
    """
    Loads NASA SMAP/MSL telemetry data.
    """

    def __init__(self):
        self.train_dir: Path = TRAIN_DIR
        self.test_dir: Path = TEST_DIR
        self.label_file: Path = LABEL_FILE

        if not self.label_file.exists():
            raise FileNotFoundError(
                f"Label file not found: {self.label_file}"
            )

        self.labels = pd.read_csv(self.label_file)

    def get_channel_names(self) -> List[str]:
        """
        Returns all telemetry channel names.
        """
        return sorted(file.stem for file in self.train_dir.glob("*.npy"))

    def load_channel(
        self,
        channel_name: str
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Load train and test data for one telemetry channel.

        Parameters
        ----------
        channel_name : str
            Example: "A-1"

        Returns
        -------
        train_data
        test_data
        anomaly_labels
        """

        train_path = self.train_dir / f"{channel_name}.npy"
        test_path = self.test_dir / f"{channel_name}.npy"

        if not train_path.exists():
            raise FileNotFoundError(train_path)

        if not test_path.exists():
            raise FileNotFoundError(test_path)

        train_data = np.load(train_path)
        test_data = np.load(test_path)

        anomaly_info = self.labels[
            self.labels["chan_id"] == channel_name
        ]

        return train_data, test_data, anomaly_info

    def load_all_channels(self):
        """
        Load every telemetry channel.

        Returns
        -------
        train_dataset : dict
        test_dataset : dict
        """

        train_dataset = {}
        test_dataset = {}

        channels = self.get_channel_names()

        for channel in channels:

            train, test, _ = self.load_channel(channel)

            train_dataset[channel] = train
            test_dataset[channel] = test

        return train_dataset, test_dataset