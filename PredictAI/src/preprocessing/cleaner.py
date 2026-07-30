"""
cleaner.py

Data preprocessing for telemetry signals.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler


class DataCleaner:

    def __init__(self):

        self.scaler = StandardScaler()

    def fit_transform(self, train_data):

        self._validate(train_data)

        return self.scaler.fit_transform(train_data)

    def transform(self, test_data):

        self._validate(test_data)

        return self.scaler.transform(test_data)

    @staticmethod
    def _validate(data):

        if np.isnan(data).any():
            raise ValueError("NaN values detected.")

        if np.isinf(data).any():
            raise ValueError("Infinite values detected.")