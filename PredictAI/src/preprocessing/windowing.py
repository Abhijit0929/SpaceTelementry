"""
windowing.py

Sliding window generator for LSTM.
"""

import numpy as np


class WindowGenerator:

    def __init__(self, window_size=100):

        self.window_size = window_size

    def create(self, data):

        windows = []

        for i in range(len(data) - self.window_size + 1):

            windows.append(
                data[i:i + self.window_size]
            )

        return np.array(windows)