import numpy as np
import pandas as pd

from src.config.config import TRAIN_DIR, TEST_DIR, LABEL_FILE


class DatasetAnalyzer:

    def __init__(self):
        self.train_files = sorted(TRAIN_DIR.glob("*.npy"))
        self.test_files = sorted(TEST_DIR.glob("*.npy"))
        self.labels = pd.read_csv(LABEL_FILE)

    def analyze(self):

        print("=" * 60)
        print("NASA SMAP/MSL DATASET ANALYSIS")
        print("=" * 60)

        print(f"Train Files : {len(self.train_files)}")
        print(f"Test Files  : {len(self.test_files)}")
        print(f"Label Rows  : {len(self.labels)}")

        print("\nFirst 5 Channels")
        print("-" * 60)

        for file in self.train_files[:5]:

            data = np.load(file)

            print(f"\nFile : {file.name}")
            print(f"Shape : {data.shape}")
            print(f"Min   : {data.min():.4f}")
            print(f"Max   : {data.max():.4f}")
            print(f"Mean  : {data.mean():.4f}")
            print(f"Std   : {data.std():.4f}")
            print(f"NaNs  : {np.isnan(data).sum()}")

        print("\nDone!")


if __name__ == "__main__":
    analyzer = DatasetAnalyzer()
    analyzer.analyze()