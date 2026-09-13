"""
visualize.py

Visualize reconstruction errors, threshold, and ground truth anomalies.
"""

import ast
import numpy as np
import matplotlib.pyplot as plt
from src.config import Config
from src.dataset import NASADataset


def main(channel: str = "A-1"):
    errors_path = Config.PREDICTION_DIR / f"{channel}_test_errors.npy"
    threshold_path = Config.PREDICTION_DIR / f"{channel}_threshold.npy"

    if not errors_path.exists():
        raise FileNotFoundError(f"Reconstruction errors not found at {errors_path}")
    if not threshold_path.exists():
        raise FileNotFoundError(f"Threshold file not found at {threshold_path}")

    errors = np.load(errors_path)
    threshold = np.load(threshold_path)[0]

    dataset = NASADataset()
    _, _, label_df = dataset.load_channel(channel)

    intervals = []
    if not label_df.empty:
        intervals = ast.literal_eval(label_df.iloc[0]["anomaly_sequences"])

    plt.figure(figsize=(15, 5))
    plt.plot(errors, label="Reconstruction Error", color="#2c3e50")
    plt.axhline(
        threshold,
        color="red",
        linestyle="--",
        label=f"Threshold ({threshold:.4f})"
    )

    window_size = Config.WINDOW_SIZE
    for start, end in intervals:
        left = max(0, start - window_size + 1)
        right = min(len(errors), end)
        plt.axvspan(
            left,
            right,
            color="orange",
            alpha=0.3,
            label="Ground Truth Anomaly" if 'Ground Truth Anomaly' not in plt.gca().get_legend_handles_labels()[1] else ""
        )

    plt.title(f"Reconstruction Errors vs Ground Truth (Channel {channel})")
    plt.xlabel("Window")
    plt.ylabel("Reconstruction Error")
    plt.legend()
    plt.tight_layout()

    plot_path = Config.PLOTS_DIR / f"{channel}_reconstruction_errors.png"
    plt.savefig(plot_path)
    plt.close()
    
    print(f"[OK] Plot successfully saved to: {plot_path}")


if __name__ == "__main__":
    import sys
    target_channel = sys.argv[1] if len(sys.argv) > 1 else "A-1"
    main(target_channel)
