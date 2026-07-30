"""
evaluate.py

Evaluate anomaly detection performance.
"""

import ast

import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.preprocessing.loader import DataLoader


def build_ground_truth(length, anomaly_info):
    """
    Convert NASA anomaly intervals into a binary vector.
    """

    ground_truth = np.zeros(length, dtype=int)

    if anomaly_info.empty:
        return ground_truth

    intervals = ast.literal_eval(
        anomaly_info.iloc[0]["anomaly_sequences"]
    )

    for start, end in intervals:

        start = max(0, start)
        end = min(length - 1, end)

        ground_truth[start:end + 1] = 1

    return ground_truth


def main():

    predictions = np.load(
        "outputs/predictions.npy"
    )

    loader = DataLoader()

    channel = loader.get_channel_names()[0]

    _, _, anomaly_info = loader.load_channel(channel)

    ground_truth = build_ground_truth(
        len(predictions),
        anomaly_info,
    )

    precision = precision_score(
        ground_truth,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        ground_truth,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        ground_truth,
        predictions,
        zero_division=0,
    )

    cm = confusion_matrix(
        ground_truth,
        predictions,
    )

    print("=" * 60)
    print("Evaluation Results")
    print("=" * 60)

    print(f"Precision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\nConfusion Matrix")

    print(cm)


if __name__ == "__main__":
    main()