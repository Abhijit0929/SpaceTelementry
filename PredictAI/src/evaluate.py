"""
evaluate.py

Evaluate anomaly detection using window-level ground truth.
"""

import ast
import json
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)
from src.config import Config
from src.dataset import NASADataset


def build_window_ground_truth(num_windows, anomaly_info):
    """
    Convert NASA anomaly intervals into window labels.

    A window is anomalous if ANY part of the window overlaps
    with an anomaly interval.
    """
    ground_truth = np.zeros(num_windows, dtype=int)

    if anomaly_info.empty:
        return ground_truth

    intervals = ast.literal_eval(
        anomaly_info.iloc[0]["anomaly_sequences"]
    )

    window_size = Config.WINDOW_SIZE
    for i in range(num_windows):
        window_start = i
        window_end = i + window_size - 1
        for anomaly_start, anomaly_end in intervals:
            overlap = (
                window_start <= anomaly_end
                and window_end >= anomaly_start
            )
            if overlap:
                ground_truth[i] = 1
                break

    return ground_truth


def main(channel: str = "A-1"):
    print("=" * 60)
    print(f"Evaluating Anomaly Detection for Channel {channel}")
    print("=" * 60)

    predictions_path = Config.PREDICTION_DIR / f"{channel}_predictions.npy"
    if not predictions_path.exists():
        raise FileNotFoundError(f"Predictions not found at {predictions_path}")

    predictions = np.load(predictions_path)
    
    dataset = NASADataset()
    _, _, label_df = dataset.load_channel(channel)

    ground_truth = build_window_ground_truth(
        len(predictions),
        label_df,
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

    print("\nResults")
    print("-" * 40)
    print(f"Windows            : {len(predictions)}")
    print(f"Ground Truth Count : {ground_truth.sum()}")
    print(f"Predicted Count    : {predictions.sum()}")

    print(f"\nPrecision : {precision:.4f}")
    print(f"Recall    : {recall:.4f}")
    print(f"F1 Score  : {f1:.4f}")

    print("\nConfusion Matrix")
    print(cm)

    report_str = classification_report(
        ground_truth,
        predictions,
        zero_division=0,
    )
    print("\nClassification Report")
    print(report_str)

    # Save metrics JSON
    metrics_path = Config.METRICS_DIR / f"{channel}_metrics.json"
    metrics_data = {
        "channel": channel,
        "windows": len(predictions),
        "ground_truth_anomalies": int(ground_truth.sum()),
        "predicted_anomalies": int(predictions.sum()),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": cm.tolist()
    }
    
    with open(metrics_path, "w") as f:
        json.dump(metrics_data, f, indent=4)
        
    print(f"\n[OK] Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    import sys
    target_channel = sys.argv[1] if len(sys.argv) > 1 else "A-1"
    main(target_channel)
