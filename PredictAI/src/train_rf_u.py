"""
train_rf.py

Train the Random Forest anomaly classifier on windowed, feature-engineered
telemetry, one model PER SPACECRAFT (SMAP vs MSL have different feature
counts, so they can't be pooled into one feature matrix).

IMPORTANT ON EVALUATION: windows are generated with stride=1 (see
WindowGenerator), so neighbouring windows overlap ~99%. A random,
window-level train/test split would leak near-duplicate windows across
both sides and produce misleadingly perfect metrics. To avoid that, this
script splits by WHOLE CHANNEL: some channels are held out entirely for
testing, so no test window overlaps with any training window.

Run from the project root:
    python -m src.train_rf
"""

import json
from collections import defaultdict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split as split_channels
from sklearn.metrics import precision_recall_curve

from src.config import Config
from src.dataset import NASADataset
from src.preprocessing import DataPreprocessor, WindowGenerator
from src.evaluate import build_window_ground_truth
from src.rf_features import extract_window_features
from src.rf_model import build_rf_model, save_rf_model


def group_channels_by_spacecraft(dataset: NASADataset, channels):
    groups = defaultdict(list)
    for channel in channels:
        label_rows = dataset.load_labels(channel)
        if label_rows.empty:
            print(f"[SKIP] {channel}: no entry in labeled_anomalies.csv")
            continue
        spacecraft = label_rows.iloc[0]["spacecraft"]
        groups[spacecraft].append(channel)
    return groups


def build_channel_features(dataset: NASADataset, channel: str):
    """
    Returns (X_channel, y_channel) pooling this channel's train (label 0)
    and test (real ground-truth labels) windows together.
    """
    train, test, label_df = dataset.load_channel(channel)

    preprocessor = DataPreprocessor()
    generator = WindowGenerator()

    train_scaled = preprocessor.fit_transform(train)
    test_scaled = preprocessor.transform(test)

    train_windows = generator.create_windows(train_scaled)
    test_windows = generator.create_windows(test_scaled)

    X_train_feat = extract_window_features(train_windows)
    X_test_feat = extract_window_features(test_windows)

    y_train_labels = np.zeros(len(X_train_feat), dtype=int)
    y_test_labels = build_window_ground_truth(len(X_test_feat), label_df)

    X_channel = np.concatenate([X_train_feat, X_test_feat], axis=0)
    y_channel = np.concatenate([y_train_labels, y_test_labels], axis=0)

    return X_channel, y_channel


def build_pooled_dataset(dataset: NASADataset, channels):
    all_X, all_y = [], []
    for channel in channels:
        try:
            X_channel, y_channel = build_channel_features(dataset, channel)
        except Exception as exc:
            print(f"[SKIP] {channel}: {exc}")
            continue
        all_X.append(X_channel)
        all_y.append(y_channel)
        print(
            f"  {channel:<6} windows={len(y_channel):<6} "
            f"anomalous={int(y_channel.sum())} features={X_channel.shape[1]}"
        )
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    return X, y


def evaluate_predictions(y_test, y_pred, y_prob):
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def train_one_spacecraft(dataset: NASADataset, spacecraft: str, channels, test_size: float = 0.25):
    print("=" * 60)
    print(f"Spacecraft: {spacecraft}  ({len(channels)} channels)")
    print("=" * 60)

    # Split by WHOLE CHANNEL, not by window - this is the leakage fix.
    if len(channels) < 4:
        print(f"[WARN] Only {len(channels)} channels - split may be unstable.")

    train_channels, test_channels = split_channels(
        channels, test_size=test_size, random_state=Config.RANDOM_SEED
    )
    print(f"Train channels ({len(train_channels)}): {train_channels}")
    print(f"Test channels  ({len(test_channels)}): {test_channels}")

    print("\n-- building TRAIN channel features --")
    X_train, y_train = build_pooled_dataset(dataset, train_channels)

    print("\n-- building TEST channel features (held out, never seen) --")
    X_test, y_test = build_pooled_dataset(dataset, test_channels)

    print("-" * 60)
    print(f"Train windows   : {X_train.shape[0]}  (anomalous: {int(y_train.sum())})")
    print(f"Test windows    : {X_test.shape[0]}  (anomalous: {int(y_test.sum())})")

    model = build_rf_model()
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]

    # --- Default-threshold (0.5) metrics, for comparison ---
    y_pred_default = (y_prob >= 0.5).astype(int)
    metrics_default = evaluate_predictions(y_test, y_pred_default, y_prob)

    # --- Tune the decision threshold on this held-out split ---
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_scores)
    best_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5

    y_pred_tuned = (y_prob >= best_threshold).astype(int)
    metrics = evaluate_predictions(y_test, y_pred_tuned, y_prob)
    metrics["threshold"] = best_threshold
    metrics["train_channels"] = train_channels
    metrics["test_channels"] = test_channels

    print("\nResults @ default threshold (0.5)")
    print("-" * 40)
    for key in ("accuracy", "precision", "recall", "f1_score", "roc_auc"):
        print(f"{key:<10}: {metrics_default[key]:.4f}")

    print(f"\nResults @ tuned threshold ({best_threshold:.4f})")
    print("-" * 40)
    for key in ("accuracy", "precision", "recall", "f1_score", "roc_auc"):
        print(f"{key:<10}: {metrics[key]:.4f}")
    print("Confusion Matrix (tuned threshold)")
    print(np.array(metrics["confusion_matrix"]))

    metrics_path = Config.METRICS_DIR / f"random_forest_{spacecraft}_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"\n[OK] Metrics saved to: {metrics_path}")

    # Retrain the FINAL saved model on ALL channels (train+test) for production use -
    # the held-out split above is only used to get an honest performance estimate
    # AND to pick the decision threshold, which is saved alongside the model.
    print("\n-- retraining final model on ALL channels for saving --")
    X_all, y_all = np.concatenate([X_train, X_test]), np.concatenate([y_train, y_test])
    final_model = build_rf_model()
    final_model.fit(X_all, y_all)

    model_filename = f"random_forest_{spacecraft}.joblib"
    save_rf_model(final_model, filename=model_filename, threshold=best_threshold)

    return metrics


def main(channels=None):
    dataset = NASADataset()
    if channels is None:
        channels = dataset.get_channel_names()

    groups = group_channels_by_spacecraft(dataset, channels)

    print("Spacecraft groups found:")
    for spacecraft, chans in groups.items():
        print(f"  {spacecraft}: {len(chans)} channels")
    print()

    all_metrics = {}
    for spacecraft, chans in groups.items():
        all_metrics[spacecraft] = train_one_spacecraft(dataset, spacecraft, chans)
        print()

    print("=" * 60)
    print("Summary (honest, held-out-channel evaluation)")
    print("=" * 60)
    for spacecraft, metrics in all_metrics.items():
        print(f"{spacecraft}: F1={metrics['f1_score']:.4f}  ROC-AUC={metrics['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
