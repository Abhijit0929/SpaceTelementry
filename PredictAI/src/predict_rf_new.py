"""
predict_rf.py

Run the trained Random Forest anomaly classifier on a single channel's
telemetry and save window-level predictions + probabilities.

Loads the spacecraft-specific model AND its tuned decision threshold
(saved by train_rf.py) - predictions use that threshold instead of the
sklearn default of 0.5.

Run from the project root:
    python -m src.predict_rf --channel A-1
"""

import argparse

import numpy as np

from src.config import Config
from src.dataset import NASADataset
from src.preprocessing import DataPreprocessor, WindowGenerator
from src.rf_features import extract_window_features
from src.rf_model import load_rf_model, load_rf_threshold


def predict_channel(channel: str):
    dataset = NASADataset()
    train, test, _ = dataset.load_channel(channel)

    label_rows = dataset.load_labels(channel)
    if label_rows.empty:
        raise ValueError(f"No entry for channel '{channel}' in labeled_anomalies.csv")
    spacecraft = label_rows.iloc[0]["spacecraft"]

    preprocessor = DataPreprocessor()
    generator = WindowGenerator()

    # Refit the scaler on this channel's train data - identical to how
    # train_rf.py scaled it - then apply to test only. Never fit on test.
    preprocessor.fit_transform(train)
    test_scaled = preprocessor.transform(test)

    test_windows = generator.create_windows(test_scaled)
    X_test = extract_window_features(test_windows)

    model_filename = f"random_forest_{spacecraft}.joblib"
    model = load_rf_model(filename=model_filename)
    threshold = load_rf_threshold(filename=model_filename)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    pred_path = Config.PREDICTION_DIR / f"{channel}_rf_predictions.npy"
    prob_path = Config.PREDICTION_DIR / f"{channel}_rf_probabilities.npy"

    np.save(pred_path, predictions)
    np.save(prob_path, probabilities)

    print(f"Channel               : {channel}  ({spacecraft})")
    print(f"Model used            : {model_filename}")
    print(f"Threshold used        : {threshold:.4f}")
    print(f"Windows               : {len(predictions)}")
    print(f"Flagged as anomalous  : {int(predictions.sum())}")
    print(f"[OK] Predictions saved   : {pred_path}")
    print(f"[OK] Probabilities saved : {prob_path}")

    return predictions, probabilities


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RF anomaly prediction for one channel.")
    parser.add_argument("--channel", type=str, default="A-1", help="Channel id, e.g. A-1")
    args = parser.parse_args()

    predict_channel(args.channel)
