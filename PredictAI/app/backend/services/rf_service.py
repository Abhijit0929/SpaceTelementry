from pathlib import Path
import sys
import numpy as np


# ---------------------------------------------------------
# Project root
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------
# Existing PredictAI modules
# ---------------------------------------------------------

from src.predict_rf_new import predict_channel
from src.dataset import NASADataset
from src.config import Config


# ---------------------------------------------------------
# Random Forest prediction service
# ---------------------------------------------------------

def run_prediction(channel: str):

    predictions, probabilities = predict_channel(channel)

    # Determine spacecraft
    dataset = NASADataset()

    label_rows = dataset.load_labels(channel)

    if label_rows.empty:
        raise ValueError(
            f"No entry found for channel '{channel}'"
        )

    spacecraft = str(
        label_rows.iloc[0]["spacecraft"]
    )

    # Model name
    model_name = f"random_forest_{spacecraft}.joblib"

    # Saved files
    prediction_path = (
        Config.PREDICTION_DIR
        / f"{channel}_rf_predictions.npy"
    )

    probability_path = (
        Config.PREDICTION_DIR
        / f"{channel}_rf_probabilities.npy"
    )

    total_windows = len(predictions)

    anomalous_windows = int(
        np.sum(predictions)
    )

    anomaly_rate = (
        anomalous_windows / total_windows * 100
        if total_windows > 0
        else 0
    )

    return {
        "channel": channel,
        "spacecraft": spacecraft,
        "model": model_name,
        "total_windows": total_windows,
        "anomalous_windows": anomalous_windows,
        "normal_windows": total_windows - anomalous_windows,
        "anomaly_rate": round(anomaly_rate, 2),
        "predictions": predictions.tolist(),
        "probabilities": probabilities.tolist(),
        "prediction_file": str(prediction_path),
        "probability_file": str(probability_path),
    }