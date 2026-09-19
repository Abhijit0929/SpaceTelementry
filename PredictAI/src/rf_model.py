"""
rf_model.py

Random Forest anomaly classifier - build, save, and load.
Also saves/loads the tuned decision threshold alongside each model,
so predict_rf.py doesn't fall back to the default (and usually wrong) 0.5.

Mirrors the persistence pattern already used for the scaler
(src/preprocessing.py) and the LSTM autoencoder (src/model.py).
"""

import json
from pathlib import Path
from typing import Optional

import joblib
from sklearn.ensemble import RandomForestClassifier

from src.config import Config

# Random Forest specific settings, kept local to this file so no edits
# to config.py are required to add this model to the pipeline.
RF_N_ESTIMATORS = 100
RF_CLASS_WEIGHT = "balanced"
RF_RANDOM_STATE = Config.RANDOM_SEED
RF_MODEL_NAME = "random_forest.joblib"
RF_DEFAULT_THRESHOLD = 0.5


def build_rf_model() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        class_weight=RF_CLASS_WEIGHT,
        random_state=RF_RANDOM_STATE,
        n_jobs=-1,
    )


def _threshold_path_for(model_path: Path) -> Path:
    # random_forest_MSL.joblib -> random_forest_MSL.threshold.json
    return model_path.with_suffix(".threshold.json")


def save_rf_model(
    model: RandomForestClassifier,
    filename: Optional[str] = None,
    threshold: Optional[float] = None,
) -> Path:
    if filename is None:
        filename = RF_MODEL_NAME

    path = Config.TRAINED_MODEL_DIR / filename
    joblib.dump(model, path)
    print(f"Random Forest model saved : {path}")

    if threshold is not None:
        threshold_path = _threshold_path_for(path)
        with open(threshold_path, "w") as f:
            json.dump({"threshold": float(threshold)}, f, indent=4)
        print(f"Decision threshold saved  : {threshold_path} (threshold={threshold:.4f})")

    return path


def load_rf_model(filename: Optional[str] = None) -> RandomForestClassifier:
    if filename is None:
        filename = RF_MODEL_NAME

    path = Config.TRAINED_MODEL_DIR / filename
    if not path.exists():
        raise FileNotFoundError(
            f"No trained Random Forest model found at {path}. Run train_rf.py first."
        )

    model = joblib.load(path)
    print(f"Random Forest model loaded : {path}")
    return model


def load_rf_threshold(filename: Optional[str] = None) -> float:
    """
    Loads the tuned decision threshold saved alongside a model.
    Falls back to RF_DEFAULT_THRESHOLD (0.5) if none was saved -
    e.g. for a model trained before threshold tuning was added.
    """
    if filename is None:
        filename = RF_MODEL_NAME

    path = Config.TRAINED_MODEL_DIR / filename
    threshold_path = _threshold_path_for(path)

    if not threshold_path.exists():
        print(
            f"[WARN] No saved threshold at {threshold_path} - "
            f"using default {RF_DEFAULT_THRESHOLD}"
        )
        return RF_DEFAULT_THRESHOLD

    with open(threshold_path, "r") as f:
        data = json.load(f)

    threshold = float(data["threshold"])
    print(f"Decision threshold loaded : {threshold_path} (threshold={threshold:.4f})")
    return threshold
