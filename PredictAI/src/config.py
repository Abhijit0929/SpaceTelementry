"""
config.py

Central configuration file for the Spacecraft Intelligence project.
All configurable parameters should be defined here.
"""

from pathlib import Path
import torch


class Config:
    """Central project configuration."""

    # ==========================================================
    # Project Directories
    # ==========================================================

    ROOT_DIR = Path(__file__).resolve().parent.parent

    DATA_DIR = ROOT_DIR / "data"

    RAW_DATA_DIR = DATA_DIR / "raw"
    TRAIN_DIR = RAW_DATA_DIR / "train"
    TEST_DIR = RAW_DATA_DIR / "test"
    LABEL_FILE = RAW_DATA_DIR / "labeled_anomalies.csv"

    PROCESSED_DATA_DIR = DATA_DIR / "processed"

    KNOWLEDGE_BASE_DIR = DATA_DIR / "knowledge_base"

    MODEL_DIR = ROOT_DIR / "models"
    TRAINED_MODEL_DIR = MODEL_DIR / "trained"
    SCALER_DIR = MODEL_DIR / "scaler"

    OUTPUT_DIR = ROOT_DIR / "outputs"
    PREDICTION_DIR = OUTPUT_DIR / "predictions"
    METRICS_DIR = OUTPUT_DIR / "metrics"
    PLOTS_DIR = OUTPUT_DIR / "plots"
    LOG_DIR = OUTPUT_DIR / "logs"

    # ==========================================================
    # Training Parameters
    # ==========================================================

    WINDOW_SIZE = 100

    BATCH_SIZE = 64

    EPOCHS = 75

    LEARNING_RATE = 1e-3

    WEIGHT_DECAY = 1e-5

    RANDOM_SEED = 42

    NUM_WORKERS = 0

    SHUFFLE = True

    # ==========================================================
    # Model Parameters
    # ==========================================================

    HIDDEN_SIZE = 64

    LATENT_SIZE = 32

    NUM_LAYERS = 2

    DROPOUT = 0.2

    # ==========================================================
    # Threshold Configuration
    # ==========================================================

    THRESHOLD_METHOD = "percentile"

    THRESHOLD_PERCENTILE = 99

    STD_FACTOR = 3

    # ==========================================================
    # Device Configuration
    # ==========================================================

    DEVICE = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    # ==========================================================
    # Logging
    # ==========================================================

    LOG_LEVEL = "INFO"

    LOG_FILE = LOG_DIR / "project.log"

    # ==========================================================
    # File Names
    # ==========================================================

    MODEL_NAME = "lstm_autoencoder.pth"

    SCALER_NAME = "scaler.pkl"

    THRESHOLD_NAME = "threshold.npy"

    TRAIN_ERRORS_NAME = "train_reconstruction_errors.npy"

    TEST_ERRORS_NAME = "test_reconstruction_errors.npy"

    PREDICTIONS_NAME = "predictions.npy"

    METRICS_NAME = "metrics.json"

    # ==========================================================
    # Utility
    # ==========================================================

    @classmethod
    def create_directories(cls):
        """
        Create all required project directories.
        """

        directories = [
            cls.TRAINED_MODEL_DIR,
            cls.SCALER_DIR,
            cls.PREDICTION_DIR,
            cls.METRICS_DIR,
            cls.PLOTS_DIR,
            cls.LOG_DIR,
            cls.PROCESSED_DATA_DIR,
            cls.KNOWLEDGE_BASE_DIR,
        ]

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


# ==========================================================
# Automatically create required folders
# ==========================================================

Config.create_directories()


if __name__ == "__main__":

    print("=" * 60)
    print("Spacecraft Intelligence Configuration")
    print("=" * 60)

    print(f"Project Root : {Config.ROOT_DIR}")
    print(f"Training Dir : {Config.TRAIN_DIR}")
    print(f"Test Dir     : {Config.TEST_DIR}")
    print(f"Device       : {Config.DEVICE}")
    print(f"Epochs       : {Config.EPOCHS}")
    print(f"Batch Size   : {Config.BATCH_SIZE}")
    print(f"Window Size  : {Config.WINDOW_SIZE}")