from pathlib import Path

# Project Root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Dataset Paths
DATA_DIR = PROJECT_ROOT / "data" / "raw" / "nasa_smap_msl"

TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"
LABEL_FILE = DATA_DIR / "labeled_anomalies.csv"

# Output Paths
OUTPUT_DIR = PROJECT_ROOT / "outputs"
PLOT_DIR = OUTPUT_DIR / "plots"
LOG_DIR = OUTPUT_DIR / "logs"
MODEL_DIR = PROJECT_ROOT / "models"

OUTPUT_DIR.mkdir(exist_ok=True)
PLOT_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)