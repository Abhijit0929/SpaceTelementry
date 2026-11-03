"""
channels.py (API route)

Auto-discovers available telemetry channels from the dataset itself
(labeled_anomalies.csv via NASADataset) instead of requiring the
frontend/user to already know channel ids.
"""

from pathlib import Path
import sys

from fastapi import APIRouter

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import NASADataset


router = APIRouter(
    prefix="/api/channels",
    tags=["Channels"]
)

_dataset = None


def get_dataset() -> NASADataset:
    global _dataset
    if _dataset is None:
        _dataset = NASADataset()
    return _dataset


@router.get("")
def list_channels():
    dataset = get_dataset()
    channel_names = dataset.get_channel_names()

    channels = []
    for channel in channel_names:
        label_rows = dataset.load_labels(channel)
        spacecraft = str(label_rows.iloc[0]["spacecraft"]) if not label_rows.empty else "UNKNOWN"
        channels.append({"channel": channel, "spacecraft": spacecraft})

    return {
        "count": len(channels),
        "channels": channels,
    }
