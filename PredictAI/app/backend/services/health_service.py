"""
health_service.py

Single place responsible for turning raw RF prediction output
(from rf_service.run_prediction) into a "telemetry health" score.

IMPORTANT - what this score IS and ISN'T:
  health_score = normal_windows / total_windows * 100

This is a TELEMETRY ANOMALY score, not a validated physical spacecraft
health measurement. It reflects how much of the processed telemetry the
RF model flagged as statistically anomalous - nothing here has been
calibrated against real engineering safety limits. The dashboard must
always label this "Telemetry Health Score", never "Spacecraft Physical
Health" - that distinction is load-bearing, not cosmetic.

Status thresholds below are dashboard display thresholds chosen for this
project, not validated operational safety limits.
"""

from pathlib import Path
import sys
from typing import Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import NASADataset
from app.backend.services.rf_service import run_prediction


# Dashboard display thresholds - NOT validated physical safety limits.
HEALTHY_MIN = 95.0
WARNING_MIN = 80.0


def compute_health_score(total_windows: int, anomalous_windows: int) -> float:
    if total_windows <= 0:
        return 100.0
    normal_windows = total_windows - anomalous_windows
    return round((normal_windows / total_windows) * 100, 2)


def classify_status(health_score: float) -> str:
    if health_score >= HEALTHY_MIN:
        return "HEALTHY"
    if health_score >= WARNING_MIN:
        return "WARNING"
    return "CRITICAL"


def compute_channel_health(channel: str) -> Dict:
    """
    Runs the existing RF prediction pipeline for one channel and adds
    health_score / health_status on top of it. Does not duplicate any
    of run_prediction's own logic.
    """
    result = run_prediction(channel)

    health_score = compute_health_score(
        result["total_windows"], result["anomalous_windows"]
    )
    health_status = classify_status(health_score)

    result["health_score"] = health_score
    result["health_status"] = health_status
    return result


def _discover_channels(dataset: NASADataset) -> List[str]:
    """
    Every raw channel name, filtered down to only ones that actually have
    a labeled_anomalies.csv entry (a handful of NASA channels, e.g. T-10,
    T-11, have raw data but no label row - those can't be scored and are
    skipped here rather than raising).
    """
    valid = []
    for ch in dataset.get_channel_names():
        label_rows = dataset.load_labels(ch)
        if not label_rows.empty:
            valid.append(ch)
    return valid


def compute_health_overview(limit: Optional[int] = None) -> Dict:
    """
    Fleet-wide health rollup across every scoreable channel.

    NOTE: this runs the full RF prediction pipeline for every channel it
    includes, synchronously, in one request - that's ~80 channels'
    worth of preprocessing + inference. Use `limit` while iterating on
    the dashboard UI so requests stay fast; drop it for the real number.
    """
    dataset = NASADataset()
    channels = _discover_channels(dataset)

    if limit is not None:
        channels = channels[:limit]

    channel_results = []
    skipped = []

    for ch in channels:
        try:
            channel_results.append(compute_channel_health(ch))
        except Exception as e:
            skipped.append({"channel": ch, "reason": str(e)})

    total_windows_all = sum(r["total_windows"] for r in channel_results)
    total_anomalous_all = sum(r["anomalous_windows"] for r in channel_results)
    overall_health = compute_health_score(total_windows_all, total_anomalous_all)

    spacecraft_groups: Dict[str, Dict] = {}
    for r in channel_results:
        sc = r["spacecraft"]
        group = spacecraft_groups.setdefault(sc, {"total_windows": 0, "anomalous_windows": 0, "channels": 0})
        group["total_windows"] += r["total_windows"]
        group["anomalous_windows"] += r["anomalous_windows"]
        group["channels"] += 1

    spacecraft_summary = {}
    for sc, g in spacecraft_groups.items():
        health = compute_health_score(g["total_windows"], g["anomalous_windows"])
        spacecraft_summary[sc] = {
            "health": health,
            "status": classify_status(health),
            "channels": g["channels"],
            "total_windows": g["total_windows"],
            "anomalous_windows": g["anomalous_windows"],
        }

    return {
        "overall_health": overall_health,
        "overall_status": classify_status(overall_health),
        "total_windows": total_windows_all,
        "total_anomalous_windows": total_anomalous_all,
        "spacecraft": spacecraft_summary,
        "channels": [
            {
                "channel": r["channel"],
                "spacecraft": r["spacecraft"],
                "health": r["health_score"],
                "status": r["health_status"],
                "anomalous_windows": r["anomalous_windows"],
                "total_windows": r["total_windows"],
            }
            for r in channel_results
        ],
        "skipped": skipped,
    }
