from pathlib import Path
import sys


# ---------------------------------------------------------
# Project root
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------
# Existing RAG pipeline
# ---------------------------------------------------------

from src.rag import explain_anomaly


def run_diagnosis(
    channel: str,
    top_k: int = 3
):

    report = explain_anomaly(
        channel=channel,
        top_k=top_k
    )

    return {
        "channel": channel,
        "episodes": report
    }