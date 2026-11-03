"""
simulation_service.py

Async adapter around src/live_simulation.py's ChannelStream + RAG retrieval,
so the FastAPI WebSocket endpoint can stream alert events to the frontend
as they happen, instead of the CLI's print/sleep loop.

Reuses the exact same detection + debounce + retrieval logic already
validated via `python -m src.live_simulation` - this does not duplicate
that logic, it just wraps it for streaming over a socket.

NOTE: a handful of NASA channels (e.g. T-10, T-11) exist as raw .npy
files but have NO row in labeled_anomalies.csv - a gap in NASA's own
published labels, not a bug here. Those channels are skipped with a
warning event rather than aborting the whole batch.
"""

import asyncio
from pathlib import Path
import sys
from typing import AsyncGenerator, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.dataset import NASADataset
from src.live_simulation import ChannelStream, CONFIRM_WINDOWS
from src.rag import KnowledgeBaseRetriever, describe_episode

# Adjust this import if priority_service.py doesn't live alongside this file
# (spec puts it at app/backend/services/priority_service.py).
from .priority_service import PriorityInput, compute_priority


def _attach_priority(event: Dict) -> Dict:
    """
    Runs the priority engine over an ALERT_PROVISIONAL or ALERT_RESOLVED
    event and attaches the result under event["priority"].

    NOTE: fault_class is intentionally omitted here. The RF model in
    live_simulation.py is a binary anomaly detector (flagged/not-flagged
    against a threshold), not a fault-TYPE classifier. Until a real
    fault-classification step exists, priority is based on severity +
    persistence only - subsystem-criticality weighting in priority_service
    stays neutral (x1.0) rather than guessing a fault class. Wire in
    fault_class here once that classifier exists.

    consecutive_windows is CONFIRM_WINDOWS (not a running counter) because
    both ALERT_PROVISIONAL and ALERT_RESOLVED only ever fire after the
    debounce streak has already reached CONFIRM_WINDOWS - see
    ChannelStream.step() in live_simulation.py.
    """
    episode = event["episode"]
    payload = PriorityInput(
        channel=event["channel"],
        spacecraft=event["spacecraft"],
        severity=episode["max_severity"],
        consecutive_windows=CONFIRM_WINDOWS,
        fault_class=None,
    )
    event = dict(event)
    event["priority"] = compute_priority(payload).to_dict()
    return event


def _serialize_matches(matches: List[Dict]) -> List[Dict]:
    return [
        {
            "name": m["name"],
            "category": m["category"],
            "similarity_score": round(m["similarity_score"], 4),
            "likely_causes": m.get("likely_causes", []),
            "recommended_actions": m.get("recommended_actions", []),
        }
        for m in matches
    ]


async def simulate(
    channels: List[str],
    sleep: float = 0.02,
    max_ticks: Optional[int] = None,
) -> AsyncGenerator[Dict, None]:
    """
    Async generator yielding one event dict per alert-worthy moment:
      - CHANNEL_SKIPPED: this channel couldn't be watched (e.g. missing
        labels), included so the frontend can show why, without aborting
        the rest of the batch
      - ALERT_PROVISIONAL: fired immediately on debounce confirmation, no explanation yet
      - ALERT_RESOLVED: fired once the episode ends (or the stream runs out - flushed,
        flagged with episode.unresolved_at_stream_end=True), includes RAG matches
    A short asyncio.sleep happens after every emitted event (not every tick) so the
    frontend gets a readable pace without waiting through thousands of silent ticks.
    """
    dataset = NASADataset()
    retriever = KnowledgeBaseRetriever()

    streams: List[ChannelStream] = []
    skipped: List[Dict] = []

    for ch in channels:
        try:
            streams.append(ChannelStream(dataset, ch))
        except Exception as e:
            skipped.append({"kind": "CHANNEL_SKIPPED", "channel": ch, "reason": str(e)})
        await asyncio.sleep(0)  # keep the websocket responsive while building many streams

    yield {
        "kind": "SIMULATION_STARTED",
        "channels": [s.channel for s in streams],
        "skipped": [s["channel"] for s in skipped],
        "spacecraft": {s.channel: s.spacecraft for s in streams},
    }

    for skip_event in skipped:
        yield skip_event

    if not streams:
        yield {"kind": "SIMULATION_COMPLETE", "ticks_processed": 0}
        return

    tick = 0

    def build_resolved_event(event: Dict) -> Dict:
        query = describe_episode(event["episode"])
        matches = retriever.retrieve(query, top_k=3)
        event = dict(event)
        event["matches"] = _serialize_matches(matches)
        event = _attach_priority(event)
        return event

    while any(s.has_next() for s in streams):
        if max_ticks is not None and tick >= max_ticks:
            break

        for stream in streams:
            if not stream.has_next():
                continue

            event = stream.step()
            if event is None:
                continue

            if event["kind"] == "ALERT_PROVISIONAL":
                # Priority is computed and sent to the dashboard immediately
                # on confirmation - operators shouldn't wait for RF/RAG to
                # finish before seeing that something is P1.
                yield _attach_priority(event)
                await asyncio.sleep(sleep)

            elif event["kind"] == "ALERT_RESOLVED":
                yield build_resolved_event(event)
                await asyncio.sleep(sleep)

        tick += 1
        # Yield to the event loop EVERY tick, not every N ticks. With the
        # ChannelStream batch-prediction fix this loop body is now just
        # array indexing (fast), so this costs almost nothing - but it's
        # cheap insurance against ever again starving uvicorn's websocket
        # keepalive ping/pong, which is what caused the 1011 "keepalive
        # ping timeout" disconnects before that fix.
        await asyncio.sleep(0)

    for stream in streams:
        flush_event = stream.flush()
        if flush_event is not None:
            yield build_resolved_event(flush_event)
            await asyncio.sleep(sleep)

    yield {"kind": "SIMULATION_COMPLETE", "ticks_processed": tick}