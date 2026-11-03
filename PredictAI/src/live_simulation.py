"""
live_simulation.py

Simulates real-time telemetry monitoring by replaying saved test data
window-by-window (as if it were arriving live), running the trained RF
detector on each window as it "arrives", and automatically triggering a
RAG-explained alert when an anomaly is confirmed.

This is a SIMULATION over saved .npy test data, not a live data feed -
it exists to validate the detect -> alert -> explain loop end-to-end
before any real telemetry source is available.

Key design choices:
  - DEBOUNCED alerting: a single flagged window does not fire an alert.
    Real monitoring systems don't either - one noisy sample shouldn't
    page anyone. An episode must stay flagged for CONFIRM_WINDOWS
    consecutive windows before an alert fires.
  - Multiple channels can be watched CONCURRENTLY, interleaved tick by
    tick, so simultaneous anomalies across channels are visible in the
    same run - this is the groundwork a prioritization/triage layer
    would sit on top of.
  - Every alert (opened AND resolved) is logged to outputs/logs/alerts.log
    as JSON lines, not just printed - a persistent trail to build on.

Run from the project root:
    python -m src.live_simulation --channels A-1 M-1
    python -m src.live_simulation --channels A-1 --sleep 0.01 --max-ticks 3000
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from src.config import Config
from src.dataset import NASADataset
from src.preprocessing import DataPreprocessor, WindowGenerator
from src.rf_features import extract_window_features
from src.rf_model import load_rf_model, load_rf_threshold
from src.rag import KnowledgeBaseRetriever, describe_episode, SHORT_DURATION_TIMESTEPS

CONFIRM_WINDOWS = 3   # consecutive flagged windows required before an alert fires
ALERT_LOG_PATH = Config.LOG_DIR / "alerts.log"


class ChannelStream:
    """
    Holds one channel's pre-featurized test windows and the running
    state of "what's happened so far" as we replay it tick by tick.
    """

    def __init__(self, dataset: NASADataset, channel: str):
        self.channel = channel

        label_rows = dataset.load_labels(channel)
        if label_rows.empty:
            raise ValueError(f"No entry for channel '{channel}' in labeled_anomalies.csv")
        self.spacecraft = label_rows.iloc[0]["spacecraft"]

        train, test, _ = dataset.load_channel(channel)

        preprocessor = DataPreprocessor()
        generator = WindowGenerator()
        preprocessor.fit_transform(train)
        test_scaled = preprocessor.transform(test)
        windows = generator.create_windows(test_scaled)

        self.X = extract_window_features(windows)
        self.model = load_rf_model(filename=f"random_forest_{self.spacecraft}.joblib")
        self.threshold = load_rf_threshold(filename=f"random_forest_{self.spacecraft}.joblib")

        # PERF-CRITICAL: predict all of this channel's probabilities in ONE
        # batch call, not one call per window in the tick loop. RF models
        # loaded with n_jobs=-1 pay real thread-dispatch overhead on every
        # single predict_proba() call - calling it thousands of times in a
        # tight loop blocks the event loop long enough to fail a websocket's
        # keepalive ping (observed as close code 1011 "keepalive ping
        # timeout" in production use). A single batched call is both correct
        # and dramatically faster.
        self.probabilities = self.model.predict_proba(self.X)[:, 1]

        self.cursor = 0                 # next window index to process
        self.n_windows = len(self.X)

        self.flag_streak = 0            # consecutive flagged windows right now
        self.alert_open = False         # has this streak already fired a PROVISIONAL alert?
        self.episode_start = None
        self.episode_probs: List[float] = []
        self.last_prob = 0.0            # previous window's probability (any window, flagged or not)
        self.onset_baseline = 0.0       # probability just BEFORE the current streak began

    def has_next(self) -> bool:
        return self.cursor < self.n_windows

    def flush(self) -> Optional[Dict]:
        """
        Called when the stream runs out of data. If an alert was still open
        (the anomaly never resolved before the data ended), emit a final
        RESOLVED-style event using whatever was seen so far, so it doesn't
        just vanish without ever being explained.
        """
        if not self.alert_open:
            return None

        event = self._make_event("ALERT_RESOLVED", self.n_windows - 1)
        event["episode"]["unresolved_at_stream_end"] = True
        self.alert_open = False
        return event

    def step(self) -> Optional[Dict]:
        """
        Processes exactly one window "arriving". Returns an event dict if
        something alert-worthy happened this tick, else None.
        """
        idx = self.cursor
        self.cursor += 1

        prob = float(self.probabilities[idx])   # precomputed batch, not a live model call
        flagged = prob >= self.threshold

        event = None

        if flagged:
            if self.flag_streak == 0:
                # streak just starting - record what "normal" looked like right before it
                self.onset_baseline = self.last_prob

            self.flag_streak += 1
            self.episode_probs.append(prob)
            if self.episode_start is None:
                self.episode_start = idx

            if self.flag_streak == CONFIRM_WINDOWS and not self.alert_open:
                self.alert_open = True
                event = self._make_event("ALERT_PROVISIONAL", idx)

        else:
            if self.alert_open:
                # episode is DONE now - this is when we know its real shape/duration,
                # so this is the point to do a properly-informed retrieval query.
                event = self._make_event("ALERT_RESOLVED", idx - 1)
            self.flag_streak = 0
            self.alert_open = False
            self.episode_start = None
            self.episode_probs = []

        self.last_prob = prob
        return event

    def _make_event(self, kind: str, end_idx: int) -> Dict:
        probs = np.array(self.episode_probs) if self.episode_probs else np.array([0.0])
        window_duration = (end_idx - self.episode_start + 1) if self.episode_start is not None else 1
        real_duration = max(1, window_duration - (Config.WINDOW_SIZE - 1))
        onset_jump = float(probs[0] - self.onset_baseline) if len(probs) else 0.0

        episode = {
            "start": int(self.episode_start) if self.episode_start is not None else end_idx,
            "end": int(end_idx),
            "window_duration": int(window_duration),
            "real_duration_estimate": int(real_duration),
            "avg_severity": float(probs.mean()),
            "max_severity": float(probs.max()),
            "onset_jump": onset_jump,
        }

        return {
            "kind": kind,
            "channel": self.channel,
            "spacecraft": self.spacecraft,
            "episode": episode,
        }


def log_alert(event: Dict, matches: Optional[List[Dict]] = None):
    ALERT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    record = dict(event)
    record["timestamp"] = datetime.now(timezone.utc).isoformat()
    if matches is not None:
        record["top_matches"] = [
            {"name": m["name"], "category": m["category"], "similarity_score": m["similarity_score"]}
            for m in matches
        ]
    with open(ALERT_LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")


def print_alert_provisional(event: Dict):
    ep = event["episode"]
    print("\n" + "!" * 70)
    print(f"  ANOMALY ALERT (provisional)  |  Channel: {event['channel']} ({event['spacecraft']})")
    print(f"  Confirmed at window {ep['end']}  |  severity avg={ep['avg_severity']:.3f} "
          f"max={ep['max_severity']:.3f}")
    print(f"  Still ongoing - full explanation will follow once the episode resolves.")
    print("!" * 70 + "\n")


def print_alert_resolved(event: Dict, matches: List[Dict]):
    ep = event["episode"]
    print(f"\n[RESOLVED] {event['channel']} ({event['spacecraft']}): anomaly cleared after "
          f"~{ep['real_duration_estimate']} timesteps, avg severity {ep['avg_severity']:.3f}")
    print("  Likely explanation (ranked, NOT certainty):")
    for m in matches:
        print(f"    [{m['similarity_score']:.3f}] {m['name']}  ({m['category']})")
        if m.get("recommended_actions"):
            print(f"        -> first action: {m['recommended_actions'][0]}")
    print()


def run_simulation(channels: List[str], sleep: float = 0.0, max_ticks: Optional[int] = None):
    dataset = NASADataset()
    retriever = KnowledgeBaseRetriever()

    print(f"Loading streams for: {channels}")
    streams = [ChannelStream(dataset, ch) for ch in channels]
    print(f"Streams ready. Watching {len(streams)} channel(s) concurrently.\n")

    tick = 0
    active_alerts = 0

    def handle_event(event: Dict):
        nonlocal active_alerts
        if event["kind"] == "ALERT_PROVISIONAL":
            active_alerts += 1
            print_alert_provisional(event)
            log_alert(event)

        elif event["kind"] == "ALERT_RESOLVED":
            active_alerts = max(0, active_alerts - 1)
            query = describe_episode(event["episode"])
            matches = retriever.retrieve(query, top_k=3)
            if event["episode"].get("unresolved_at_stream_end"):
                print(f"\n[STREAM ENDED - still anomalous] {event['channel']} ({event['spacecraft']}): "
                      f"data ran out while this anomaly was still ongoing.")
            print_alert_resolved(event, matches)
            log_alert(event, matches)

    while any(s.has_next() for s in streams):
        if max_ticks is not None and tick >= max_ticks:
            print(f"[STOP] max_ticks ({max_ticks}) reached.")
            break

        for stream in streams:
            if not stream.has_next():
                continue

            event = stream.step()
            if event is not None:
                handle_event(event)

        tick += 1
        if sleep > 0:
            time.sleep(sleep)

    # Flush any streams whose anomaly was still open when their data ran out.
    for stream in streams:
        flush_event = stream.flush()
        if flush_event is not None:
            handle_event(flush_event)

    print("=" * 70)
    print(f"Simulation complete. {tick} ticks processed. Alerts logged to: {ALERT_LOG_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulated real-time anomaly monitoring over saved test data.")
    parser.add_argument("--channels", nargs="+", default=["A-1"], help="Channel ids to watch concurrently, e.g. --channels A-1 M-1")
    parser.add_argument("--sleep", type=float, default=0.0, help="Seconds to sleep between ticks, for a 'real-time feel' (0 = as fast as possible)")
    parser.add_argument("--max-ticks", type=int, default=None, help="Stop after this many ticks (useful for a quick demo run)")
    args = parser.parse_args()

    run_simulation(args.channels, sleep=args.sleep, max_ticks=args.max_ticks)