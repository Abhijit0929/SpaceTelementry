"""
rag.py

Retrieval-Augmented explanation for detected anomalies.

IMPORTANT - how matching actually works (read before trusting the output):
There is no ground-truth link between a detected anomaly and a knowledge
base entry - NASA's dataset does not label channels by subsystem, and the
knowledge base entries are general domain knowledge, not tied to specific
channel IDs. So this module matches on WHAT THE ANOMALY LOOKS LIKE
(duration, severity, onset sharpness - i.e. its "telemetry signature"),
not what subsystem it belongs to.

Pipeline:
  1. find_anomaly_episodes()   - turn a raw 0/1 prediction array into
                                  contiguous anomaly episodes (start, end,
                                  duration, severity stats).
  2. describe_episode()        - turn one episode into a short text query,
                                  e.g. "point anomaly high severity short
                                  duration abrupt onset".
  3. KnowledgeBaseRetriever    - TF-IDF + cosine similarity over all
                                  knowledge_base/*.json entries' text
                                  (name + telemetry_signature + likely_causes).
  4. explain_anomaly()         - ties it together: loads a channel's saved
                                  RF predictions, finds episodes, retrieves
                                  top-k matches per episode, and returns a
                                  structured, human-readable report.

The similarity scores returned are LEXICAL TEXT-MATCH scores, not a
calibrated diagnostic confidence. Always show multiple ranked candidates,
never just the top one, so a human stays in the loop.

Run from the project root:
    python -m src.rag --channel A-1
"""

import argparse
import json
from pathlib import Path
from typing import List, Dict, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.config import Config

KNOWLEDGE_BASE_DIR = Config.KNOWLEDGE_BASE_DIR

# --- Episode-classification thresholds ---
# Windows overlap (stride=1), so a real anomaly of duration D raw timesteps
# gets flagged across roughly D + WINDOW_SIZE - 1 CONSECUTIVE WINDOWS (every
# window that contains at least one anomalous timestep). We correct for that
# inflation before classifying point vs contextual, or almost everything
# would look "long" regardless of the real underlying anomaly length.
SHORT_DURATION_TIMESTEPS = 15   # corrected (real) duration <= this -> "point" anomaly
SEVERITY_HIGH = 0.7
SEVERITY_MEDIUM = 0.4


# ---------------------------------------------------------------------------
# 1. Knowledge base loading + retrieval
# ---------------------------------------------------------------------------

def load_knowledge_base(kb_dir: Optional[Path] = None) -> List[Dict]:
    """
    Loads every *.json file in the knowledge_base directory and flattens
    all "faults" entries into one list, tagging each with its source file.
    """
    kb_dir = Path(kb_dir) if kb_dir else KNOWLEDGE_BASE_DIR
    entries = []

    for json_path in sorted(kb_dir.glob("*.json")):
        with open(json_path, "r") as f:
            data = json.load(f)

        category = data.get("category", json_path.stem)
        for fault in data.get("faults", []):
            fault = dict(fault)
            fault["category"] = category
            fault["_source_file"] = json_path.name
            entries.append(fault)

    if not entries:
        print(f"[WARN] No knowledge base entries found under {kb_dir}")

    return entries


def _entry_text(entry: Dict) -> str:
    """Text used for matching: name + telemetry_signature + likely_causes."""
    parts = [
        entry.get("name", ""),
        entry.get("telemetry_signature", ""),
        " ".join(entry.get("likely_causes", [])),
    ]
    return " ".join(parts)


class KnowledgeBaseRetriever:
    """
    TF-IDF + cosine similarity retrieval over knowledge base entries.

    This is deliberately simple and transparent (no embeddings, no external
    API calls) - it is a lexical similarity match, and its scores should be
    read as "how much symptom-language overlap", not diagnostic certainty.
    """

    def __init__(self, entries: Optional[List[Dict]] = None):
        self.entries = entries if entries is not None else load_knowledge_base()
        self._texts = [_entry_text(e) for e in self.entries]

        if self._texts:
            self.vectorizer = TfidfVectorizer(stop_words="english")
            self._matrix = self.vectorizer.fit_transform(self._texts)
        else:
            self.vectorizer = None
            self._matrix = None

    def retrieve(self, query: str, top_k: int = 3) -> List[Dict]:
        """
        Returns the top_k knowledge base entries most similar to `query`,
        each annotated with a `similarity_score` (0-1, cosine similarity).
        """
        if self.vectorizer is None:
            return []

        query_vec = self.vectorizer.transform([query])
        scores = cosine_similarity(query_vec, self._matrix).ravel()

        ranked_idx = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in ranked_idx:
            entry = dict(self.entries[idx])
            entry["similarity_score"] = float(scores[idx])
            results.append(entry)

        return results


# ---------------------------------------------------------------------------
# 2. Turning raw predictions into anomaly episodes + text queries
# ---------------------------------------------------------------------------

def find_anomaly_episodes(predictions: np.ndarray, probabilities: np.ndarray) -> List[Dict]:
    """
    Collapses a 0/1 prediction array into contiguous anomaly episodes.
    Returns a list of dicts: start, end, duration, avg_severity, max_severity.
    """
    episodes = []
    in_episode = False
    start = None

    for i, flag in enumerate(predictions):
        if flag == 1 and not in_episode:
            in_episode = True
            start = i
        elif flag == 0 and in_episode:
            in_episode = False
            end = i - 1
            episodes.append(_build_episode(start, end, probabilities))

    if in_episode:
        episodes.append(_build_episode(start, len(predictions) - 1, probabilities))

    return episodes


def _build_episode(start: int, end: int, probabilities: np.ndarray, window_size: int = Config.WINDOW_SIZE) -> Dict:
    window_duration = end - start + 1
    # Correct for stride=1 window overlap inflation (see threshold comment above).
    real_duration = max(1, window_duration - (window_size - 1))

    window_probs = probabilities[start:end + 1]

    onset_prev = probabilities[start - 1] if start > 0 else 0.0
    onset_jump = float(probabilities[start] - onset_prev)

    return {
        "start": int(start),
        "end": int(end),
        "window_duration": int(window_duration),
        "real_duration_estimate": int(real_duration),
        "avg_severity": float(window_probs.mean()),
        "max_severity": float(window_probs.max()),
        "onset_jump": onset_jump,
    }


def describe_episode(episode: Dict) -> str:
    """
    Turns one episode's stats into a short query string for retrieval.

    IMPORTANT: only uses words that actually occur in the knowledge base
    text (telemetry_signature / likely_causes) - a query word the TF-IDF
    vectorizer never saw during fit() is silently ignored, so padding the
    query with vocabulary the knowledge base doesn't use (e.g. "severity",
    "windows") does nothing except dilute the words that DO matter. Key
    classification words are repeated to boost their TF-IDF weight against
    a small (16-entry) knowledge base corpus.
    """
    is_point = episode["real_duration_estimate"] <= SHORT_DURATION_TIMESTEPS

    if is_point:
        # matches wording used across point-anomaly entries: "sudden", "sharp",
        # "step-change", "point anomaly", "spike"
        type_terms = "point anomaly sudden sharp step-change spike " * 3
    else:
        # matches wording used across contextual-anomaly entries: "gradual",
        # "sustained", "drift", "contextual anomaly"
        type_terms = "contextual anomaly gradual sustained drift " * 3

    onset_terms = "sudden sharp abrupt " * 2 if episode["onset_jump"] > 0.3 else "gradual sustained drift " * 2

    return f"{type_terms} {onset_terms}".strip()


# ---------------------------------------------------------------------------
# 3. Full explanation pipeline
# ---------------------------------------------------------------------------

def explain_anomaly(
    channel: str,
    predictions: Optional[np.ndarray] = None,
    probabilities: Optional[np.ndarray] = None,
    spacecraft: Optional[str] = None,
    top_k: int = 3,
    retriever: Optional[KnowledgeBaseRetriever] = None,
) -> List[Dict]:
    """
    Full pipeline: episodes -> query -> retrieval, for one channel.
    If predictions/probabilities are not passed in, loads them from the
    files predict_rf.py already saves (Config.PREDICTION_DIR).
    """
    if predictions is None or probabilities is None:
        pred_path = Config.PREDICTION_DIR / f"{channel}_rf_predictions.npy"
        prob_path = Config.PREDICTION_DIR / f"{channel}_rf_probabilities.npy"

        if not pred_path.exists() or not prob_path.exists():
            print(f"[INFO] No saved predictions for {channel} - running predict_rf.predict_channel() now...")
            from src.predict_rf import predict_channel
            predictions, probabilities = predict_channel(channel)
        else:
            predictions = np.load(pred_path)
            probabilities = np.load(prob_path)

    if retriever is None:
        retriever = KnowledgeBaseRetriever()

    episodes = find_anomaly_episodes(predictions, probabilities)

    report = []
    for episode in episodes:
        query = describe_episode(episode)
        matches = retriever.retrieve(query, top_k=top_k)

        report.append({
            "channel": channel,
            "spacecraft": spacecraft,
            "episode": episode,
            "query": query,
            "matches": matches,
        })

    return report


def print_report(report: List[Dict]):
    if not report:
        print("No anomaly episodes found for this channel.")
        return

    for item in report:
        ep = item["episode"]
        print("=" * 70)
        print(f"Channel: {item['channel']}  |  Windows {ep['start']}-{ep['end']}  "
              f"(window-span {ep['window_duration']}, est. real duration ~{ep['real_duration_estimate']} timesteps)")
        print(f"Severity: avg={ep['avg_severity']:.3f}  max={ep['max_severity']:.3f}  "
              f"onset_jump={ep['onset_jump']:.3f}")
        print(f"Query used for retrieval: \"{item['query']}\"")
        print("-" * 70)
        print("Top matching knowledge base entries (lexical similarity, NOT certainty):")
        for m in item["matches"]:
            print(f"\n  [{m['similarity_score']:.3f}] {m['name']}  ({m['category']})")
            print(f"    Likely causes:")
            for cause in m.get("likely_causes", []):
                print(f"      - {cause}")
            print(f"    Recommended actions:")
            for action in m.get("recommended_actions", []):
                print(f"      - {action}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Explain detected anomalies for one channel via knowledge base retrieval.")
    parser.add_argument("--channel", type=str, default="A-1", help="Channel id, e.g. A-1")
    parser.add_argument("--top_k", type=int, default=3, help="Number of knowledge base matches to show per episode")
    args = parser.parse_args()

    report = explain_anomaly(args.channel, top_k=args.top_k)
    print_report(report)
