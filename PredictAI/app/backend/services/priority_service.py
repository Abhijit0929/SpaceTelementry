"""
priority_service.py

Priority Engine for PredictAI.

Combines anomaly severity, persistence (debounce/confirmation windows),
fault classification, and subsystem criticality into an operational
priority (P1-P4) for the mission control dashboard.

Design notes (per project spec):
- Priority is NOT the raw RAG similarity score. RAG scores are retrieval
  scores over a failure-mode knowledge base, not physical probabilities.
- Severity thresholds below are OPERATIONAL THRESHOLDS this system
  defines and can tune — not NASA flight rules. Surface them as such
  in the UI/report ("Telemetry Health Score", "Candidate Failure Mode",
  not "confirmed failure").
- Persistence uses a 3-consecutive-window confirmation, matching the
  existing simulation_service.py debounce logic.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Priority(str, Enum):
    P1 = "P1"  # CRITICAL
    P2 = "P2"  # HIGH
    P3 = "P3"  # MEDIUM
    P4 = "P4"  # LOW


class Status(str, Enum):
    SUSPECTED = "INVESTIGATING"
    CONFIRMED = "CONFIRMED"


# Subsystem criticality weights — tune per mission/domain knowledge.
# Unknown fault classes default to 1.0 (neutral).
SUBSYSTEM_CRITICALITY = {
    "power_anomaly": 1.15,
    "power": 1.15,
    "thermal_anomaly": 1.05,
    "thermal": 1.05,
    "attitude_control": 1.10,
    "communications": 1.00,
    "propulsion": 1.10,
    "sensor_drift": 0.90,
}

CONFIRMATION_WINDOWS_REQUIRED = 3


def severity_to_band(severity: float) -> Priority:
    """Map a 0-1 severity score to an operational priority band."""
    if severity >= 0.80:
        return Priority.P1
    elif severity >= 0.60:
        return Priority.P2
    elif severity >= 0.30:
        return Priority.P3
    else:
        return Priority.P4


PRIORITY_LABEL = {
    Priority.P1: "CRITICAL",
    Priority.P2: "HIGH",
    Priority.P3: "MEDIUM",
    Priority.P4: "LOW",
}


@dataclass
class PriorityInput:
    channel: str
    spacecraft: str
    severity: float                 # 0.0 - 1.0, from LSTM autoencoder reconstruction error
    consecutive_windows: int        # how many consecutive windows have flagged anomalous
    fault_class: Optional[str] = None   # from RF classifier, e.g. "power_anomaly"
    fault_confidence: Optional[float] = None  # RF predict_proba, if meaningful for that model
    rag_candidates: int = 0         # number of candidate failure modes retrieved


@dataclass
class PriorityResult:
    channel: str
    spacecraft: str
    priority: Priority
    label: str
    severity: float
    adjusted_severity: float
    status: Status
    reasons: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "channel": self.channel,
            "spacecraft": self.spacecraft,
            "priority": self.priority.value,
            "label": self.label,
            "severity": round(self.severity, 4),
            "adjusted_severity": round(self.adjusted_severity, 4),
            "status": self.status.value,
            "reasons": self.reasons,
        }


def compute_priority(payload: PriorityInput) -> PriorityResult:
    """
    Core priority calculation.

    Priority = f(severity, persistence, fault_class -> subsystem criticality)

    This does NOT use rag_candidates count/score directly in the priority
    number — RAG informs diagnosis, not urgency. It's accepted here only
    so the caller can pass the full event and we can note it in `reasons`
    if useful for the dashboard/audit trail.
    """
    reasons = []

    severity = max(0.0, min(1.0, payload.severity))

    # --- Persistence gate ---
    confirmed = payload.consecutive_windows >= CONFIRMATION_WINDOWS_REQUIRED
    status = Status.CONFIRMED if confirmed else Status.SUSPECTED

    if confirmed:
        reasons.append(
            f"Anomaly persisted across {payload.consecutive_windows} confirmation windows"
        )
    else:
        reasons.append(
            f"Only {payload.consecutive_windows}/{CONFIRMATION_WINDOWS_REQUIRED} "
            "confirmation windows — not yet confirmed"
        )

    # --- Subsystem criticality weighting ---
    weight = SUBSYSTEM_CRITICALITY.get((payload.fault_class or "").lower(), 1.0)
    adjusted_severity = max(0.0, min(1.0, severity * weight))

    if weight != 1.0 and payload.fault_class:
        reasons.append(
            f"{payload.fault_class.replace('_', ' ').title()} weighted "
            f"{'up' if weight > 1.0 else 'down'} (x{weight})"
        )

    if severity >= 0.80:
        reasons.append("High anomaly severity")

    # --- Unconfirmed anomalies are capped below CRITICAL ---
    # A single spike shouldn't page mission control as P1.
    if not confirmed and adjusted_severity >= 0.80:
        adjusted_severity = 0.79
        reasons.append("Capped below CRITICAL pending confirmation")

    priority = severity_to_band(adjusted_severity)

    if payload.fault_class:
        reasons.append(f"Fault classification: {payload.fault_class.replace('_', ' ')}")

    return PriorityResult(
        channel=payload.channel,
        spacecraft=payload.spacecraft,
        priority=priority,
        label=PRIORITY_LABEL[priority],
        severity=severity,
        adjusted_severity=adjusted_severity,
        status=status,
        reasons=reasons,
    )


# --- Optional FastAPI wiring -------------------------------------------------
# Uncomment / adapt if you want this exposed directly as an endpoint rather
# than called in-process from simulation_service.py.
#
# from fastapi import APIRouter
# router = APIRouter()
#
# @router.post("/priority")
# def priority_endpoint(payload: PriorityInput):
#     result = compute_priority(payload)
#     return result.to_dict()


if __name__ == "__main__":
    # Quick smoke test matching the spec's worked example (A-1 / SMAP).
    example = PriorityInput(
        channel="A-1",
        spacecraft="SMAP",
        severity=0.955,
        consecutive_windows=3,
        fault_class="power_anomaly",
        fault_confidence=0.934,
        rag_candidates=3,
    )
    result = compute_priority(example)
    import json
    print(json.dumps(result.to_dict(), indent=2))
