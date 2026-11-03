"""
test_priority_integration.py

Throwaway smoke test for the priority-engine wiring in simulation_service.py.
Drives simulate() directly as an async generator - no FastAPI/WebSocket
involved - so a failure here means the integration itself is broken, not
your web layer.

Run from wherever simulation_service.py's package root is, e.g.:
    python -m app.backend.services.test_priority_integration
(adjust the import below to match your actual package path)
"""

import asyncio

from app.backend.services.simulation_service import simulate  # adjust import path if needed


async def main():
    seen_provisional = False
    seen_resolved = False

    async for event in simulate(channels=["A-1"], sleep=0, max_ticks=6000):
        kind = event.get("kind")

        if kind == "ALERT_PROVISIONAL":
            seen_provisional = True
            has_priority = "priority" in event
            has_matches = "matches" in event  # should NOT be here yet
            print(f"[PROVISIONAL] channel={event['channel']} "
                  f"priority_attached={has_priority} matches_attached={has_matches}")
            if has_priority:
                print(f"    -> {event['priority']}")
            assert has_priority, "ALERT_PROVISIONAL missing 'priority' key!"
            assert not has_matches, "RAG matches shouldn't exist yet at PROVISIONAL time"

        elif kind == "ALERT_RESOLVED":
            seen_resolved = True
            has_priority = "priority" in event
            has_matches = "matches" in event
            print(f"[RESOLVED]    channel={event['channel']} "
                  f"priority_attached={has_priority} matches_attached={has_matches}")
            if has_priority:
                print(f"    -> {event['priority']}")
            assert has_priority, "ALERT_RESOLVED missing 'priority' key!"
            assert has_matches, "ALERT_RESOLVED missing RAG 'matches' key!"

        elif kind == "CHANNEL_SKIPPED":
            print(f"[SKIPPED] {event['channel']}: {event['reason']}")

        elif kind == "SIMULATION_COMPLETE":
            print(f"[DONE] {event['ticks_processed']} ticks processed")

    if not seen_provisional:
        print("\nWARNING: no ALERT_PROVISIONAL ever fired - try a channel/max_ticks "
              "combination known to contain an anomaly (A-1 is used in the project docs).")
    else:
        print("\nOK: priority engine is attached to both PROVISIONAL and RESOLVED events.")


if __name__ == "__main__":
    asyncio.run(main())
