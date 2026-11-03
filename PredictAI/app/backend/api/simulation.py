"""
simulation.py (API route)

WebSocket endpoint that streams live_simulation alert events to the
frontend in real time, backed by services/simulation_service.py.

Client protocol:
  1. Connect to ws://<host>/api/simulation/ws
  2. Send one JSON message to start: {"channels": ["A-1", "M-1"], "sleep": 0.02, "max_ticks": null}
  3. Receive a stream of JSON event messages (see simulation_service.simulate for the shapes)
  4. Server closes the socket after sending {"kind": "SIMULATION_COMPLETE", ...}
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from app.backend.services.simulation_service import simulate


router = APIRouter(
    prefix="/api/simulation",
    tags=["Simulation"]
)


@router.websocket("/ws")
async def simulation_ws(websocket: WebSocket):
    await websocket.accept()

    try:
        params = await websocket.receive_json()
        channels = params.get("channels") or ["A-1"]
        sleep = float(params.get("sleep", 0.02))
        max_ticks = params.get("max_ticks")

        async for event in simulate(channels, sleep=sleep, max_ticks=max_ticks):
            await websocket.send_json(event)

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"kind": "ERROR", "message": str(e)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
