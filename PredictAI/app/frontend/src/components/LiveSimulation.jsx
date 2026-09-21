import { useEffect, useRef, useState } from "react";

const WS_URL = "ws://127.0.0.1:8000/api/simulation/ws";

// DIAGNOSTIC BUILD v2 - if you don't see "[LiveSimulation v2] component mounted"
// in the browser console when this tab loads, this file is NOT the one running.
function LiveSimulation() {
  console.log("[LiveSimulation v2] component mounted");

  const [channelsInput, setChannelsInput] = useState("A-1, M-1");
  const [running, setRunning] = useState(false);
  const [connected, setConnected] = useState(false);
  const [events, setEvents] = useState([]);
  const [error, setError] = useState(null);

  const socketRef = useRef(null);

  const startSimulation = () => {
    console.log("[LiveSimulation v2] startSimulation called, running=", running, "socketRef=", socketRef.current);

    if (running || socketRef.current) {
      console.log("[LiveSimulation v2] blocked duplicate start");
      return;
    }

    setError(null);
    setEvents([]);

    const channels = channelsInput
      .split(",")
      .map((c) => c.trim())
      .filter(Boolean);

    if (channels.length === 0) {
      setError("Enter at least one channel id (e.g. A-1, M-1)");
      return;
    }

    console.log("[LiveSimulation v2] creating WebSocket to", WS_URL);
    const socket = new WebSocket(WS_URL);
    socketRef.current = socket;

    socket.onopen = () => {
      console.log("[LiveSimulation v2] socket OPEN, sending:", channels);
      setConnected(true);
      setRunning(true);
      socket.send(JSON.stringify({ channels, sleep: 0.05 }));
    };

    socket.onmessage = (msg) => {
      console.log("[LiveSimulation v2] message received:", msg.data);
      try {
        const event = JSON.parse(msg.data);

        if (event.kind === "SIMULATION_COMPLETE" || event.kind === "ERROR") {
          setRunning(false);
        }
        if (event.kind === "ERROR") {
          setError(event.message);
        }

        setEvents((prev) => [event, ...prev]);
      } catch (err) {
        console.error("[LiveSimulation v2] failed to parse event:", err, msg.data);
      }
    };

    socket.onclose = (closeEvent) => {
      console.log(
        `[LiveSimulation v2] socket CLOSED - code=${closeEvent.code} reason="${closeEvent.reason}" wasClean=${closeEvent.wasClean}`
      );
      setConnected(false);
      setRunning(false);
      socketRef.current = null;
    };

    socket.onerror = (err) => {
      console.error("[LiveSimulation v2] socket ERROR event:", err);
      setError("WebSocket connection failed. Is the backend running?");
      setRunning(false);
    };
  };

  const stopSimulation = () => {
    console.log("[LiveSimulation v2] stopSimulation called by user");
    socketRef.current?.close();
  };

  useEffect(() => {
    console.log("[LiveSimulation v2] mount effect ran");
    return () => {
      console.log("[LiveSimulation v2] UNMOUNT cleanup firing, closing socket:", socketRef.current);
      socketRef.current?.close();
    };
  }, []);

  return (
    <>
      <section className="section-heading">
        <div>
          <span className="eyebrow">REAL-TIME MONITORING</span>
          <h2>Live Simulation</h2>
        </div>

        <div className={`model-badge ${connected ? "" : "offline-badge"}`}>
          ● {connected ? "STREAMING" : "IDLE"}
        </div>
      </section>

      <div className="control-panel">
        <div style={{ flex: 1 }}>
          <span className="control-label">CHANNELS TO WATCH (comma-separated)</span>
          <input
            type="text"
            value={channelsInput}
            onChange={(e) => setChannelsInput(e.target.value)}
            disabled={running}
            placeholder="A-1, M-1, S-1"
            style={{
              width: "100%",
              padding: "10px 12px",
              borderRadius: "8px",
              border: "1px solid var(--border, #333)",
              background: "transparent",
              color: "inherit",
            }}
          />
        </div>

        {!running ? (
          <button className="run-button" onClick={startSimulation}>
            START SIMULATION
          </button>
        ) : (
          <button className="run-button danger-button" onClick={stopSimulation}>
            STOP
          </button>
        )}
      </div>

      {error && (
        <div className="error-banner">
          <strong>SIMULATION ERROR</strong>
          <span>{error}</span>
        </div>
      )}

      <div className="alert-feed">
        {events.length === 0 && !running && (
          <p className="empty-feed">No simulation run yet. Enter channels and press Start.</p>
        )}

        {events.map((event, idx) => (
          <AlertEvent key={idx} event={event} />
        ))}
      </div>
    </>
  );
}

function AlertEvent({ event }) {
  if (event.kind === "SIMULATION_STARTED") {
    return (
      <div className="alert-item alert-info">
        <strong>Simulation started</strong> — watching {event.channels.join(", ")}
      </div>
    );
  }

  if (event.kind === "SIMULATION_COMPLETE") {
    return (
      <div className="alert-item alert-info">
        <strong>Simulation complete</strong> — {event.ticks_processed} ticks processed
      </div>
    );
  }

  if (event.kind === "ERROR") {
    return (
      <div className="alert-item alert-error">
        <strong>Error:</strong> {event.message}
      </div>
    );
  }

  if (event.kind === "CHANNEL_SKIPPED") {
    return (
      <div className="alert-item alert-skipped">
        <strong>{event.channel}</strong> skipped — {event.reason}
      </div>
    );
  }

  const ep = event.episode || {};

  if (event.kind === "ALERT_PROVISIONAL") {
    return (
      <div className="alert-item alert-provisional">
        <div className="alert-item-header">
          <span className="pulse-dot" />
          <strong>{event.channel}</strong>
          <span className="alert-spacecraft">({event.spacecraft})</span>
          <span className="alert-tag">PROVISIONAL</span>
        </div>
        <div className="alert-item-body">
          Anomaly confirmed at window {ep.end} — severity avg {ep.avg_severity?.toFixed(3)},
          max {ep.max_severity?.toFixed(3)}. Explanation pending resolution...
        </div>
      </div>
    );
  }

  if (event.kind === "ALERT_RESOLVED") {
    return (
      <div className="alert-item alert-resolved">
        <div className="alert-item-header">
          <strong>{event.channel}</strong>
          <span className="alert-spacecraft">({event.spacecraft})</span>
          <span className="alert-tag resolved-tag">
            {ep.unresolved_at_stream_end ? "STREAM ENDED (still anomalous)" : "RESOLVED"}
          </span>
        </div>
        <div className="alert-item-body">
          Duration ~{ep.real_duration_estimate} timesteps, avg severity {ep.avg_severity?.toFixed(3)}
        </div>

        {event.matches && event.matches.length > 0 && (
          <div className="alert-matches">
            {event.matches.map((m, i) => (
              <div key={i} className="alert-match">
                <span className="match-score">[{m.similarity_score.toFixed(3)}]</span>
                <span className="match-name">{m.name}</span>
                <span className="match-category">({m.category})</span>
                {m.recommended_actions?.[0] && (
                  <div className="match-action">→ {m.recommended_actions[0]}</div>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    );
  }

  return null;
}

export default LiveSimulation;
