import { useEffect, useState } from "react";

import {
  getHealth,
  getPredictions,
} from "./services/api";

import Header from "./components/Header";
import ChannelSelector from "./components/ChannelSelector";
import PredictionOverview from "./components/PredictionOverview";
import LiveSimulation from "./components/LiveSimulation";

function App() {
  const [channel, setChannel] = useState("C-1");
  const [view, setView] = useState("analysis"); // "analysis" | "live"

  const [health, setHealth] = useState(null);
  const [prediction, setPrediction] = useState(null);

  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  const loadPrediction = async () => {
    try {
      setLoading(true);
      setError(null);

      const [healthData, predictionData] = await Promise.all([
        getHealth(),
        getPredictions(channel),
      ]);

      setHealth(healthData);
      setPrediction(predictionData.data);
    } catch (err) {
      console.error(err);

      setError(
        err.response?.data?.detail ||
        err.message ||
        "Unable to connect to PredictAI backend."
      );
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadPrediction();
  }, []);

  return (
    <div className="app">
      <Header healthy={health?.status === "healthy"} />

      <main className="dashboard">

        {error && (
          <div className="error-banner">
            <strong>BACKEND CONNECTION ERROR</strong>
            <span>{error}</span>
          </div>
        )}

        <section className="hero">
          <div>
            <span className="eyebrow">
              SPACECRAFT MONITORING SYSTEM
            </span>

            <h2>
              Telemetry <span>Intelligence</span>
            </h2>

            <p>
              Machine-learning powered spacecraft anomaly
              detection and health monitoring.
            </p>
          </div>

          {prediction && view === "analysis" && (
            <div className="mission-chip">
              <span>ACTIVE CHANNEL</span>
              <strong>{prediction.channel}</strong>
            </div>
          )}
        </section>

        <div className="view-tabs">
          <button
            className={`view-tab ${view === "analysis" ? "active" : ""}`}
            onClick={() => setView("analysis")}
          >
            Single Channel Analysis
          </button>
          <button
            className={`view-tab ${view === "live" ? "active" : ""}`}
            onClick={() => setView("live")}
          >
            Live Simulation
          </button>
        </div>

        {view === "analysis" ? (
          <>
            <ChannelSelector
              channel={channel}
              setChannel={setChannel}
              onRun={loadPrediction}
              loading={loading}
            />

            {loading && !prediction ? (
              <div className="loading-panel">
                <div className="loader" />
                <p>Loading spacecraft telemetry intelligence...</p>
              </div>
            ) : (
              <PredictionOverview prediction={prediction} />
            )}
          </>
        ) : (
          <LiveSimulation />
        )}

      </main>

      <footer>
        <span>PREDICTAI</span>
        <span>SPACECRAFT TELEMETRY INTELLIGENCE</span>
        <span>v1.0</span>
      </footer>
    </div>
  );
}

export default App;
