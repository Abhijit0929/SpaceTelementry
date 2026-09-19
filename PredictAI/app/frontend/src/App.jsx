import { useEffect, useState } from "react";
import { getHealth, getPredictions } from "./services/api";

function App() {
  const [health, setHealth] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const loadData = async () => {
      try {
        const healthData = await getHealth();
        const predictionData = await getPredictions("C-1");

        setHealth(healthData);
        setPrediction(predictionData);
      } catch (err) {
        console.error(err);
        setError(
          err.response?.data?.detail ||
          err.message ||
          "Failed to connect to PredictAI backend"
        );
      } finally {
        setLoading(false);
      }
    };

    loadData();
  }, []);

  if (loading) {
    return <h1>Connecting to PredictAI...</h1>;
  }

  if (error) {
    return (
      <div>
        <h1>Backend Connection Error</h1>
        <p>{error}</p>
      </div>
    );
  }

  return (
    <div>
      <h1>PredictAI</h1>

      <h2>Backend</h2>

      <p>
        Status: <strong>{health?.status}</strong>
      </p>

      <h2>Random Forest Prediction</h2>

      <p>
        Spacecraft:{" "}
        <strong>{prediction?.data?.spacecraft}</strong>
      </p>

      <p>
        Channel:{" "}
        <strong>{prediction?.data?.channel}</strong>
      </p>

      <p>
        Model:{" "}
        <strong>{prediction?.data?.model}</strong>
      </p>

      <p>
        Total Windows:{" "}
        <strong>{prediction?.data?.total_windows}</strong>
      </p>

      <p>
        Anomalous Windows:{" "}
        <strong>{prediction?.data?.anomalous_windows}</strong>
      </p>

      <p>
        Normal Windows:{" "}
        <strong>{prediction?.data?.normal_windows}</strong>
      </p>

      <p>
        Anomaly Rate:{" "}
        <strong>{prediction?.data?.anomaly_rate}%</strong>
      </p>
    </div>
  );
}

export default App;