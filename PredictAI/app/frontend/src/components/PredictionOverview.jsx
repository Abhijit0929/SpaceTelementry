import StatusCard from "./StatusCard";

function PredictionOverview({ prediction }) {
  if (!prediction) {
    return null;
  }

  return (
    <>
      <section className="section-heading">
        <div>
          <span className="eyebrow">MODEL OUTPUT</span>
          <h2>Prediction Overview</h2>
        </div>

        <div className="model-badge">
          ● {prediction.model}
        </div>
      </section>

      <div className="stats-grid">
        <StatusCard
          label="TOTAL WINDOWS"
          value={prediction.total_windows}
          description="Telemetry windows analyzed"
          icon="▦"
        />

        <StatusCard
          label="ANOMALOUS"
          value={prediction.anomalous_windows}
          description="Detected anomaly windows"
          icon="⚠"
          danger
        />

        <StatusCard
          label="NORMAL"
          value={prediction.normal_windows}
          description="Normal telemetry windows"
          icon="✓"
        />

        <StatusCard
          label="ANOMALY RATE"
          value={`${prediction.anomaly_rate}%`}
          description="Percentage of anomalous windows"
          icon="◉"
          danger
        />
      </div>

      <div className="analysis-grid">
        <div className="panel">
          <div className="panel-header">
            <div>
              <span className="eyebrow">ANOMALY DISTRIBUTION</span>
              <h3>Telemetry Classification</h3>
            </div>
          </div>

          <div className="distribution">
            <div className="distribution-row">
              <div className="distribution-info">
                <span>Normal</span>
                <strong>{prediction.normal_windows}</strong>
              </div>

              <div className="bar">
                <div
                  className="bar-normal"
                  style={{
                    width: `${
                      (prediction.normal_windows /
                        prediction.total_windows) *
                      100
                    }%`,
                  }}
                />
              </div>
            </div>

            <div className="distribution-row">
              <div className="distribution-info">
                <span>Anomalous</span>
                <strong>{prediction.anomalous_windows}</strong>
              </div>

              <div className="bar">
                <div
                  className="bar-danger"
                  style={{
                    width: `${
                      (prediction.anomalous_windows /
                        prediction.total_windows) *
                      100
                    }%`,
                  }}
                />
              </div>
            </div>
          </div>
        </div>

        <div className="panel spacecraft-panel">
          <span className="eyebrow">SPACECRAFT</span>

          <div className="spacecraft-id">
            🛰️
            <span>{prediction.spacecraft}</span>
          </div>

          <div className="info-row">
            <span>CHANNEL</span>
            <strong>{prediction.channel}</strong>
          </div>

          <div className="info-row">
            <span>MODEL</span>
            <strong>{prediction.model}</strong>
          </div>

          <div className="info-row">
            <span>STATUS</span>
            <strong className="healthy-text">OPERATIONAL</strong>
          </div>
        </div>
      </div>
    </>
  );
}

export default PredictionOverview;