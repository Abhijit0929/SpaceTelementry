import { useEffect, useState } from "react";
import { getHealthOverview } from "../services/api";
import StatusCard from "./StatusCard";

// Fleet-wide scoring runs the RF pipeline over every channel synchronously.
// Keep this limited for fast dashboard loading.
const DEFAULT_LIMIT = 10;

// Mission-control style icons
const HealthIcon = () => (
  <svg viewBox="0 0 24 24" aria-hidden="true">
    <path
      d="M12 3v3M12 18v3M3 12h3M18 12h3M5.64 5.64l2.12 2.12M16.24 16.24l2.12 2.12M18.36 5.64l-2.12 2.12M7.76 16.24l-2.12 2.12"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinecap="round"
    />
    <circle
      cx="12"
      cy="12"
      r="4"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
    />
  </svg>
);

const WindowsIcon = () => (
  <svg viewBox="0 0 24 24" aria-hidden="true">
    <rect
      x="3"
      y="3"
      width="7"
      height="7"
      rx="1"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
    />
    <rect
      x="14"
      y="3"
      width="7"
      height="7"
      rx="1"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
    />
    <rect
      x="3"
      y="14"
      width="7"
      height="7"
      rx="1"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
    />
    <rect
      x="14"
      y="14"
      width="7"
      height="7"
      rx="1"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
    />
  </svg>
);

const AnomalyIcon = () => (
  <svg viewBox="0 0 24 24" aria-hidden="true">
    <path
      d="M12 3L22 20H2L12 3Z"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.7"
      strokeLinejoin="round"
    />
    <path
      d="M12 9v5"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
    />
    <circle cx="12" cy="17" r="0.9" fill="currentColor" />
  </svg>
);

const RateIcon = () => (
  <svg viewBox="0 0 24 24" aria-hidden="true">
    <path
      d="M4 17l5-5 3 3 8-9"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
    <path
      d="M15 6h5v5"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
    />
  </svg>
);

function HealthOverview() {
  const [overview, setOverview] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    getHealthOverview(DEFAULT_LIMIT)
      .then(setOverview)
      .catch((err) =>
        setError(
          err.response?.data?.detail ||
            err.message ||
            "Failed to load health overview."
        )
      )
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <section className="health-overview">
        <div className="loading-panel">
          <div className="loader" />
          <p>Calculating fleet telemetry health...</p>
        </div>
      </section>
    );
  }

  if (error) {
    return (
      <section className="health-overview">
        <div className="error-banner">
          <strong>HEALTH OVERVIEW ERROR</strong>
          <span>{error}</span>
        </div>
      </section>
    );
  }

  if (!overview) return null;

  const anomalyRate =
    overview.total_windows > 0
      ? (
          (overview.total_anomalous_windows / overview.total_windows) *
          100
        ).toFixed(2)
      : "0.00";

  const isCritical = overview.overall_status === "CRITICAL";

  return (
    <section className="health-overview">
      {/* Section Header */}
      <section className="section-heading health-heading">
        <div>
          <div className="heading-label">
            <span className="heading-pulse" />
            <span className="eyebrow">MISSION TELEMETRY HEALTH</span>
          </div>

          <h2>Real-time spacecraft anomaly monitoring</h2>

          <p className="section-description">
            Fleet telemetry processed through the PredictAI anomaly detection
            pipeline.
          </p>
        </div>

        <div
          className={`fleet-status ${
            isCritical ? "fleet-critical" : "fleet-operational"
          }`}
        >
          <span className="status-dot" />
          <span>{overview.overall_status || "OPERATIONAL"}</span>
        </div>
      </section>

      {/* Health Cards */}
      <div className="stats-grid health-stats-grid">
        <StatusCard
          icon={<HealthIcon />}
          label="TELEMETRY HEALTH"
          value={`${overview.overall_health}%`}
          description={overview.overall_status}
          danger={isCritical}
        />

        <StatusCard
          icon={<WindowsIcon />}
          label="TOTAL WINDOWS"
          value={overview.total_windows.toLocaleString()}
          description="Telemetry windows analyzed"
        />

        <StatusCard
          icon={<AnomalyIcon />}
          label="ANOMALOUS WINDOWS"
          value={overview.total_anomalous_windows.toLocaleString()}
          description="Detected across scored channels"
          danger={overview.total_anomalous_windows > 0}
        />

        <StatusCard
          icon={<RateIcon />}
          label="ANOMALY RATE"
          value={`${anomalyRate}%`}
          description="Across processed windows"
        />
      </div>

      {/* Skipped Channels */}
      {overview.skipped?.length > 0 && (
        <div className="skipped-channels">
          <div className="skip-icon">
            <AnomalyIcon />
          </div>

          <div>
            <strong>
              {overview.skipped.length} channel
              {overview.skipped.length > 1 ? "s" : ""} skipped
            </strong>

            <span>
              Missing labels:{" "}
              {overview.skipped.map((s) => s.channel).join(", ")}
            </span>
          </div>
        </div>
      )}
    </section>
  );
}

export default HealthOverview;
